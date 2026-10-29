"""
GraphRAG logic: read -> chunk -> embed -> store in Neo4j -> extract a knowledge graph with Groq -> answer.

Only these GraphRAG names are used, so other data in Neo4j is never touched:
    (:GraphRAGChunk {id, source, chunk_index, text, embedding, graph_status})
    (:GraphRAGChunk)-[:GRAPHRAG_MENTIONS]->(:GraphRAGEntity {key, name, type})
    (:GraphRAGEntity)-[:GRAPHRAG_RELATED {type}]->(:GraphRAGEntity)
    vector index graphrag_chunk_embeddings (384 dimensions, cosine)
"""
import hashlib
import io
import json
import re
import time
import unicodedata

from docx import Document
from neo4j import GraphDatabase
from pypdf import PdfReader

# Settings used throughout the app
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"   # 384-number embeddings
CHUNK_WORDS = 130       # words per chunk (about 800 characters)
OVERLAP_WORDS = 25      # words shared by neighbouring chunks
TOP_K = 5               # chunks retrieved per question (fewer = shorter prompt = faster answer)
MAX_FACTS = 15          # graph facts sent to the LLM per question
MIN_SCORE = 0.6         # ignore chunks less similar than this (0 to 1)
NOT_FOUND = "I couldn't find the answer in the uploaded documents."

# Columns returned for a relationship (used by the answer evidence and the graph view)
EDGE_COLUMNS = """a.key AS source_key, a.name AS source, a.type AS source_type, r.type AS relation,
                  b.key AS target_key, b.name AS target, b.type AS target_type"""


def read_document(filename, data):
    """Return the text of a PDF, DOCX or TXT file, given the file's bytes."""
    name = filename.lower()
    # The file type is decided by its extension; BytesIO lets the readers use the bytes like a file
    if name.endswith(".pdf"):
        return "\n".join(page.extract_text() or "" for page in PdfReader(io.BytesIO(data)).pages)
    if name.endswith(".docx"):
        return "\n".join(paragraph.text for paragraph in Document(io.BytesIO(data)).paragraphs)
    if name.endswith(".txt"):
        return data.decode("utf-8", errors="replace")
    raise ValueError(f"Unsupported file type: {filename}. Use PDF, DOCX or TXT.")

def split_text(text):
    """Split text into chunks of CHUNK_WORDS words; neighbouring chunks share OVERLAP_WORDS words."""
    words = text.split()
    if not words:
        return []
    # Each chunk starts `step` words after the previous one, so neighbouring chunks overlap
    step = CHUNK_WORDS - OVERLAP_WORDS
    return [" ".join(words[i:i + CHUNK_WORDS]) for i in range(0, len(words), step)]

def embed(model, texts):
    """One 384-number embedding per text."""
    # normalize_embeddings=True gives every vector the same length, which suits cosine similarity
    return model.encode(texts, normalize_embeddings=True).tolist()


def run_query(driver, query, **params):
    """Run a Cypher query and return the rows as dictionaries."""
    # execute_query also retries automatically after short connection problems
    records, _, _ = driver.execute_query(query, params)
    return [record.data() for record in records]


def connect(uri, username, password):
    """Connect to Neo4j and create the GraphRAG constraints and vector index if they are missing."""
    driver = GraphDatabase.driver(uri, auth=(username, password), notifications_min_severity="OFF")
    # Fails here with a clear error if the URI or the credentials are wrong
    driver.verify_connectivity()
    # Uniqueness rules: no two chunks with the same id, no two entities with the same key.
    # IF NOT EXISTS makes these lines safe to run every time the app connects.
    run_query(driver, "CREATE CONSTRAINT graphrag_chunk_id IF NOT EXISTS FOR (c:GraphRAGChunk) REQUIRE c.id IS UNIQUE")
    run_query(driver, "CREATE CONSTRAINT graphrag_entity_key IF NOT EXISTS FOR (e:GraphRAGEntity) REQUIRE e.key IS UNIQUE")
    # Vector index used to find similar chunks: 384 numbers per embedding, compared with cosine similarity
    run_query(driver, """
        CREATE VECTOR INDEX graphrag_chunk_embeddings IF NOT EXISTS FOR (c:GraphRAGChunk) ON c.embedding
        OPTIONS {indexConfig: {`vector.dimensions`: 384, `vector.similarity_function`: 'cosine'}}""")
    for _ in range(60):   # wait up to 2 minutes until the index can be searched
        rows = run_query(driver, "SHOW VECTOR INDEXES YIELD name, state WHERE name = 'graphrag_chunk_embeddings' RETURN state")
        if rows and rows[0]["state"] == "ONLINE":
            return driver
        time.sleep(2)
    raise TimeoutError("The Neo4j vector index is not ready yet. Try again in a minute.")


def store_document(driver, model, filename, text):
    """Chunk, embed and store a document. Already stored chunks are skipped. Returns (chunks, new_chunks)."""
    chunks = split_text(text)
    if not chunks:
        raise ValueError(f"No text found in {filename} (scanned PDFs are not supported).")

    # The ID depends on file name, position and text, so the same chunk is never stored twice
    rows = [{"id": hashlib.sha256(f"{filename}::{i}::{chunk}".encode()).hexdigest(),
             "source": filename, "chunk_index": i, "text": chunk} for i, chunk in enumerate(chunks)]
    # Ask Neo4j which of these chunks it already has, and keep only the new ones
    stored = run_query(driver, "MATCH (c:GraphRAGChunk) WHERE c.id IN $ids RETURN c.id AS id", ids=[r["id"] for r in rows])
    stored_ids = {row["id"] for row in stored}
    new_rows = [row for row in rows if row["id"] not in stored_ids]

    if new_rows:
        # Embed all new chunks in one call and attach each vector to its row
        for row, vector in zip(new_rows, embed(model, [row["text"] for row in new_rows])):
            row["embedding"] = vector
        run_query(driver, """
            UNWIND $rows AS row
            MERGE (c:GraphRAGChunk {id: row.id})
            ON CREATE SET c.graph_status = 'pending'
            SET c += row""", rows=new_rows) 
        # 1. UNWIND $rows AS row takes each dictionary from the new_rows list and processes it individually in Neo4j.
        # 2. MERGE creates a GraphRAGChunk node if its ID doesn't already exist; ON CREATE SET marks new nodes as pending.
        # 3. #It copies the properties from the row dictionary onto the Neo4j node c.
    return len(rows), len(new_rows)


def ask_groq(client, model_name, prompt, stream=False):
    """Send one prompt to the LLM. Returns the reply text, or with stream=True the reply's pieces as they arrive."""
    # Reasoning models (gpt-oss) use fewer tokens, and hit fewer rate limits, when they think briefly
    extra = {"reasoning_effort": "low"} if model_name.startswith("openai/gpt-oss") else {}
    # temperature=0 makes the replies as consistent as possible
    response = client.chat.completions.create(model=model_name, temperature=0, stream=stream,
                                              messages=[{"role": "user", "content": prompt}], **extra)
    if not stream:
        return response.choices[0].message.content or ""
    # Some streamed pieces carry no text (e.g. Azure's first piece), so they are skipped
    return (part.choices[0].delta.content for part in response if part.choices and part.choices[0].delta.content)


def clean_answer(text):
    """Tidy an LLM answer; any "couldn't find the answer" style reply becomes the exact required sentence."""
    text = unicodedata.normalize("NFKC", text).strip()      # turns special spaces (e.g. U+202F) into normal ones
    return NOT_FOUND if "couldn't find the answer" in text.replace("’", "'").lower() else text


# Instructions sent to Groq for every chunk; the chunk's text is added after "Text:"
EXTRACTION_PROMPT = """Extract the important entities and relationships from the text below.
Return ONLY a JSON object like this:
{"entities": [{"name": "Transformer", "type": "Model"}],
 "relationships": [{"source": "Transformer", "relation": "USES", "target": "Self-attention"}]}
Rules: use only facts stated in the text, at most 10 entities and 10 relationships,
and every relationship source/target must also be listed in "entities".

Text:
"""


def clean_name(value):
    """'  Self-Attention. ' -> 'Self-Attention'"""
    # Collapse extra spaces, then remove punctuation and quotes at both ends
    return " ".join(str(value).split()).strip(" .,;:'\"")


def pending_chunks(driver):
    """Chunks whose knowledge graph has not been extracted yet."""
    # Every chunk not marked 'done': new chunks and chunks that failed before (so they are retried)
    return run_query(driver, """
        MATCH (c:GraphRAGChunk) WHERE c.graph_status <> 'done'
        RETURN c.id AS id, c.text AS text ORDER BY c.source, c.chunk_index""")


def extract_graph(driver, client, model_name, chunk):
    """Ask Groq for one chunk's entities and relationships and save them in Neo4j."""
    # Ask Groq, then read the JSON part of its reply (json.loads raises an error if it is malformed)
    reply = ask_groq(client, model_name, EXTRACTION_PROMPT + chunk["text"])
    match = re.search(r"\{.*\}", reply, flags=re.DOTALL)        # the JSON object, even inside ``` fences
    if not match:
        raise ValueError("Groq's reply contained no JSON.")
    data = json.loads(match.group(0))

    # Entities are keyed by their lower-case name, so "OpenAI" and "openai" become one node
    entities = {}
    for item in data.get("entities", []):
        name = clean_name(item.get("name", ""))
        if name:
            entities[name.lower()] = {"key": name.lower(), "name": name, "type": clean_name(item.get("type", "")) or "Concept"}

    # Keep only relationships between two of these entities; "is part of" becomes IS_PART_OF
    relationships = []
    for item in data.get("relationships", []):
        source, target = clean_name(item.get("source", "")).lower(), clean_name(item.get("target", "")).lower()
        relation = re.sub(r"[^A-Z0-9]+", "_", clean_name(item.get("relation", "")).upper()).strip("_")
        if source in entities and target in entities and relation:
            relationships.append({"source": source, "target": target, "type": relation})

    # One query saves everything and marks the chunk done (all or nothing). MERGE prevents duplicates.
    # First FOREACH: create (or reuse) each entity and link the chunk to it with GRAPHRAG_MENTIONS.
    # Second FOREACH: create (or reuse) each GRAPHRAG_RELATED link between two entities.
    run_query(driver, """
        MATCH (c:GraphRAGChunk {id: $id})
        FOREACH (ent IN $entities |
            MERGE (e:GraphRAGEntity {key: ent.key}) ON CREATE SET e.name = ent.name, e.type = ent.type
            MERGE (c)-[:GRAPHRAG_MENTIONS]->(e))
        FOREACH (rel IN $relationships |
            MERGE (a:GraphRAGEntity {key: rel.source})
            MERGE (b:GraphRAGEntity {key: rel.target})
            MERGE (a)-[:GRAPHRAG_RELATED {type: rel.type}]->(b))
        SET c.graph_status = 'done'""", id=chunk["id"], entities=list(entities.values()), relationships=relationships)


# Instructions for answering; {question}, {context} and {facts} are filled in by answer_question()
ANSWER_PROMPT = """Answer the question using ONLY the document excerpts and graph facts below.
Cite the source file name in square brackets, e.g. [report.pdf].
If they do not contain enough information, reply with exactly:
I couldn't find the answer in the uploaded documents.

Question: {question}

Document excerpts:
{context}

Graph facts (from the knowledge graph built from the same documents):
{facts}
"""


def answer_question(driver, model, client, model_name, question, stream=False):
    """Vector search + graph retrieval + grounded answer.
    Returns {answer, sources, chunks, facts}. Without a Groq client the answer is None and Groq is never called.
    With stream=True the answer is a generator of text pieces (pass the full text to clean_answer() afterwards)."""
    # 1. Vector search: the chunks most similar to the question
    #    The score is 0 to 1 (higher = more similar); chunks below MIN_SCORE are left out
    chunks = run_query(driver, """
        CALL db.index.vector.queryNodes('graphrag_chunk_embeddings', $top_k, $vector) YIELD node, score
        WHERE score >= $min_score
        RETURN node.id AS id, node.source AS source, node.chunk_index AS chunk_index, node.text AS text, score""",
        top_k=TOP_K, vector=embed(model, [question])[0], min_score=MIN_SCORE)

    # 2. Graph retrieval: relationships around the entities those chunks mention
    #    DISTINCT avoids repeating a relationship; LIMIT keeps the context short
    facts = run_query(driver, f"""
        MATCH (c:GraphRAGChunk)-[:GRAPHRAG_MENTIONS]->(:GraphRAGEntity)-[r:GRAPHRAG_RELATED]-(:GraphRAGEntity)
        WHERE c.id IN $ids
        WITH DISTINCT r LIMIT $max_facts
        WITH startNode(r) AS a, r, endNode(r) AS b
        RETURN {EDGE_COLUMNS}""", ids=[chunk["id"] for chunk in chunks], max_facts=MAX_FACTS)

    # Collect the evidence; each source file name is listed once
    result = {"answer": None, "sources": sorted({chunk["source"] for chunk in chunks}), "chunks": chunks, "facts": facts}
    # No Groq key: return the evidence only, Groq is never called
    if client is None:
        return result
    if not chunks:                       # nothing relevant: no need to call Groq
        result["answer"] = NOT_FOUND
        return result

    # 3. Groq answers from both kinds of evidence only
    context = "\n\n".join(f"[{chunk['source']}]\n{chunk['text']}" for chunk in chunks)
    fact_lines = "\n".join(f"- {f['source']} -[{f['relation']}]-> {f['target']}" for f in facts) or "(none)"
    prompt = ANSWER_PROMPT.format(question=question, context=context, facts=fact_lines)
    if stream:      # the app shows the pieces as they arrive, then cleans the full text
        result["answer"] = ask_groq(client, model_name, prompt, stream=True)
    else:
        result["answer"] = clean_answer(ask_groq(client, model_name, prompt))
    return result


def list_documents(driver):
    """Stored documents with their number of chunks and how many already have their graph."""
    return run_query(driver, """
        MATCH (c:GraphRAGChunk)
        RETURN c.source AS document, count(*) AS chunks,
               sum(CASE WHEN c.graph_status = 'done' THEN 1 ELSE 0 END) AS graph_done
        ORDER BY document""")


def graph_stats(driver):
    """Number of GraphRAG chunks, entities and relationships."""
    return run_query(driver, """
        RETURN COUNT { (:GraphRAGChunk) } AS chunks, COUNT { (:GraphRAGEntity) } AS entities,
               COUNT { (:GraphRAGEntity)-[:GRAPHRAG_RELATED]->(:GraphRAGEntity) } AS relationships""")[0]


def graph_edges(driver, search="", limit=150):
    """Relationships to draw, optionally only those touching an entity whose name contains `search`."""
    # An empty search matches every relationship (up to `limit`); the name match ignores upper/lower case
    return run_query(driver, f"""
        MATCH (a:GraphRAGEntity)-[r:GRAPHRAG_RELATED]->(b:GraphRAGEntity)
        WHERE $search = '' OR toLower(a.name) CONTAINS toLower($search) OR toLower(b.name) CONTAINS toLower($search)
        RETURN {EDGE_COLUMNS} LIMIT $limit""", search=search.strip(), limit=limit)


def delete_document(driver, source):
    """Delete a document's chunks, then the GraphRAG entities no other chunk mentions."""
    # DETACH DELETE also removes the relationships attached to the deleted nodes.
    # Only GraphRAG labels are matched, so no other data in Neo4j is affected.
    run_query(driver, "MATCH (c:GraphRAGChunk {source: $source}) DETACH DELETE c", source=source)
    run_query(driver, "MATCH (e:GraphRAGEntity) WHERE NOT (e)<-[:GRAPHRAG_MENTIONS]-() DETACH DELETE e")
