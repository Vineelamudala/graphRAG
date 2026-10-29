"""
GraphRAG web app: upload documents, build a knowledge graph in Neo4j with Groq, and ask questions.
Run with:  streamlit run app.py      (the GraphRAG logic is in rag_core.py)
"""
import os
import sys

import streamlit as st
import streamlit.components.v1 as components
from dotenv import find_dotenv, load_dotenv
from groq import Groq
from pyvis.network import Network
from sentence_transformers import SentenceTransformer

# Our own module with all the GraphRAG logic (reading files, Neo4j, Groq)
import rag_core as rag

# Started with "python app.py" (e.g. VS Code's Run button)? Relaunch it the right way: "streamlit run app.py".
if __name__ == "__main__" and not st.runtime.exists():
    from streamlit.web import cli
    sys.argv = ["streamlit", "run", __file__, "--server.fileWatcherType", "none"]   # see .streamlit/config.toml
    sys.exit(cli.main())

st.set_page_config(page_title="GraphRAG", page_icon="🕸️", layout="wide")
load_dotenv(find_dotenv(usecwd=True))   # optional .env file with credentials


# Streamlit re-runs this whole file after every click. @st.cache_resource creates these objects
# only once and reuses them, so the model isn't reloaded and Neo4j isn't reconnected each time.
@st.cache_resource(show_spinner="Loading the embedding model...")
def load_model():
    return SentenceTransformer(rag.EMBEDDING_MODEL)


# One cached connection per set of credentials (it also creates the GraphRAG index if needed)
@st.cache_resource(show_spinner="Connecting to Neo4j...")
def get_driver(uri, username, password):
    return rag.connect(uri, username, password)


# Cached Neo4j reads: they run again only after documents change (refresh_cache), not on every click.
# The leading "_" tells Streamlit not to use the driver itself as part of the cache key.
@st.cache_data(show_spinner=False)
def stored_documents(_driver, uri):
    return rag.list_documents(_driver)


@st.cache_data(show_spinner=False)
def graph_overview(_driver, uri, search, limit):
    return rag.graph_stats(_driver), rag.graph_edges(_driver, search, limit)


def refresh_cache():
    stored_documents.clear()
    graph_overview.clear()


def credential_input(label, env_name, secret=False):
    """Sidebar text box. When left empty, the .env value is used; that value is never shown in the page."""
    from_env = os.getenv(env_name, "")
    # The box only shows a placeholder text; the real .env value stays on the server
    typed = st.sidebar.text_input(label, type="password" if secret else "default",
                                  placeholder="Using the value from .env" if from_env else "")
    return typed.strip() or from_env


# Clear messages for common errors: Neo4j errors by their class name, Groq errors by HTTP status code
ERROR_MESSAGES = {
    "AuthError": "Neo4j rejected the username or password.",
    "ServiceUnavailable": "Could not reach Neo4j. Check the URI (Aura: neo4j+s://...) and that the instance is running.",
    "ConfigurationError": "The Neo4j URI is invalid. Example: neo4j+s://xxxxxxxx.databases.neo4j.io",
    "APIConnectionError": "Could not reach the LLM (Groq or LiteLLM gateway). Check the connection or the gateway URL.",
    401: "The LLM API key was rejected.",
    404: "The LLM doesn't know this model name.",
    429: "The LLM's rate limit or daily quota was reached. Wait a little and try again.",
}


def friendly_error(error):
    """A clear message for common problems (never includes credentials)."""
    # Look up the Groq status code first, then the error's class name
    message = ERROR_MESSAGES.get(getattr(error, "status_code", None)) or ERROR_MESSAGES.get(type(error).__name__)
    if message:
        return message
    # ValueErrors from rag_core (e.g. unsupported file, no text found) already have a readable message
    return str(error) if isinstance(error, ValueError) else f"{type(error).__name__}: {error}"


# Node colors for the graph view (one color per entity type)
COLORS = ["#4C78A8", "#F58518", "#54A24B", "#E45756", "#72B7B2", "#B279A2", "#FF9DA6", "#9D755D"]


def draw_graph(edges, height=600):
    """Interactive network: drag nodes, zoom, hover for details. Same entity type = same color."""
    net = Network(height=f"{height}px", width="100%", directed=True, cdn_resources="remote")
    # For each relationship: add both entities as nodes (pyvis skips a node that already exists),
    # then the arrow between them. The color is calculated from the type name, so a type always gets the same color.
    for e in edges:
        for key, name, kind in [(e["source_key"], e["source"], e["source_type"]), (e["target_key"], e["target"], e["target_type"])]:
            net.add_node(key, label=name, title=f"{name} ({kind})", color=COLORS[sum(map(ord, kind or "")) % len(COLORS)])
        net.add_edge(e["source_key"], e["target_key"], label=e["relation"], font={"size": 10})
    # Spread the nodes apart, then show the generated HTML page inside the app
    net.repulsion(node_distance=180, spring_length=180)
    components.html(net.generate_html(), height=height + 30)


def build_graph(driver, client, model_name):
    """Extract entities and relationships for every pending chunk, with a progress bar."""
    chunks = rag.pending_chunks(driver)
    if not chunks:
        st.info("Every chunk already has its knowledge graph.")
        return
    progress = st.progress(0.0, text="Extracting entities and relationships...")
    failed = 0
    # One Groq call per chunk; the progress bar moves after each chunk
    for number, chunk in enumerate(chunks, start=1):
        try:
            rag.extract_graph(driver, client, model_name, chunk)
        except Exception as error:
            if getattr(error, "status_code", None):   # Groq API error (key, model, quota): it would repeat for every chunk
                st.warning(f"{friendly_error(error)} Stopped; the remaining chunks are processed next time.")
                return
            failed += 1                                # e.g. a malformed reply: the chunk stays pending and is retried
        progress.progress(number / len(chunks), text=f"Chunk {number} of {len(chunks)}")
    st.success(f"Graph extraction finished. {failed} chunks failed and will be retried next time.")


# ----------------------------------------------------------------------------- Sidebar
# Credentials are typed here, or taken from .env when a field is left empty
st.sidebar.header("🔑 Connection settings")
st.sidebar.caption("Credentials are only used while the app runs. They are never saved or displayed.")
neo4j_uri = credential_input("Neo4j URI", "NEO4J_URI")
neo4j_user = credential_input("Neo4j username", "NEO4J_USERNAME")
neo4j_password = credential_input("Neo4j password", "NEO4J_PASSWORD", secret=True)

# Connect (or reuse the cached connection) and show how much GraphRAG data is stored
if st.sidebar.button("Test Neo4j connection"):
    try:
        stats = rag.graph_stats(get_driver(neo4j_uri, neo4j_user, neo4j_password))
        st.sidebar.success(f"Connected ✅ ({stats['chunks']} chunks, {stats['entities']} entities stored)")
    except Exception as error:
        st.sidebar.error(friendly_error(error))

st.sidebar.divider()
# LLM: choose Groq or a LiteLLM gateway. Each choice asks for its own settings.
provider = st.sidebar.radio("LLM provider", ["Groq", "LiteLLM"], horizontal=True)
if provider == "LiteLLM":
    litellm_url = st.sidebar.text_input("LiteLLM gateway URL", value=os.getenv("LITELLM_BASE_URL") or "http://localhost:4000")
    groq_key = credential_input("LiteLLM API key", "LITELLM_API_KEY", secret=True)
    groq_model = st.sidebar.text_input("LiteLLM model name", value=os.getenv("LLM_MODEL") or "")
else:
    groq_key = credential_input("Groq API key", "GROQ_API_KEY", secret=True)
    groq_model = st.sidebar.text_input("Groq model", value="openai/gpt-oss-20b")

# No key or model = no LLM client, so no LLM is ever called. max_retries waits out short rate limits.
groq_client = None
if groq_key and groq_model and provider == "Groq":
    groq_client = Groq(api_key=groq_key, max_retries=5)
elif groq_key and groq_model and provider == "LiteLLM" and litellm_url:
    try:
        from openai import OpenAI   # the LiteLLM gateway speaks the OpenAI API
        groq_client = OpenAI(base_url=litellm_url, api_key=groq_key, max_retries=5)
    except ImportError:
        st.sidebar.error("LiteLLM needs the openai package: pip install openai")

if st.sidebar.button(f"Test {provider} key", disabled=groq_client is None):
    try:
        groq_client.models.list()       # free call that only checks the key
        st.sidebar.success(f"{provider} key works ✅")
    except Exception as error:
        st.sidebar.error(friendly_error(error))

# ----------------------------------------------------------------------------- Main page
st.title("🕸️ GraphRAG: chat with your documents")

# Everything below needs Neo4j: st.stop() ends the page here until we can connect
if not (neo4j_uri and neo4j_user and neo4j_password):
    st.info("👈 Enter your Neo4j URI, username and password in the sidebar to start.")
    st.stop()
try:
    driver = get_driver(neo4j_uri, neo4j_user, neo4j_password)
except Exception as error:
    st.error(friendly_error(error))
    st.stop()

model = load_model()
if groq_client is None:
    st.warning(f"No {provider} API key: documents can be uploaded and searched, but graph extraction and answers need a key.")

# Three tabs: upload documents, ask questions, explore the knowledge graph
tab_docs, tab_ask, tab_graph = st.tabs(["📄 Documents", "💬 Ask questions", "🕸️ Knowledge graph"])

with tab_docs:
    files = st.file_uploader("Upload PDF, DOCX or TXT files", type=["pdf", "docx", "txt"], accept_multiple_files=True)
    extract = st.checkbox(f"Also extract entities and relationships with {provider}",
                          value=groq_client is not None, disabled=groq_client is None)
    if st.button("Process documents", type="primary", disabled=not files):
        # Read, chunk, embed and store each file; an error in one file doesn't stop the others
        for file in files:
            try:
                with st.spinner(f"Processing {file.name}..."):
                    total, new = rag.store_document(driver, model, file.name, rag.read_document(file.name, file.getvalue()))
                st.write(f"✅ **{file.name}**: {total} chunks ({new} new)")
            except Exception as error:
                st.error(f"{file.name}: {friendly_error(error)}")
        # Then build the knowledge graph for all chunks that don't have one yet
        if extract:
            build_graph(driver, groq_client, groq_model)
        refresh_cache()

    st.subheader("Stored documents")
    documents = stored_documents(driver, neo4j_uri)
    if not documents:
        st.info("No documents yet.")
    else:
        st.dataframe(documents, hide_index=True)
        # Chunks still without a graph (new ones, or ones that failed earlier)
        remaining = sum(doc["chunks"] - doc["graph_done"] for doc in documents)
        if remaining and groq_client and st.button(f"Extract graph for remaining chunks ({remaining})"):
            build_graph(driver, groq_client, groq_model)
            refresh_cache()

        with st.expander("Delete a document"):
            to_delete = st.selectbox("Document", [doc["document"] for doc in documents])
            # The Delete button only works after the checkbox is ticked; st.rerun() then refreshes the page
            if st.button("Delete", disabled=not st.checkbox("Yes, delete it and the entities only it mentions")):
                rag.delete_document(driver, to_delete)
                refresh_cache()
                st.rerun()

with tab_ask:
    question = st.text_input("Your question", placeholder="e.g. Who co-authored AlexNet and later co-founded OpenAI?")
    if st.button("Ask", type="primary", disabled=not question.strip()):
        try:
            with st.spinner("Searching the documents and the knowledge graph..."):
                result = rag.answer_question(driver, model, groq_client, groq_model, question, stream=True)
        except Exception as error:
            st.error(friendly_error(error))
        else:
            # The answer is None when there is no Groq key; then only the passages are shown
            if result["answer"] is None:
                st.warning(f"Answers need a {provider} API key. Here are the most relevant passages instead.")
            else:
                st.markdown("#### Answer")
                answer_box = st.empty()
                try:
                    # Show the answer word by word as the LLM writes it (it is plain text when nothing was found)
                    answer = result["answer"] if isinstance(result["answer"], str) else answer_box.write_stream(result["answer"])
                    answer_box.markdown(rag.clean_answer(answer))
                except Exception as error:
                    answer_box.error(friendly_error(error))
            st.markdown("**Sources:** " + (", ".join(result["sources"]) or "none"))

            # The evidence behind the answer: the retrieved chunks and the graph facts
            with st.expander(f"📑 Retrieved passages ({len(result['chunks'])})", expanded=result["answer"] is None):
                for hit in result["chunks"]:
                    st.markdown(f"**{hit['source']}**, chunk {hit['chunk_index']} (similarity {hit['score']:.2f})")
                    st.caption(hit["text"])
            with st.expander(f"🕸️ Graph facts used ({len(result['facts'])})"):
                for fact in result["facts"]:
                    st.write(f"{fact['source']} → **{fact['relation']}** → {fact['target']}")
                if result["facts"]:
                    draw_graph(result["facts"], height=400)

with tab_graph:
    # Drawn only when switched on: otherwise every click elsewhere (e.g. "Ask") would redraw the whole graph
    if st.toggle("Show the knowledge graph"):
        search = st.text_input("Only show entities whose name contains")
        limit = st.slider("Maximum relationships to draw", 20, 500, 150, step=10)
        stats, edges = graph_overview(driver, neo4j_uri, search, limit)
        col1, col2, col3 = st.columns(3)
        col1.metric("Chunks", stats["chunks"])
        col2.metric("Entities", stats["entities"])
        col3.metric("Relationships", stats["relationships"])
        if edges:
            draw_graph(edges)
        else:
            st.info(f"Nothing to show yet. Upload documents and extract the graph (needs a {provider} key).")
