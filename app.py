"""
app.py - Streamlit Chatbot Interface for PDF Question Answering with RAG

End-to-End Workflow:
PDF -> Extract Text -> Chunk -> Embed -> Vector Store -> Retrieve Relevant Chunks -> Context + Question -> LLM -> Answer -> Streamlit Chat UI
"""

import os
import streamlit as st
from dotenv import load_dotenv

import rag_engine
from sample_data import generate_sample_pdf

load_dotenv()

# ==============================================================================
# Page Configuration & Styling
# ==============================================================================
st.set_page_config(
    page_title="PDF Chatbot Agent (RAG)",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    /* Clean layout styling */
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1E293B;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #475569;
        margin-bottom: 1.2rem;
    }
    .workflow-pill {
        display: inline-block;
        background: #EEF2FF;
        color: #4338CA;
        padding: 4px 10px;
        border-radius: 6px;
        font-size: 0.85rem;
        font-weight: 600;
        margin: 2px;
        border: 1px solid #C7D2FE;
    }
    .source-box {
        background-color: #F8FAFC;
        border-left: 3px solid #3B82F6;
        padding: 8px 12px;
        border-radius: 4px;
        font-size: 0.88rem;
        margin-bottom: 8px;
    }
    .metric-card {
        background: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-radius: 8px;
        padding: 10px 14px;
        margin-bottom: 8px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ==============================================================================
# Session State Initialization
# ==============================================================================
if "messages" not in st.session_state:
    st.session_state.messages = []

if "vector_store" not in st.session_state:
    st.session_state.vector_store = None

if "doc_stats" not in st.session_state:
    st.session_state.doc_stats = {
        "filename": None,
        "page_count": 0,
        "chunk_count": 0,
        "status": "No document loaded",
    }

if "last_processed_file" not in st.session_state:
    st.session_state.last_processed_file = None


# ==============================================================================
# Helper Functions for Indexing
# ==============================================================================
def process_and_index_pdf(file_source, filename, chunk_size, chunk_overlap):
    """Executes Stages 2, 3, 4, and 5 of the pipeline."""
    with st.spinner(f"Extracting and indexing '{filename}'..."):
        # Step 2: Extract Text
        docs = rag_engine.load_pdf_documents(file_source, filename=filename)

        # Step 3: Chunk Text
        chunks = rag_engine.chunk_documents(docs, chunk_size=chunk_size, chunk_overlap=chunk_overlap)

        # Step 4 & 5: Embed & Store in FAISS
        embeddings = rag_engine.get_embeddings()
        vector_store = rag_engine.create_vector_store(chunks, embeddings=embeddings)

        st.session_state.vector_store = vector_store
        st.session_state.doc_stats = {
            "filename": filename,
            "page_count": len(docs),
            "chunk_count": len(chunks),
            "status": "Indexed & Ready",
        }
        st.session_state.last_processed_file = filename


# ==============================================================================
# Sidebar UI: Upload, Parameters & Vector Store Stats
# ==============================================================================
with st.sidebar:
    st.markdown("### 📄 Document Ingestion")

    # Step 1: Upload / Select PDF
    uploaded_file = st.file_uploader(
        "Upload a PDF Document",
        type=["pdf"],
        help="Upload any text-based PDF to chat with.",
    )

    col_btn1, col_btn2, col_btn3 = st.columns(3)
    with col_btn1:
        load_sample = st.button("⚡ Sample PDF", help="Load built-in AI & RAG Handbook for quick testing", use_container_width=True)
    with col_btn2:
        clear_chat = st.button("🧹 Clear Chat", help="Clear conversation history", use_container_width=True)
    with col_btn3:
        reset_all = st.button("🔄 Reset All", help="Reset chat and knowledge base", use_container_width=True)

    if clear_chat:
        st.session_state.messages = []
        st.rerun()

    if reset_all:
        st.session_state.messages = []
        st.session_state.vector_store = None
        st.session_state.doc_stats = {
            "filename": None,
            "page_count": 0,
            "chunk_count": 0,
            "status": "No document loaded",
        }
        st.session_state.last_processed_file = None
        st.session_state.raw_doc_source = None
        st.rerun()

    st.markdown("---")
    st.markdown("### ⚙️ Model & Retrieval Settings")

    # LLM Provider selection
    provider = st.selectbox(
        "LLM Provider",
        options=["OpenAI", "DeepSeek"],
        index=0,
        help="Choose the model provider for grounded answer generation.",
    )

    if provider == "OpenAI":
        model_name = st.selectbox("OpenAI Model", ["gpt-4o-mini", "gpt-4o"], index=0)
    else:
        model_name = st.selectbox("DeepSeek Model", ["deepseek-chat"], index=0)

    temperature = st.slider(
        "Temperature",
        min_value=0.0,
        max_value=1.0,
        value=0.2,
        step=0.05,
        help="Lower values yield more factual and grounded responses.",
    )

    top_k = st.slider(
        "Top-K Chunks to Retrieve",
        min_value=1,
        max_value=8,
        value=4,
        step=1,
        help="Number of most similar context chunks passed to the LLM.",
    )

    with st.expander("🛠️ Advanced Chunking Parameters"):
        chunk_size = st.slider("Chunk Size (characters)", 400, 2000, 1000, 100)
        chunk_overlap = st.slider("Chunk Overlap (characters)", 50, 400, 200, 25)
        reindex_btn = st.button("🔄 Re-Index with New Chunk Settings", use_container_width=True)

    st.markdown("---")
    st.markdown("### 📊 Knowledge Base Status")
    stats = st.session_state.doc_stats
    st.markdown(
        f"""
        <div class="metric-card">
            <div><b>Active PDF:</b> {stats['filename'] or 'None'}</div>
            <div><b>Total Pages:</b> {stats['page_count']}</div>
            <div><b>Indexed Chunks:</b> {stats['chunk_count']}</div>
            <div><b>Vector Store:</b> <code>{stats['status']}</code></div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # API Key check notifications
    openai_key_set = bool(os.getenv("OPENAI_API_KEY"))
    deepseek_key_set = bool(os.getenv("DEEPSEEK_API_KEY"))
    if not openai_key_set:
        st.warning("⚠️ OPENAI_API_KEY is not set in .env. Required for embeddings & OpenAI.")
    if provider == "DeepSeek" and not deepseek_key_set:
        st.warning("⚠️ DEEPSEEK_API_KEY is not set in .env.")


# ==============================================================================
# Document Ingestion Trigger
# ==============================================================================
if load_sample:
    sample_path = generate_sample_pdf()
    try:
        process_and_index_pdf(
            sample_path,
            filename="quantum_and_ai_handbook.pdf",
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )
        st.sidebar.success("Sample PDF loaded and indexed in FAISS!")
    except Exception as e:
        st.sidebar.error(f"Error loading sample PDF: {e}")

elif uploaded_file is not None:
    if st.session_state.last_processed_file != uploaded_file.name or reindex_btn:
        try:
            process_and_index_pdf(
                uploaded_file,
                filename=uploaded_file.name,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
            )
            st.sidebar.success(f"'{uploaded_file.name}' indexed successfully!")
        except Exception as e:
            st.sidebar.error(f"Error processing PDF: {e}")

elif reindex_btn and st.session_state.doc_stats["filename"] == "quantum_and_ai_handbook.pdf":
    sample_path = generate_sample_pdf()
    try:
        process_and_index_pdf(
            sample_path,
            filename="quantum_and_ai_handbook.pdf",
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )
        st.sidebar.success("Sample PDF re-indexed with new chunk settings!")
    except Exception as e:
        st.sidebar.error(f"Error re-indexing: {e}")


# ==============================================================================
# Main View: Header & Pipeline Workflow
# ==============================================================================
st.markdown('<div class="main-header">📄 AI-Powered PDF Chatbot</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-header">Grounded Question Answering with Document Loading, Chunking, Embeddings, FAISS Vector Search & LLMs</div>',
    unsafe_allow_html=True,
)

with st.expander("ℹ️ How This Agent Works: 10-Stage End-to-End RAG Pipeline", expanded=False):
    st.markdown(
        """
        | Step | Stage | What Happens |
        | :--- | :--- | :--- |
        | **1** | **Upload / Select PDF** | User supplies a PDF via file upload or selects the built-in sample document. |
        | **2** | **Extract Text** | `PyPDFLoader` loads the PDF pages and extracts raw text streams with page metadata. |
        | **3** | **Chunk the Text** | `RecursiveCharacterTextSplitter` divides text into overlapping chunks to preserve contextual boundaries. |
        | **4** | **Generate Embeddings** | OpenAI's `text-embedding-3-small` creates 1,536-dimensional dense vector representations. |
        | **5** | **Store in Vector DB** | Vectors and chunk texts are indexed in an in-memory **FAISS** vector database. |
        | **6** | **Accept User Question** | Natural language queries are captured through the Streamlit chat input. |
        | **7** | **Retrieve Relevant Chunks** | FAISS semantic similarity search extracts the top-k most relevant chunks. |
        | **8** | **Build Grounded Prompt** | Prompt engineering binds retrieved context, conversation history, and anti-hallucination rules. |
        | **9** | **Generate Answer with LLM** | LLM (`OpenAI` or `DeepSeek`) streams a factual answer citing document page numbers. |
        | **10** | **Display in Chat UI** | Streamlit renders the response with collapsible sources and preserves session history. |
        
        **Pipeline Flow:**
        `PDF` ➔ `Extract Text` ➔ `Chunk` ➔ `Embed` ➔ `Vector Store` ➔ `Retrieve Chunks` ➔ `Context + Question` ➔ `LLM` ➔ `Answer` ➔ `Streamlit UI`
        """
    )

# If vector store is not ready, display an onboarding banner
if st.session_state.vector_store is None:
    st.info("👈 **Get Started:** Please upload a PDF document from the sidebar or click **⚡ Sample PDF** to test instantly with the built-in AI & RAG Handbook.")

# Quick sample questions if sample document is active
if st.session_state.doc_stats["filename"] == "quantum_and_ai_handbook.pdf":
    st.markdown("**Suggested Questions for the Sample Document:**")
    quick_cols = st.columns(3)
    sample_prompts = [
        "What is the recommended chunk size and overlap?",
        "What is FAISS and what index structures does it support?",
        "How does RAG prevent hallucinations?",
    ]
    for col, prompt_text in zip(quick_cols, sample_prompts):
        if col.button(f"💬 {prompt_text}", use_container_width=True):
            st.session_state.pending_prompt = prompt_text
            st.rerun()


# ==============================================================================
# Step 10: Display Chat Messages & Preserve Conversation State
# ==============================================================================
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        # If there are sources attached to the assistant message, display them
        if msg.get("sources"):
            with st.expander(f"🔍 Retrieved Sources & Context ({len(msg['sources'])} chunks)", expanded=False):
                for idx, src in enumerate(msg["sources"], 1):
                    st.markdown(
                        f"""
                        <div class="source-box">
                            <b>Chunk #{src.get('chunk_index', idx)}</b> | 
                            <b>Document:</b> <code>{src.get('source', 'PDF')}</code> | 
                            <b>Page:</b> {src.get('page', 'N/A')} | 
                            <b>Similarity Score:</b> {src.get('similarity_score', 0)}%
                            <p style="margin-top: 5px; color: #334155; font-style: italic;">
                                "{src.get('content', '')[:300]}..."
                            </p>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

# ==============================================================================
# Steps 6 -> 7 -> 8 -> 9 -> 10: Chat Input & Retrieval-Augmented Generation
# ==============================================================================
# Always render the chat input at the bottom of the page
chat_input_val = st.chat_input("Ask a question about the uploaded PDF...")

# Determine effective query (either from suggestion click or typed input)
user_query = None
if "pending_prompt" in st.session_state and st.session_state.pending_prompt:
    user_query = st.session_state.pending_prompt
    st.session_state.pending_prompt = None
elif chat_input_val:
    user_query = chat_input_val.strip()

if user_query:
    if st.session_state.vector_store is None:
        st.error("Please upload and index a PDF document before asking questions.")
    else:
        # Append User Message to State
        st.session_state.messages.append({"role": "user", "content": user_query})

        # Display User Message
        with st.chat_message("user"):
            st.markdown(user_query)

        # Assistant Execution: Steps 7, 8, 9, 10
        with st.chat_message("assistant"):
            try:
                # Step 9: Initialize LLM
                llm = rag_engine.get_chat_model(
                    provider=provider,
                    model_name=model_name,
                    temperature=temperature,
                    streaming=True,
                )

                # Steps 7, 8, 9: Stream tokens and retrieve chunks
                token_stream, retrieved_chunks = rag_engine.stream_answer_generator(
                    query=user_query,
                    vector_store=st.session_state.vector_store,
                    llm=llm,
                    k=top_k,
                    chat_history=st.session_state.messages[:-1],
                )

                # Stream response into Streamlit
                full_response = st.write_stream(token_stream)

                # Ensure string type for storage across reruns
                if isinstance(full_response, list):
                    full_response_text = "".join(str(item) for item in full_response)
                else:
                    full_response_text = str(full_response or "")

                # Show sources immediately under the answer
                if retrieved_chunks:
                    with st.expander(f"🔍 Retrieved Sources & Context ({len(retrieved_chunks)} chunks)", expanded=False):
                        for idx, src in enumerate(retrieved_chunks, 1):
                            st.markdown(
                                f"""
                                <div class="source-box">
                                    <b>Chunk #{src.get('chunk_index', idx)}</b> | 
                                    <b>Document:</b> <code>{src.get('source', 'PDF')}</code> | 
                                    <b>Page:</b> {src.get('page', 'N/A')} | 
                                    <b>Similarity Score:</b> {src.get('similarity_score', 0)}%
                                    <p style="margin-top: 5px; color: #334155; font-style: italic;">
                                        "{src.get('content', '')[:300]}..."
                                    </p>
                                </div>
                                """,
                                unsafe_allow_html=True,
                            )

                # Step 10: Preserve Assistant Message and Sources in Session State
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": full_response_text,
                    "sources": retrieved_chunks,
                })

            except Exception as err:
                error_msg = f"Error during generation: {str(err)}"
                st.error(error_msg)
                st.session_state.messages.append({"role": "assistant", "content": error_msg})

