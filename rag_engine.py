"""
rag_engine.py - Core Retrieval-Augmented Generation (RAG) Engine

Implements the complete 10-step PDF chatbot workflow:
1. Upload / Select PDF
2. Extract Text (PyPDFLoader)
3. Chunk Text (RecursiveCharacterTextSplitter)
4. Generate Embeddings (OpenAI text-embedding-3-small)
5. Store in Vector Database (FAISS)
6. Accept User Question
7. Retrieve Relevant Chunks (FAISS similarity search)
8. Build Grounded Prompt (Strict context + history)
9. Generate Answer with LLM (OpenAI / DeepSeek with streaming)
10. Display & Preserve State
"""

import os
import tempfile
from typing import List, Dict, Any, Generator, Optional, Tuple
from dotenv import load_dotenv

from langchain_core.documents import Document
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_deepseek import ChatDeepSeek

load_dotenv()


# ==============================================================================
# Step 2: Document Loading & Text Extraction
# ==============================================================================
def load_pdf_documents(file_source: Any, filename: Optional[str] = None) -> List[Document]:
    """
    Extracts text and page metadata from a PDF file path or Streamlit UploadedFile object.
    Supports both PyPDFLoader and direct in-memory pypdf.PdfReader parsing.
    """
    import io
    import pypdf

    temp_file_path = None
    display_name = filename or getattr(file_source, "name", "uploaded_document.pdf")

    try:
        # Case 1: file_source is an existing file path string on disk
        if isinstance(file_source, str) and os.path.exists(file_source):
            display_name = filename or os.path.basename(file_source)
            try:
                loader = PyPDFLoader(file_source)
                raw_docs = loader.load()
            except Exception:
                # Fallback to direct pypdf reader if PyPDFLoader fails
                reader = pypdf.PdfReader(file_source)
                raw_docs = []
                for idx, page in enumerate(reader.pages):
                    text = page.extract_text() or ""
                    if text.strip():
                        raw_docs.append(Document(page_content=text, metadata={"page": idx, "source": display_name}))

        # Case 2: file_source is bytes or a file-like object (Streamlit UploadedFile / BytesIO)
        else:
            # Ensure pointer is at the very beginning before reading
            if hasattr(file_source, "seek"):
                file_source.seek(0)

            if hasattr(file_source, "read"):
                data = file_source.read()
            elif isinstance(file_source, bytes):
                data = file_source
            else:
                data = bytes(file_source)

            # Reset pointer again for downstream callers
            if hasattr(file_source, "seek"):
                file_source.seek(0)

            if not data or len(data) == 0:
                raise ValueError(f"Uploaded file '{display_name}' is empty (0 bytes).")

            # Try direct in-memory extraction with pypdf first (avoids Windows temp file locks)
            try:
                reader = pypdf.PdfReader(io.BytesIO(data))
                raw_docs = []
                for idx, page in enumerate(reader.pages):
                    text = page.extract_text() or ""
                    raw_docs.append(Document(page_content=text, metadata={"page": idx, "source": display_name}))
            except Exception:
                # Fallback to PyPDFLoader via temporary file
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                    temp_file_path = tmp.name
                    tmp.write(data)
                loader = PyPDFLoader(temp_file_path)
                raw_docs = loader.load()

        if not raw_docs:
            raise ValueError(
                f"No readable text pages found in PDF '{display_name}'. The file might be scanned images, corrupted, or password-protected."
            )

        # Standardize metadata across documents (1-indexed page numbering for user clarity)
        processed_docs = []
        for doc in raw_docs:
            page_num = doc.metadata.get("page", 0) + 1  # 1-indexed
            doc.metadata["page"] = page_num
            doc.metadata["source"] = display_name
            cleaned_content = doc.page_content.strip()
            if cleaned_content:
                processed_docs.append(Document(page_content=cleaned_content, metadata=doc.metadata))

        if not processed_docs:
            raise ValueError(
                f"No extractable text was found in '{display_name}'. If this document contains scanned images, an OCR parser is required."
            )

        return processed_docs

    finally:
        # Clean up temporary file if one was created
        if temp_file_path and os.path.exists(temp_file_path):
            try:
                os.remove(temp_file_path)
            except OSError:
                pass


# ==============================================================================
# Step 3: Text Chunking
# ==============================================================================
def chunk_documents(
    documents: List[Document],
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
) -> List[Document]:
    """
    Splits loaded documents into overlapping chunks to preserve context across boundaries.
    """
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
        add_start_index=True,
    )
    chunks = text_splitter.split_documents(documents)

    # Attach chunk index for clear provenance
    for idx, chunk in enumerate(chunks):
        chunk.metadata["chunk_index"] = idx + 1

    return chunks


# ==============================================================================
# Step 4: Generate Embeddings
# ==============================================================================
def get_embeddings(model_name: str = "text-embedding-3-small", api_key: Optional[str] = None) -> OpenAIEmbeddings:
    """
    Returns an OpenAI embeddings model instance.
    """
    key = api_key or os.getenv("OPENAI_API_KEY", "")
    if not key:
        raise ValueError("OPENAI_API_KEY is required to generate vector embeddings.")
    return OpenAIEmbeddings(model=model_name, openai_api_key=key)


# ==============================================================================
# Step 5: Vector Database (FAISS)
# ==============================================================================
def create_vector_store(chunks: List[Document], embeddings: Optional[OpenAIEmbeddings] = None) -> FAISS:
    """
    Generates embeddings for all chunks and indexes them in an in-memory FAISS vector store.
    """
    if not chunks:
        raise ValueError("Cannot create vector store: No document chunks provided.")

    if embeddings is None:
        embeddings = get_embeddings()

    vector_store = FAISS.from_documents(chunks, embeddings)
    return vector_store


def add_documents_to_vector_store(vector_store: FAISS, new_chunks: List[Document]) -> FAISS:
    """
    Adds additional document chunks to an existing FAISS vector store.
    """
    if new_chunks:
        vector_store.add_documents(new_chunks)
    return vector_store


# ==============================================================================
# Step 7: Retrieve Relevant Chunks
# ==============================================================================
def retrieve_relevant_chunks(
    vector_store: FAISS,
    query: str,
    k: int = 4,
) -> List[Dict[str, Any]]:
    """
    Performs semantic similarity search against the FAISS vector database.
    Returns the top-k chunks with metadata and normalized relevance score.
    """
    # similarity_search_with_score returns (Document, score) where score is L2 distance
    results_with_score = vector_store.similarity_search_with_score(query, k=k)

    retrieved = []
    for doc, distance in results_with_score:
        # Convert L2 distance to a relative similarity percentage for intuitive display:
        # In normalized embeddings, distance is between 0 and 2.
        # similarity = 1 - (distance / 2)
        similarity_pct = max(0.0, min(100.0, (1.0 - (distance / 2.0)) * 100.0))
        retrieved.append({
            "content": doc.page_content,
            "source": doc.metadata.get("source", "PDF Document"),
            "page": doc.metadata.get("page", 1),
            "chunk_index": doc.metadata.get("chunk_index", 0),
            "l2_distance": round(float(distance), 4),
            "similarity_score": round(similarity_pct, 1),
        })

    return retrieved


# ==============================================================================
# Step 8: Build Grounded Prompt
# ==============================================================================
def build_grounded_prompt(
    query: str,
    retrieved_chunks: List[Dict[str, Any]],
    chat_history: Optional[List[Dict[str, str]]] = None,
) -> Tuple[str, str]:
    """
    Constructs a grounded prompt containing retrieved context, chat history, and system constraints.
    Returns (system_message, user_message).
    """
    system_prompt = (
        "You are an expert, meticulous AI assistant and PDF Chatbot.\n"
        "Your task is to answer the user's question accurately using ONLY the retrieved document excerpts provided below.\n\n"
        "STRICT GROUNDING RULES:\n"
        "1. Base your answers strictly on the retrieved context below. Do NOT fabricate, assume, or extrapolate facts.\n"
        "2. If the retrieved context does not contain sufficient information to answer the question, clearly state:\n"
        "   'The uploaded document does not contain enough information to answer this question.'\n"
        "3. Explicitly cite your sources by mentioning the Document Name and Page Number where the information was found "
        "(e.g., '[Source: handbook.pdf, Page 2]').\n"
        "4. Be concise, structured, professional, and directly address the user's inquiry."
    )

    # Format the context blocks with provenance tags
    context_sections = []
    for idx, chunk in enumerate(retrieved_chunks, 1):
        header = f"--- [Excerpt {idx} | Source: {chunk['source']} | Page: {chunk['page']}] ---"
        context_sections.append(f"{header}\n{chunk['content']}\n")

    combined_context = "\n".join(context_sections)

    # Format recent conversation history (last 4 interactions)
    history_str = ""
    if chat_history:
        recent_history = chat_history[-6:]
        history_lines = []
        for msg in recent_history:
            role = "User" if msg["role"] == "user" else "Assistant"
            history_lines.append(f"{role}: {msg['content']}")
        if history_lines:
            history_str = "RECENT CONVERSATION HISTORY:\n" + "\n".join(history_lines) + "\n\n"

    user_message = (
        f"{history_str}"
        f"RETRIEVED DOCUMENT CONTEXT:\n"
        f"==================================================\n"
        f"{combined_context}\n"
        f"==================================================\n\n"
        f"USER QUESTION: {query}\n\n"
        f"GROUNDED ANSWER:"
    )

    return system_prompt, user_message


# ==============================================================================
# Step 9: LLM Initialization & Answer Generation
# ==============================================================================
def get_chat_model(
    provider: str = "OpenAI",
    model_name: Optional[str] = None,
    temperature: float = 0.2,
    streaming: bool = True,
):
    """
    Instantiates an LLM client for either OpenAI or DeepSeek.
    """
    provider_clean = provider.strip().lower()

    if "deepseek" in provider_clean:
        api_key = os.getenv("DEEPSEEK_API_KEY", "")
        if not api_key:
            raise ValueError("DEEPSEEK_API_KEY not found in environment.")
        return ChatDeepSeek(
            model=model_name or "deepseek-chat",
            temperature=temperature,
            streaming=streaming,
            api_key=api_key,
        )
    else:
        # Default to OpenAI
        api_key = os.getenv("OPENAI_API_KEY", "")
        if not api_key:
            raise ValueError("OPENAI_API_KEY not found in environment.")
        return ChatOpenAI(
            model=model_name or "gpt-4o-mini",
            temperature=temperature,
            streaming=streaming,
            openai_api_key=api_key,
        )


def generate_answer_sync(
    query: str,
    vector_store: FAISS,
    llm,
    k: int = 4,
    chat_history: Optional[List[Dict[str, str]]] = None,
) -> Dict[str, Any]:
    """
    Executes steps 7 -> 8 -> 9 synchronously (useful for testing and non-streaming environments).
    """
    retrieved_chunks = retrieve_relevant_chunks(vector_store, query, k=k)
    system_prompt, user_message = build_grounded_prompt(query, retrieved_chunks, chat_history)

    from langchain_core.messages import SystemMessage, HumanMessage
    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_message),
    ]

    response = llm.invoke(messages)
    answer = response.content if hasattr(response, "content") else str(response)

    return {
        "answer": answer,
        "retrieved_chunks": retrieved_chunks,
        "system_prompt": system_prompt,
        "user_message": user_message,
    }


def stream_answer_generator(
    query: str,
    vector_store: FAISS,
    llm,
    k: int = 4,
    chat_history: Optional[List[Dict[str, str]]] = None,
) -> Tuple[Generator[str, None, None], List[Dict[str, Any]]]:
    """
    Yields tokens incrementally for Streamlit's st.write_stream, and returns the retrieved chunks.
    """
    retrieved_chunks = retrieve_relevant_chunks(vector_store, query, k=k)
    system_prompt, user_message = build_grounded_prompt(query, retrieved_chunks, chat_history)

    from langchain_core.messages import SystemMessage, HumanMessage
    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_message),
    ]

    def token_stream():
        for chunk in llm.stream(messages):
            if hasattr(chunk, "content"):
                c = chunk.content
                if isinstance(c, str):
                    if c:
                        yield c
                elif isinstance(c, list):
                    for item in c:
                        if isinstance(item, str) and item:
                            yield item
                        elif isinstance(item, dict) and "text" in item and item["text"]:
                            yield item["text"]
            elif isinstance(chunk, str) and chunk:
                yield chunk

    return token_stream(), retrieved_chunks

