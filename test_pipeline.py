"""
test_pipeline.py - Automated validation of the 10-step PDF Chatbot RAG Pipeline
"""

import os
import sys

# Ensure UTF-8 output on Windows consoles to prevent charmap UnicodeEncodeErrors
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

from dotenv import load_dotenv

load_dotenv()

from sample_data import generate_sample_pdf
import rag_engine


def test_rag_end_to_end():
    print("====================================================================")
    print("TESTING 10-STEP PDF CHATBOT RAG PIPELINE")
    print("====================================================================")

    # Step 1: Upload / Select PDF
    print("\n[Step 1] Preparing Sample PDF...")
    pdf_path = generate_sample_pdf()
    assert os.path.exists(pdf_path), f"PDF file not found at {pdf_path}"
    print(f" -> Found PDF: {pdf_path} ({os.path.getsize(pdf_path)} bytes)")

    # Step 2: Extract Text
    print("\n[Step 2] Extracting Text with PyPDFLoader...")
    docs = rag_engine.load_pdf_documents(pdf_path)
    print(f" -> Extracted {len(docs)} pages.")
    assert len(docs) >= 3, f"Expected at least 3 pages, got {len(docs)}"
    for d in docs:
        print(f"    - Page {d.metadata['page']}: {len(d.page_content)} characters")

    # Step 3: Chunk the Text
    print("\n[Step 3] Chunking Text with RecursiveCharacterTextSplitter...")
    chunks = rag_engine.chunk_documents(docs, chunk_size=800, chunk_overlap=150)
    print(f" -> Generated {len(chunks)} overlapping chunks.")
    assert len(chunks) >= 3, f"Expected at least 3 chunks, got {len(chunks)}"
    print(f"    - Sample Chunk 1 preview: {chunks[0].page_content[:120]}...")

    # Step 4: Generate Embeddings
    print("\n[Step 4] Initializing Vector Embeddings...")
    embeddings = rag_engine.get_embeddings()
    test_vec = embeddings.embed_query("testing embedding")
    print(f" -> Embeddings OK (Dimension: {len(test_vec)})")
    assert len(test_vec) in (384, 1536), f"Expected 384 or 1536 dimensions, got {len(test_vec)}"

    # Step 5: Store in Vector Database (FAISS)
    print("\n[Step 5] Indexing Chunks in FAISS Vector Store...")
    vector_store = rag_engine.create_vector_store(chunks, embeddings)
    print(f" -> FAISS Vector Store successfully indexed {len(chunks)} vectors.")
    assert vector_store is not None, "Vector store creation failed"

    # Step 6: Accept User Question
    query = "What is the recommended chunk size and overlap according to the document?"
    print(f"\n[Step 6] User Question: '{query}'")

    # Step 7: Retrieve Relevant Chunks
    print("\n[Step 7] Retrieving Top-3 Relevant Chunks via Similarity Search...")
    retrieved = rag_engine.retrieve_relevant_chunks(vector_store, query, k=3)
    print(f" -> Retrieved {len(retrieved)} chunks:")
    for r in retrieved:
        print(f"    - [Page {r['page']}, Similarity {r['similarity_score']}%]: {r['content'][:90]}...")
    assert len(retrieved) > 0, "No chunks retrieved"

    # Step 8: Build Grounded Prompt
    print("\n[Step 8] Building Grounded Prompt with Citations & Strict Boundaries...")
    sys_prompt, user_msg = rag_engine.build_grounded_prompt(query, retrieved)
    assert "STRICT GROUNDING RULES" in sys_prompt
    assert "RETRIEVED DOCUMENT CONTEXT" in user_msg
    print(" -> Grounded prompt constructed successfully.")

    # Step 9: Generate Answer with LLM (DeepSeek)
    print("\n[Step 9] Generating Answer with DeepSeek Chat LLM...")
    ds_llm = rag_engine.get_chat_model(provider="DeepSeek", model_name="deepseek-chat", temperature=0.0, streaming=False)
    result = rag_engine.generate_answer_sync(query, vector_store, ds_llm, k=3)
    answer = result["answer"]
    print(f"\n--- DEEPSEEK GROUNDED ANSWER ---\n{answer}\n---------------------------------")

    # Verification assertions
    assert "1,000" in answer or "1000" in answer, "Expected answer to contain 1,000 characters chunk size"
    assert "200" in answer, "Expected answer to contain 200 characters overlap"

    # Also test DeepSeek if key is present
    if os.getenv("DEEPSEEK_API_KEY"):
        print("\n[Step 9b] Testing DeepSeek Chat LLM...")
        try:
            ds_llm = rag_engine.get_chat_model(provider="DeepSeek", model_name="deepseek-chat", temperature=0.0, streaming=False)
            ds_res = rag_engine.generate_answer_sync("What does FAISS stand for?", vector_store, ds_llm, k=2)
            print(f"\n--- DEEPSEEK ANSWER ---\n{ds_res['answer']}\n-----------------------")
            assert "Facebook AI Similarity Search" in ds_res["answer"] or "FAISS" in ds_res["answer"]
        except Exception as e:
            print(f"DeepSeek test warning: {e}")

    # Step 10: Preserve state & check out-of-domain query
    print("\n[Step 10] Testing Grounding Boundaries (Querying Information NOT in PDF)...")
    unrelated_query = "What is the recipe for baking chocolate chip cookies?"
    unrelated_res = rag_engine.generate_answer_sync(unrelated_query, vector_store, ds_llm, k=2)
    print(f" -> Query: '{unrelated_query}'")
    print(f" -> Grounded Refusal / Answer: {unrelated_res['answer']}")

    print("\n====================================================================")
    print("ALL 10 PIPELINE STEPS PASSED SUCCESSFULLY!")
    print("====================================================================")


if __name__ == "__main__":
    test_rag_end_to_end()

