# 📄 AI-Powered PDF Chatbot Agent

> **An End-to-End Retrieval-Augmented Generation (RAG) System** powered by **DeepSeek LLM**, free local **FastEmbed** vector embeddings, strict factual grounding, real-time token streaming, and an interactive Streamlit interface.

[![Python](https://img.shields.io/badge/Python-3.11%2B-blue.svg)](https://www.python.org/)
[![LLM](https://img.shields.io/badge/LLM-DeepSeek--Chat-blueviolet.svg)](https://platform.deepseek.com/)
[![Embeddings](https://img.shields.io/badge/Embeddings-FastEmbed%20(BGE--Small)-green.svg)](https://github.com/qdrant/fastembed)
[![Vector Store](https://img.shields.io/badge/Vector%20Store-FAISS-008080.svg)](https://github.com/facebookresearch/faiss)
[![UI](https://img.shields.io/badge/UI-Streamlit-FF4B4B.svg)](https://streamlit.io/)
[![Package Manager](https://img.shields.io/badge/Package%20Manager-uv-purple.svg)](https://astral.sh/uv)

---

## 📌 Table of Contents

- [Overview](#-overview)
- [10-Stage RAG Pipeline Architecture](#-10-stage-rag-pipeline-architecture)
- [Key Features](#-key-features)
- [Repository Structure](#-repository-structure)
- [Prerequisites & Setup](#-prerequisites--setup)
  - [1. Clone Repository](#1-clone-repository)
  - [2. Environment Configuration](#2-environment-configuration)
  - [3. Install Dependencies](#3-install-dependencies)
- [Usage & Running Locally](#-usage--running-locally)
- [Deploying to Streamlit Community Cloud](#-deploying-to-streamlit-community-cloud)
- [Automated Verification Suite](#-automated-verification-suite)
- [Configuration & Hyperparameter Tuning](#-configuration--hyperparameter-tuning)
- [Troubleshooting & Windows Specifics](#-troubleshooting--windows-specifics)
- [License](#-license)

---

## 🔍 Overview

The **AI-Powered PDF Chatbot Agent** enables natural-language question answering over user-uploaded PDF documents and technical manuals. Built upon enterprise RAG principles, it overcomes large language model hallucinations through strict contextual bounding: answers are synthesized exclusively from retrieved document chunks, complete with explicit document name and page number citations.

### Core Workflow at a Glance

$$\text{PDF Document} \longrightarrow \text{Text Extraction} \longrightarrow \text{Recursive Chunking} \longrightarrow \text{FastEmbed Local Vectors} \longrightarrow \text{FAISS Index}$$

$$\text{User Query} \longrightarrow \text{k-NN Semantic Retrieval} \longrightarrow \text{Context Synthesis} \longrightarrow \text{DeepSeek LLM Reasoning} \longrightarrow \text{Streaming UI Response}$$

---

## 🏗️ 10-Stage RAG Pipeline Architecture

```
  +-----------------------------------------------------------------------------+
  |                             DOCUMENT INGESTION                              |
  |                                                                             |
  |  [ Step 1: Input ]               [ Step 2: Extraction ]                     |
  |   Uploaded PDF or       --->      PyPDFLoader / In-Memory                   |
  |   Built-in Sample PDF             Page & Metadata Extraction                |
  |                                              |                              |
  |                                              v                              |
  |  [ Step 4: Embedding ]           [ Step 3: Chunking ]                       |
  |   FastEmbed (BGE-small-en)      <--- RecursiveCharacterTextSplitter         |
  |   384-dim CPU Vectors (Free)      (1000 chars, 200 overlap)                 |
  |             |                                                               |
  |             v                                                               |
  |  [ Step 5: Indexing ]                                                       |
  |   FAISS Vector Database (In-Memory L2 Distance Index)                       |
  +-----------------------------------------------------------------------------+
                                     |
                                     v
  +-----------------------------------------------------------------------------+
  |                           RETRIEVAL & GENERATION                            |
  |                                                                             |
  |  [ Step 6: Query ]               [ Step 7: Retrieval ]                      |
  |   User Question via Chat UI ---> Top-K Vector Similarity Search             |
  |                                   Normalized Relevance Scoring              |
  |                                              |                              |
  |                                              v                              |
  |  [ Step 9: Reasoning ]           [ Step 8: Prompt Synthesis ]               |
  |   DeepSeek (deepseek-chat)      <--- Strict Grounding Rules +               |
  |   [or OpenAI gpt-4o-mini]         Retrieved Context + History               |
  |             |                                                               |
  |             v                                                               |
  |  [ Step 10: Interface ]                                                     |
  |   Streamlit Token Streaming UI with Collapsible Source Provenance Cards     |
  +-----------------------------------------------------------------------------+
```

### Stage Breakdown

| Stage | Name | Component | Description |
| :---: | :--- | :--- | :--- |
| **1** | **Document Intake** | Streamlit File Uploader / Sample Generator | Ingests arbitrary user PDF files or generates a technical sample reference handbook. |
| **2** | **Text Extraction** | `PyPDFLoader` & `pypdf.PdfReader` | Extracts raw textual streams while preserving 1-indexed page metadata and file provenance. |
| **3** | **Context Chunking** | `RecursiveCharacterTextSplitter` | Decomposes documents with hierarchical separators (`\n\n`, `\n`, `. `, ` `) to preserve boundary semantics. |
| **4** | **Dense Embeddings** | `FastEmbed` (`BAAI/bge-small-en-v1.5`) | Generates 384-dimensional dense vectors locally on CPU via ONNX Runtime. **100% free with zero API credits required**. (OpenAI `text-embedding-3-small` supported as optional fallback). |
| **5** | **Vector Indexing** | `FAISS` (Facebook AI Similarity Search) | Builds an efficient in-memory similarity search index across all chunk embeddings. |
| **6** | **Question Intake** | Streamlit Chat Interface | Captures natural-language user queries with conversation state maintenance. |
| **7** | **Semantic Retrieval** | `similarity_search_with_score` | Performs k-nearest-neighbor search, returning top-$k$ chunks with L2 distance converted to percentage similarity. |
| **8** | **Prompt Synthesis** | Prompt Engineering Template | Compiles retrieved context, anti-hallucination boundaries, and recent conversation turns into a grounded prompt. |
| **9** | **LLM Generation** | `ChatDeepSeek` (`deepseek-chat`) | Primary reasoning LLM generates verified, fact-grounded responses citing specific page numbers with real-time token streaming. |
| **10** | **Interactive UI** | Streamlit Session State & Streaming | Renders markdown responses with collapsible source accordions displaying chunk text and confidence scores. |

---

## ✨ Key Features

- **Primary DeepSeek Intelligence:** Configured out-of-the-box with **DeepSeek** (`deepseek-chat`) for high-quality, grounded reasoning and cost efficiency.
- **Zero-Cost Local Embeddings:** Embeds documents on your local CPU via **FastEmbed** (`BAAI/bge-small-en-v1.5`). You do **not** need OpenAI credits to vectorize and search your documents.
- **Multi-Source Ingestion:** Drag-and-drop any PDF or load the built-in *Handbook of Modern AI Agents & Vector Architecture* with a single click.
- **Robust In-Memory Extraction:** Dual-mode PDF parsing handles both file paths and byte streams directly, bypassing Windows file locks.
- **Tunable Chunking & Retrieval:** Live controls in the UI for Chunk Size, Chunk Overlap, Top-K Chunks (1–8), and Model Temperature.
- **Dual LLM Provider Support:** Seamlessly switch between **DeepSeek** (`deepseek-chat`) and **OpenAI** (`gpt-4o-mini`, `gpt-4o`).
- **Strict Anti-Hallucination Guardrails:** If the retrieved context is insufficient, the agent explicitly refuses to guess and discloses that the document lacks the answer.
- **Transparent Provenance:** Collapsible source badges show the exact document name, page number, and semantic similarity percentage for every retrieved chunk.
- **Cloud-Ready Deployment:** Includes pre-configured `requirements.txt` and automatic `st.secrets` bridging for Streamlit Community Cloud.
- **Automated Validation Suite:** Includes a standalone test script verifying all 10 stages end-to-end with unit assertions.

---

## 📂 Repository Structure

```
ai-powered-chatbot-agent/
├── app.py                                  # Streamlit web application & UI layout
├── rag_engine.py                           # Core RAG engine: loader, chunker, FastEmbed, FAISS & LLM
├── models.py                               # Centralized LLM provider resolver (DeepSeek / OpenAI fallback)
├── sample_data.py                          # ReportLab script generating sample technical PDF
├── test_pipeline.py                        # Standalone end-to-end automated test suite (10 stages)
├── 00_setup_check.py                       # Environment, API key, and dependency verification script
├── run_app.py                              # Python launcher bypassing Windows AppLocker/execution blocks
├── run.bat                                 # Windows 1-click batch launcher
├── requirements.txt                        # Production dependencies for Streamlit Community Cloud
├── pyproject.toml                          # Project metadata and pinned dependencies
├── uv.lock                                 # Exact reproducible lockfile
├── .env.example                            # Safe environment variable template
├── .gitattributes                          # Git binary safety rules (prevents PDF corruption)
├── sample_documents/                       # Pre-generated technical sample PDF documents
│   ├── ai_agents_guide.pdf
│   └── quantum_and_ai_handbook.pdf
├── presentation_architecture_workflow.pdf  # Project architecture slide deck (PDF)
├── presentation_architecture_workflow.pptx # Project architecture slide deck (PowerPoint)
└── README.md                               # Project documentation
```

---

## 🚀 Prerequisites & Setup

### Requirements

- **Python:** 3.11 or higher
- **Package Manager:** [`uv`](https://astral.sh/uv) (recommended) or standard `pip`
- **API Keys:**
  - `DEEPSEEK_API_KEY`: **Required** for DeepSeek chat generation.
  - `OPENAI_API_KEY`: Optional; only required if using OpenAI models.

---

### 1. Clone Repository

```bash
git clone https://github.com/veeren07/ai-powered-chatbot-agent.git
cd ai-powered-chatbot-agent
```

### 2. Environment Configuration

Copy the template file to create your `.env`:

```bash
cp .env.example .env
```

Edit `.env` and insert your active API key:

```ini
# DeepSeek API Key (Required for chat)
DEEPSEEK_API_KEY=sk-your-deepseek-key-here

# OpenAI API Key (Optional)
OPENAI_API_KEY=your_openai_api_key_here
```

> [!CAUTION]
> Never commit your `.env` file to version control. The repository is pre-configured with [.gitignore](.gitignore) to ignore `.env`.

### 3. Install Dependencies

Using `uv` (fastest & reproducible):

```bash
uv sync
```

Alternatively, using standard `pip`:

```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# Linux/macOS:
source .venv/bin/activate

pip install -r requirements.txt
```

Verify your setup:

```bash
uv run python 00_setup_check.py
```

---

## 🖥️ Usage & Running Locally

### Option A: 1-Click Launcher (Windows)
Double-click `run.bat` or run:
```powershell
.\run.bat
```

### Option B: Python Launcher (Bypasses Windows AppLocker restrictions)
```powershell
uv run python run_app.py
```

### Option C: Standard Streamlit CLI
```powershell
uv run streamlit run app.py
```

Once running, navigate to `http://localhost:8501` in your browser.

### Interacting with the Agent:
1. **Load a Document:** Upload your PDF in the sidebar, or click **⚡ Sample PDF** to test with the built-in AI Handbook.
2. **Configure Settings:** Choose DeepSeek or OpenAI, adjust Temperature, and set Top-K retrieval chunks in the sidebar.
3. **Ask Questions:** Type your question into the chat input.
4. **Inspect Sources:** Expand the "Retrieved Sources" card under any assistant response to view exact text excerpts and page numbers.

---

## ☁️ Deploying to Streamlit Community Cloud

Deploying to [Streamlit Community Cloud](https://share.streamlit.io/) takes under 2 minutes:

1. **Push your code to GitHub:**
   ```powershell
   git push origin capstone-case-aipoweredchatbotagent
   ```
2. **Open Streamlit Cloud:** Go to [share.streamlit.io](https://share.streamlit.io/) and log in with your GitHub account.
3. **Create App:** Click **New app** $\rightarrow$ **I already have an app**.
4. **Select Repository:**
   - **Repository:** `veeren07/ai-powered-chatbot-agent`
   - **Branch:** `capstone-case-aipoweredchatbotagent` (or `main`)
   - **Main file path:** `app.py`
5. **Configure Secrets (Advanced Settings):**
   - Click **Advanced settings...**
   - Python version: select **3.11**
   - In the **Secrets** box, paste your DeepSeek key in TOML format:
     ```toml
     DEEPSEEK_API_KEY = "sk-your-deepseek-api-key"
     ```
6. **Deploy:** Click **Deploy!** Streamlit Cloud will automatically install `requirements.txt` and launch your agent with a public URL.

---

## 🧪 Automated Verification Suite

Run the automated pipeline test to validate all 10 stages:

```bash
uv run python test_pipeline.py
```

### What `test_pipeline.py` Validates:
1. **Document Generation:** Checks sample PDF synthesis via ReportLab.
2. **Text Extraction:** Asserts valid page extraction and 1-indexed numbering.
3. **Chunking:** Tests chunk boundaries and overlap sizing.
4. **FastEmbed Local Vectors:** Verifies 384-dimensional dense vector embeddings generated on CPU.
5. **FAISS Vector Store:** Verifies indexing and document storage.
6. **Similarity Search:** Queries vector store and asserts normalized similarity scores.
7. **Prompt Construction:** Ensures strict anti-hallucination prompts and context insertion.
8. **DeepSeek LLM Generation:** Validates grounded answer generation and source citation.
9. **Out-of-Domain Rejection:** Confirms the agent refuses to answer queries not covered by the document.

---

## ⚙️ Configuration & Hyperparameter Tuning

| Parameter | Default | Recommended Range | Description |
| :--- | :---: | :---: | :--- |
| **LLM Provider** | `DeepSeek` | `DeepSeek` / `OpenAI` | Primary reasoning model (`deepseek-chat`). |
| **Embedding Model** | `FastEmbed` | Fixed | `BAAI/bge-small-en-v1.5` (384-dim dense vectors). Runs locally on CPU for free. |
| **Chunk Size** | `1000` | `500 - 1500` | Size of each text window in characters. |
| **Chunk Overlap** | `200` | `100 - 300` | Overlap between adjacent chunks to prevent cutting concepts across boundaries. |
| **Top-K Chunks** | `4` | `2 - 6` | Number of nearest chunks injected into the prompt context. |
| **Temperature** | `0.2` | `0.0 - 0.3` | Controls model randomness. Keep low ($\le 0.2$) for strict factual grounding. |

---

## 🛠️ Troubleshooting & Windows Specifics

- **Windows Permission Denied on `.git/logs/HEAD`:**
  If git returns `unable to append to '.git/logs/HEAD': Permission denied`, files may have inherited the Windows Read-Only attribute:
  ```powershell
  Get-ChildItem -Path .git -Recurse -Force | ForEach-Object { $_.IsReadOnly = $false }
  ```
- **AppLocker / Windows Application Control Blocks (`os error 4551`):**
  If `streamlit.exe` is blocked by Windows organization policies, always launch via the Python module:
  ```powershell
  uv run python run_app.py
  # or
  uv run python -m streamlit run app.py
  ```
- **Scanned Image PDFs:**
  `PyPDF` extracts textual streams. If a PDF contains scanned images of text without an embedded text layer, run the PDF through an OCR tool before uploading.

---

## 📄 License

This project is licensed under the [MIT License](LICENSE).
