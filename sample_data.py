"""
sample_data.py - Generates rich sample PDF documents for testing the PDF Chatbot Agent.
"""

import os
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, PageBreak, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors


def generate_sample_pdf(output_path="sample_documents/quantum_and_ai_handbook.pdf"):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    doc = SimpleDocTemplate(
        output_path,
        pagesize=letter,
        rightMargin=54,
        leftMargin=54,
        topMargin=54,
        bottomMargin=54,
    )
    styles = getSampleStyleSheet()

    title_style = ParagraphStyle(
        'DocTitle',
        parent=styles['Heading1'],
        fontSize=22,
        leading=26,
        textColor=colors.HexColor('#1E293B'),
        spaceAfter=12,
    )
    h2_style = ParagraphStyle(
        'DocH2',
        parent=styles['Heading2'],
        fontSize=14,
        leading=18,
        textColor=colors.HexColor('#2563EB'),
        spaceBefore=12,
        spaceAfter=6,
    )
    body_style = ParagraphStyle(
        'DocBody',
        parent=styles['BodyText'],
        fontSize=10,
        leading=15,
        textColor=colors.HexColor('#334155'),
        spaceAfter=8,
    )

    story = []

    # Page 1: Overview and Architecture
    story.append(Paragraph("Handbook of Modern AI Agents & Vector Architecture", title_style))
    story.append(Paragraph("A Technical Reference on RAG, Vector Search, and Autonomous Agents", styles['Italic']))
    story.append(Spacer(1, 16))

    story.append(Paragraph("1. Executive Summary & Core Workflow", h2_style))
    story.append(Paragraph(
        "Modern intelligent chatbots rely on Retrieval-Augmented Generation (RAG) to ground large language models "
        "in verifiable facts. The end-to-end RAG workflow follows a ten-stage pipeline: (1) Document Acquisition, "
        "(2) Text Extraction via PDF parsers, (3) Contextual Text Chunking, (4) High-dimensional Vector Embedding, "
        "(5) Vector Store Indexing with FAISS, (6) Natural Language Question Intake, (7) Semantic k-Nearest Neighbor Retrieval, "
        "(8) Grounded Context Synthesis into Prompts, (9) LLM Reasoning & Token Generation, and (10) Interactive State-Preserving Chat UI.",
        body_style,
    ))

    story.append(Paragraph("2. Document Parsing with PyPDF", h2_style))
    story.append(Paragraph(
        "PDF extraction extracts textual streams from diverse layout streams. PyPDFLoader processes pages sequentially, "
        "retaining document metadata including file names, page numbers, and stream bounding coordinates. Handling PDF fonts "
        "and character encodings such as UTF-8 ensures that specialized symbols and numerical figures are preserved intact.",
        body_style,
    ))

    story.append(PageBreak())

    # Page 2: Chunking & Embeddings
    story.append(Paragraph("3. Semantic Chunking Strategies", h2_style))
    story.append(Paragraph(
        "Text chunking determines the granularity of semantic memory. A RecursiveCharacterTextSplitter decomposes text using "
        "a hierarchy of separators: double newlines, single newlines, spaces, and punctuation. For standard technical documentation, "
        "the recommended chunk size is 1,000 characters with an overlap of 200 characters. Overlapping chunks ensure that definitions, "
        "equations, and relationships spanning paragraph boundaries remain coherent during vector similarity retrieval.",
        body_style,
    ))

    story.append(Paragraph("4. Vector Embeddings and Dense Representations", h2_style))
    story.append(Paragraph(
        "Embeddings map arbitrary text fragments into continuous dense vector spaces where semantic similarity corresponds to geometric proximity. "
        "Models such as OpenAI's text-embedding-3-small project text into a 1,536-dimensional vector space. In this space, cosine similarity "
        "measures the directional alignment of concepts. DeepSeek and OpenAI language models can then digest retrieved chunks with high precision.",
        body_style,
    ))

    story.append(PageBreak())

    # Page 3: FAISS & Prompt Engineering
    story.append(Paragraph("5. Vector Indexing with FAISS", h2_style))
    story.append(Paragraph(
        "Facebook AI Similarity Search (FAISS) is an industry-standard library for indexing dense vectors. It supports multiple index types, "
        "including IndexFlatL2 for exact Euclidean distance search and IndexHNSW for approximate nearest neighbors at massive scale. "
        "For interactive PDF chatbots, FAISS delivers sub-5-millisecond query retrieval over thousands of document chunks.",
        body_style,
    ))

    table_data = [
        ["Component", "Technology", "Key Parameter / Metric"],
        ["PDF Parser", "PyPDFLoader / pypdf", "Extracts text and page metadata"],
        ["Chunker", "RecursiveCharacterTextSplitter", "Size: 1000 chars, Overlap: 200 chars"],
        ["Embeddings", "OpenAI text-embedding-3-small", "1536 Dimensions, Normalized Cosine"],
        ["Vector Database", "FAISS In-Memory Index", "L2 distance / Inner Product top-k"],
        ["LLM Providers", "OpenAI (GPT-4o-mini) & DeepSeek", "Grounded prompt answering"],
        ["User Interface", "Streamlit", "Stateful chat with streaming tokens"],
    ]

    t = Table(table_data, colWidths=[110, 180, 200])
    t.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#2563EB')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 10),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 6),
        ('BACKGROUND', (0, 1), (-1, -1), colors.HexColor('#F8FAFC')),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#CBD5E1')),
        ('FONTSIZE', (0, 1), (-1, -1), 9),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.HexColor('#FFFFFF'), colors.HexColor('#F1F5F9')]),
    ]))
    story.append(t)
    story.append(Spacer(1, 14))

    story.append(Paragraph("6. Grounded Prompt Engineering and Hallucination Mitigation", h2_style))
    story.append(Paragraph(
        "A critical advantage of RAG over vanilla LLMs is hallucination suppression. By configuring system instructions to strictly constrain "
        "answers to the retrieved document snippets, the agent produces reliable, auditable answers with source citations. If information is absent "
        "from the retrieved context, the agent politely states that the document does not contain the answer rather than guessing.",
        body_style,
    ))

    doc.build(story)
    print(f"Generated sample PDF at: {output_path}")
    return output_path


if __name__ == "__main__":
    generate_sample_pdf()

