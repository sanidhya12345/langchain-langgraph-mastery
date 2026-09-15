from pathlib import Path

from dotenv import load_dotenv
from google import genai
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings


# ==========================================================
# Configuration
# ==========================================================
DB_PATH = Path("chroma_report_db")
COLLECTION_NAME = "research_report_pdf"

MODEL_NAME = "gemini-3.6-flash"

TOP_K = 3


# ==========================================================
# Setup
# ==========================================================
print("=" * 75)
print("CHAT WITH PDF - EXPLAINABLE RAG")
print("=" * 75)

load_dotenv()

if not DB_PATH.exists():
    raise FileNotFoundError(
        f"\nChromaDB folder not found: {DB_PATH.resolve()}\n"
        "Pehle ingest_pdf.py run karo."
    )

print("\n[1/3] Initializing Gemini client...")
client = genai.Client()
print("✓ Gemini client ready")

print("\n[2/3] Loading the SAME embedding model...")
embedding_model = HuggingFaceEmbeddings(
    model_name="all-mpnet-base-v2"
)
print("✓ Embedding model ready")

print("\n[3/3] Loading persisted ChromaDB...")
vectorstore = Chroma(
    collection_name=COLLECTION_NAME,
    persist_directory=str(DB_PATH),
    embedding_function=embedding_model
)

total_chunks = vectorstore._collection.count()

if total_chunks == 0:
    raise ValueError(
        "ChromaDB loaded, but it contains 0 chunks. "
        "Run ingest_pdf.py again."
    )

print(f"✓ ChromaDB loaded with {total_chunks} chunks")


# ==========================================================
# Helper: Retrieved documents → Gemini context
# ==========================================================
def format_context(search_results):
    """
    search_results format:
    [(Document, distance), (Document, distance), ...]
    """

    blocks = []

    for index, (doc, distance) in enumerate(search_results, start=1):
        source = doc.metadata.get("source", "Unknown source")
        page_number = doc.metadata.get("page_number")

        if page_number is None:
            page_number = doc.metadata.get("page", 0) + 1

        chunk_id = doc.metadata.get("chunk_id", "Unknown")

        block = f"""
[SOURCE {index}]
File: {source}
Page: {page_number}
Chunk ID: {chunk_id}
Retrieval distance: {distance:.4f}

Content:
{doc.page_content}
"""
        blocks.append(block.strip())

    return "\n\n".join(blocks)


# ==========================================================
# Main RAG Function
# ==========================================================
def ask_pdf(question):
    """
    1. Query embedding is generated internally by Chroma.
    2. Top-k relevant chunks are retrieved from ChromaDB.
    3. Gemini answers using only those retrieved chunks.
    """

    # Chroma lower distance = generally closer match.
    search_results = vectorstore.similarity_search_with_score(
        query=question,
        k=TOP_K
    )

    context = format_context(search_results)

    prompt = f"""
You are a precise research assistant answering questions about one PDF.

Use ONLY the supplied PDF context below. Do not use your own general
knowledge. Do not add facts that are not explicitly supported by the context.

Rules:
1. If the answer is absent from the supplied context, reply exactly:
"I don't have enough information in the provided PDF."

2. Keep the answer concise and clear.

3. Add a page citation after each factual statement in this exact format:
[Page N]

4. Never invent a page number, source, method, result, limitation, or citation.

SUPPLIED PDF CONTEXT:
{context}

USER QUESTION:
{question}

ANSWER:
"""

    interaction = client.interactions.create(
        model=MODEL_NAME,
        input=prompt
    )

    return interaction.output_text, search_results


# ==========================================================
# Interactive Chat Loop
# ==========================================================
print("\n" + "=" * 75)
print("Ready! Ask questions about your PDF.")
print("Type 'exit' or 'quit' to close the chatbot.")
print("=" * 75)

while True:
    question = input("\nAsk a question: ").strip()

    if question.lower() in {"exit", "quit"}:
        print("\nChat closed. Great work!")
        break

    if not question:
        print("Please enter a valid question.")
        continue

    try:
        answer, search_results = ask_pdf(question)

        print("\n" + "-" * 75)
        print("ANSWER")
        print("-" * 75)
        print(answer)

        print("\n" + "-" * 75)
        print("RETRIEVAL DEBUG INFORMATION")
        print("-" * 75)

        for index, (doc, distance) in enumerate(search_results, start=1):
            source = doc.metadata.get("source", "Unknown source")
            page_number = doc.metadata.get("page_number")

            if page_number is None:
                page_number = doc.metadata.get("page", 0) + 1

            chunk_id = doc.metadata.get("chunk_id", "Unknown")

            print(f"\n[{index}] Source: {source}")
            print(f"    Page: {page_number} | Chunk: {chunk_id}")
            print(f"    Distance: {distance:.4f}")
            print(f"    Preview: {doc.page_content[:300].replace(chr(10), ' ')}...")

    except Exception as error:
        print(f"\nError: {error}")
        print("Check your API key, Gemini model name, and ChromaDB path.")