import shutil
from pathlib import Path

from pypdf import PdfReader
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter


PDF_PATH = Path("data/research_report_rag.pdf")
DB_PATH = Path("chroma_report_db")
COLLECTION_NAME = "research_report_pdf"

print("=" * 70)
print("PDF INGESTION: PDF → CHUNKS → EMBEDDINGS → CHROMADB")
print("=" * 70)

# ==========================================================
# 1. Check and read the PDF with pypdf
# ==========================================================
if not PDF_PATH.exists():
    raise FileNotFoundError(
        f"\nPDF not found: {PDF_PATH.resolve()}\n"
        "Put research_report_rag.pdf inside the Day-3/data folder."
    )

print("\n[1/5] Reading PDF with pypdf...")
reader = PdfReader(str(PDF_PATH))
print(f"✓ Read {len(reader.pages)} PDF pages")

# ==========================================================
# 2. Convert PyPDF PageObject objects to LangChain Documents
# Each Document retains file and page metadata for citations.
# ==========================================================
print("\n[2/5] Extracting text and creating LangChain Documents...")

documents = []

for page_index, page in enumerate(reader.pages):
    page_text = page.extract_text() or ""

    if not page_text.strip():
        print(f"  ! Skipped empty page {page_index + 1}")
        continue

    document = Document(
        page_content=page_text,
        metadata={
            "source": PDF_PATH.name,
            "page": page_index,          
            "page_number": page_index + 1
        }
    )

    documents.append(document)

print(f"Created {len(documents)} LangChain page documents")

# ==========================================================
# 3. Split Documents into overlapping chunks
# ==========================================================
print("\n[3/5] Splitting pages into overlapping chunks...")

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=800,
    chunk_overlap=150,
    add_start_index=True
)

chunks = text_splitter.split_documents(documents)

# Readable chunk ID for debugging and source display
for chunk_id, chunk in enumerate(chunks, start=1):
    chunk.metadata["chunk_id"] = chunk_id

print(f"Created {len(chunks)} chunks")

if not chunks:
    raise ValueError(
        "No chunks were created. The PDF may be image-based/scanned "
        "or pypdf could not extract any text."
    )

print("\nFirst chunk preview:")
print(f"Source: {chunks[0].metadata['source']}")
print(f"Page: {chunks[0].metadata['page_number']}")
print(f"Chunk ID: {chunks[0].metadata['chunk_id']}")
print("-" * 70)
print(chunks[0].page_content[:500])

# ==========================================================
# 4. Delete earlier database to prevent duplicate embeddings
# ==========================================================
if DB_PATH.exists():
    print("\n[4/5] Removing existing ChromaDB...")
    shutil.rmtree(DB_PATH)
    print("✓ Old database removed")
else:
    print("\n[4/5] No old ChromaDB found")

# ==========================================================
# 5. Create embeddings and persist chunks to ChromaDB
# ==========================================================
print("\n[5/5] Loading embedding model...")

embedding_model = HuggingFaceEmbeddings(
    model_name="all-mpnet-base-v2"
)

print("✓ Embedding model ready")
print("Creating ChromaDB and storing PDF chunks...")

vectorstore = Chroma.from_documents(
    documents=chunks,
    embedding=embedding_model,
    collection_name=COLLECTION_NAME,
    persist_directory=str(DB_PATH)
)

print("✓ PDF ingestion completed successfully")
print(f"✓ Total vectors stored: {vectorstore._collection.count()}")