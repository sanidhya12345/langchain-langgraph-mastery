from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document
import shutil
from pathlib import Path
print("Loading Embedding Model...")

embeddings=HuggingFaceEmbeddings(
    model_name="all-mpnet-base-v2"
)

print("Embeddings ready!")

# Sample Document

docs = [
    Document(
        page_content="Explainable AI (XAI) helps humans understand how machine learning models make decisions.",
        metadata={"source": "xai_paper.pdf", "topic": "XAI"}
    ),
    Document(
        page_content="RAG combines information retrieval with language models to generate accurate answers.",
        metadata={"source": "rag_paper.pdf", "topic": "RAG"}
    ),
    Document(
        page_content="Cybersecurity forensics analyzes digital evidence to investigate security breaches.",
        metadata={"source": "forensics.pdf", "topic": "Forensics"}
    )
]

# ChromaDB vector store
# each and every time it will store the embeddings into the same chroma db 
# that will create redundancy in embeddings.
print("Creating ChromaDB vector store...")

db_path=Path("./chroma_db")

if db_path.exists():
    shutil.rmtree(db_path)

vectorstore=Chroma.from_documents(
    documents=docs,
    embedding=embeddings,
    persist_directory="./chroma_db"
)

# Retrieval Test

query="What is explainable AI?"
print(f"Query: {query}\n")

retrieval = vectorstore.as_retriever(search_kwargs={"k":1})
results = retrieval.invoke(query)

print(f"Found {len(results)} relevant documents\n")

for i, doc in enumerate(results, 1):
    print(f"Document {i}: ")
    print(f"  Content: {doc.page_content}")
    print(f"   Source: {doc.metadata['source']}")
    print()