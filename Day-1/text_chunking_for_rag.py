print("\n" + "=" * 60)
print("TEXT CHUNKING FOR RAG")
print("=" * 60)

def chunk_text(text, chunk_size=50, overlap=10):
    """
    It will divide the text into overlapping chunks
    
    Args:
        text: Input text (string)
        chunk_size: Words per chunk
        overlap: Overlapping words between chunks
    """
    words = text.split()
    chunks = []
    
    for i in range(0, len(words), chunk_size - overlap):
        chunk = ' '.join(words[i:i + chunk_size])
        chunks.append(chunk)
    
    return chunks

# Sample research abstract
abstract = """
Explainable AI (XAI) is a field that focuses on making machine learning 
models more transparent and interpretable. In cybersecurity and digital 
forensics, XAI helps investigators understand why a model classified 
certain evidence as malicious. This is crucial for legal proceedings 
and building trust in automated systems. Our research focuses on 
combining XAI with retrieval-augmented generation to create systems 
that can answer forensic queries with citations from research papers.
Large language models often hallucinate facts, but RAG systems can 
reduce hallucinations by grounding answers in retrieved documents.
"""

# Chunking apply karo
chunks = chunk_text(abstract, chunk_size=30, overlap=5)

print(f"Original text length: {len(abstract)} characters")
print(f"Total chunks created: {len(chunks)}\n")

for i, chunk in enumerate(chunks):
    print(f"Chunk {i}:")
    print(f"  {chunk.strip()}")
    print(f"  (Words: {len(chunk.split())})")
    print("-" * 80)