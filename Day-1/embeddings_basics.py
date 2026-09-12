from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# Load the pretrained embeddings

print("Loading the embedding model...")
model=SentenceTransformer('all-MiniLM-L6-v2')
print("Model loaded!")

# We are providing some sample texts 

texts = [
    "Explainable AI (XAI) is a research field focused on making AI systems transparent so humans can understand model predictions and trust automated decisions in critical applications like healthcare and cybersecurity",
    "XAI techniques provide interpretability for deep learning models by highlighting important features, generating explanations, and visualizing decision boundaries for domain experts",
    "Digital forensics and cybersecurity investigation involves collecting and analyzing electronic evidence from computers and networks to identify security breaches and support legal proceedings",
    "The cat is sitting on the mat in the living room, watching birds outside the window while sleeping peacefully in the warm sunlight",
    "LangChain is a framework for developing applications powered by language models, enabling developers to chain prompts, connect external data sources, and build autonomous agents",
    "Retrieval-augmented generation (RAG) enhances language models by retrieving relevant documents from a knowledge base and using them as context to generate more accurate and grounded responses"
]

# Generate the Embeddings.

print("Generate the embeddings...")
embeddings=model.encode(texts)

print(f"Number of texts: {len(texts)}")
print(f"Embedding shape: {embeddings.shape}")
print(f"Each text--> {embeddings.shape[1]}-dimensional vector\n")

# Now we will check the cosine similarity 

print("COSINE SIMILARITY MATRIX")
similarity_matrix=cosine_similarity(embeddings)

for i in range(len(texts)):
    for j in range(i+1,len(texts)):
        sim=similarity_matrix[i][j]

        text_i=texts[i][:50]+"..." if len(texts[i])> 50 else texts[i]
        text_j=texts[j][:50]+"..." if len(texts[j])>50 else texts[j]

        print(f"\nText {i} <-> Text {j}: {sim:.4f}")
        print(f" T{i}: {text_i}")
        print(f" T{j}: {text_j}")

        if sim > 0.7:
            print("  → HIGH similarity (very similar meaning)")
        elif sim > 0.5:
            print("  → MODERATE-HIGH similarity")
        elif sim > 0.3:
            print("  → MODERATE similarity")
        else:
            print("  → LOW similarity (different topics)")