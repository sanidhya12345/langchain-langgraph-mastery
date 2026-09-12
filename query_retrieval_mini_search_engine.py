from embeddings_basics import embeddings,model,texts
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

print("\n"+ "="*60)
print("QUERY-BASED RETRIEVAL (Mini Search Engine)")
print("="*60)

query="What is explainable AI and why it is important?"
print(f"Query: '{query}'\n")

# we will generate the embeddings of query

query_embedding=model.encode([query])

# calculate the cosine similarity between query and the embeddings generated 

similarities=cosine_similarity(query_embedding,embeddings)[0]

print("Similarity scores:")
for idx, sim in enumerate(similarities):
    text_preview = texts[idx][:60] + "..." if len(texts[idx]) > 60 else texts[idx]
    print(f"  {sim:.4f} → Text {idx}: {text_preview}")

# best match

best_match_idx = np.argmax(similarities)
best_score = similarities[best_match_idx]

print(f"\n BEST MATCH: Text {best_match_idx} (Score: {best_score:.4f})")
print(f"   '{texts[best_match_idx]}'")