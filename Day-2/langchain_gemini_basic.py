from google import genai
from dotenv import load_dotenv
load_dotenv()

print("Initializing Gemini LLM")

#simple query test

client=genai.Client()

query="Explain RAG in 2 lines for a CS student"
print(f"Query: {query}\n")

interaction=client.interactions.create(
    model="gemini-3.6-flash",
    input=query
)

print("Response:")
print(interaction.output_text)
