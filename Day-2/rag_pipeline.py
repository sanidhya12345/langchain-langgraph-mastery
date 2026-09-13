from dotenv import load_dotenv
from google import genai
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

# load the environment variables and gemini client
load_dotenv()
client=genai.Client()

print("\nLoading embedding model...")
embedding_model = HuggingFaceEmbeddings(
    model_name="all-mpnet-base-v2"
)
print("✓ Embedding model ready")


#load the existing chromadb

print("\nLoading existing chromadb....")

vectorstore=Chroma(
    persist_directory="./chroma_db",
    embedding_function=embedding_model
)
print("Chroma DB loaded")
print(f"Total stored chunks/documents: {vectorstore._collection.count()}")


# configure the retriever

retriever=vectorstore.as_retriever(search_kwargs={"k":2})


#rag function

def ask_rag(question):

    'Retrieve relevant context from ChromaDB and ask Gemini'

    # retrieve the relevant documents

    retrieved_docs=retriever.invoke(question)

    # convert the docs into readable context

    context_parts=[]

    for index, doc in enumerate(retrieved_docs,start=1):

        source=doc.metadata.get("source","Unknown Source")
        topic=doc.metadata.get("topic","Unknown Topic")

        context_parts.append(
            f"[Context {index}]\n"
            f"Source: {source}\n"
            f"Topic: {topic}\n"
            f"Content: {doc.page_content}"
        )
    context = "\n\n".join(context_parts)

    # grounded prompt

    prompt= f"""
            You are a helpful research assistant.

            Answer the user's question using ONLY the supplied context.
            Do not use outside knowledge.
            If the answer is not present in the context, say exactly:
            "I don't have enough information in the provided documents."

            Write a concise answer in clear language.
            After the answer, provide a short "Sources used" line with the source names.

            SUPPLIED CONTEXT:
            {context}

            USER QUESTION:
            {question}

            ANSWER:
            """
    
    interaction=client.interactions.create(
         model="gemini-3.6-flash",
         input=prompt
    )

    return {
        "answer":interaction.output_text,
        "documents":retrieved_docs
    }


# test the queries:

test_queries=[

    "What is explainable AI?",
    "How does RAG improve answers from language models?",
    "What does cybersecurity forensics involve?",
    "What is the capital of India?"
]

for question in test_queries:

    result=ask_rag(question)

    print("RETRIEVED DOCUMENTS:")

    for index,doc in enumerate(result["documents"],start=1):
            print(f"\n[{index}] {doc.metadata.get('source', 'Unknown source')}")
            print(f"Topic: {doc.metadata.get('topic', 'Unknown topic')}")
            print(f"Content: {doc.page_content}")

    print("\nRAG ANSWER:")
    print(result["answer"])