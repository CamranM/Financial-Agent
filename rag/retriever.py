from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma


VECTORSTORE_PATH = "rag/vectorstore"


embeddings = OllamaEmbeddings(
    model="nomic-embed-text"
)


vectorstore = Chroma(
    persist_directory=VECTORSTORE_PATH,
    embedding_function=embeddings
)


retriever = vectorstore.as_retriever(
    search_kwargs={"k": 4}
)


query = "Why did Apple's revenue increase?"

results = retriever.invoke(query)


for i, document in enumerate(results, start=1):
    print(f"\n--- Result {i} ---")
    print(document.page_content)