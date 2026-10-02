from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma


BASE_DIR = Path(__file__).resolve().parent.parent

PDF_PATH = BASE_DIR / "rag" / "documents" / "AAPL_10K.pdf"
VECTORSTORE_PATH = BASE_DIR / "rag" / "vectorstore"


# Load PDF
loader = PyPDFLoader(str(PDF_PATH))
documents = loader.load()

print(f"Loaded {len(documents)} pages")


# Split document into chunks
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)

chunks = text_splitter.split_documents(documents)

print(f"Created {len(chunks)} chunks")


# Create embeddings using Ollama
embeddings = OllamaEmbeddings(
    model="nomic-embed-text"
)


# Store embeddings in Chroma
vectorstore = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    persist_directory=str(VECTORSTORE_PATH)
)

print("Vector store created successfully.")