import fitz
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
import chromadb

def load_pdf(path):
    doc = fitz.open(path)
    raw_text = ''
    for page in doc:
        page_text = page.get_text()
        raw_text += page_text
    return raw_text

def split_text(raw_text):
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    return splitter.split_text(raw_text)

def get_embeddings():
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    return embeddings

def store_in_chroma(chunks, embeddings):
    chroma_client = chromadb.PersistentClient(path="chroma_db")
    collection = chroma_client.get_or_create_collection("my_docs")
    embedded_chunks = embeddings.embed_documents(chunks)
    collection.add(documents=chunks, embeddings=embedded_chunks, ids=[str(i) for i in range(len(embedded_chunks))])
