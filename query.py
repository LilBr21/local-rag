from langchain_huggingface import HuggingFaceEmbeddings
import chromadb

def retrieve(question, n_results=3):
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    question_vector = embeddings.embed_query(question)
    chroma_client = chromadb.PersistentClient(path="chroma_db")
    docs_collection = chroma_client.get_collection("my_docs")
    return docs_collection.query(query_embeddings=[question_vector], n_results=n_results)
