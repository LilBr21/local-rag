from langchain_huggingface import HuggingFaceEmbeddings
import chromadb
import ollama

def retrieve(question, n_results=3):
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    question_vector = embeddings.embed_query(question)
    chroma_client = chromadb.PersistentClient(path="chroma_db")
    docs_collection = chroma_client.get_collection("my_docs")
    return docs_collection.query(query_embeddings=[question_vector], n_results=n_results)

def answer_question(question):
    answer = retrieve(question)
    chunks = answer["documents"][0]
    context = "\n\n".join(chunks)

    prompt = f"""You are a helpful assistant. Answer the user's question using ONLY the context below.
      If the answer is not in the context, say "I don't have enough information to answer that."                                                                                                                                                                                       

      Context:                                                                                                                                                                                                                           
      {context}       

      Question: {question}

      Answer:"""

    response = ollama.chat(model="llama3", messages=[{"role": "user", "content": prompt}])
    return response["message"]["content"]
