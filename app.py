import streamlit as st
from ingest import load_pdf, split_text, store_in_chroma, get_embeddings
from query import answer_question

st.title("Ask a question about your pdf file")
pdf_file = st.file_uploader("Upload a PDF", type="pdf")

if pdf_file is not None:
    with open("data/" + pdf_file.name, "wb") as f:
        f.write(pdf_file.read())

    loaded_raw_pdf = load_pdf("data/" + pdf_file.name)
    splitted_text = split_text(loaded_raw_pdf)
    store_in_chroma(splitted_text, get_embeddings())
    st.success("Document ingested!")

    question = st.chat_input("Ask a question about your document")
    if question:
        answer = answer_question(question)
        st.write(answer)