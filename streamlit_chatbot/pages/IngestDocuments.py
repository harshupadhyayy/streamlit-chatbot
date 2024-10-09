import streamlit as st
from PyPDF2 import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS

def get_pdf_text(pdf_docs):
    text=""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text+= page.extract_text()
    return text

def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    chunks = text_splitter.split_text(text)
    return chunks

def get_vector_store(text_chunks):
    embeddings = OpenAIEmbeddings()
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    # vector_store.save_local("faiss_index")
    st.session_state.vector_store = vector_store


if 'pdf_docs' not in st.session_state:
    st.session_state.pdf_docs = []

if 'vector_store' not in st.session_state:
    st.session_state.vector_store = None

st.title("Ingest documents in this page")

allowed_extensions = ['pdf']

uploaded_files = st.file_uploader(
    "Choose a PDF file", accept_multiple_files=True, type=allowed_extensions
)
for uploaded_file in uploaded_files:
    st.session_state.pdf_docs.append(uploaded_file)

button = st.button("Submit and process")
    
if button:
    with st.spinner("Processing..."):
        extracted_text = get_pdf_text(st.session_state.pdf_docs)
        chunked_data = get_text_chunks(extracted_text)
        get_vector_store(chunked_data)
        st.success("Done")
    

if st.session_state.pdf_docs:
    st.write("Current documents in the index: ")
    for index, doc in enumerate(st.session_state.pdf_docs):
        st.write(f"{index + 1}. {doc.name}")

