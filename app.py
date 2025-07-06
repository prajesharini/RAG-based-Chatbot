from pathlib import Path
import streamlit as st
import requests
from PyPDF2 import PdfReader
from bs4 import BeautifulSoup
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_community.llms import Ollama
from langchain.chains.question_answering import load_qa_chain
import tempfile
import sys

st.set_page_config(page_title="Ask about PDFs or URLs", layout="wide")
st.title("📄🧠 Ask Questions from PDF or Website (Ollama-powered)")

# Show current Python executable for debugging
st.caption(f"🧪 Python running from: {sys.executable}")

# Text Extraction from PDF
def extract_text_from_pdf(file_path):
    reader = PdfReader(file_path)
    text = ""
    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            text += page_text
    return text

# Text Extraction from Website
def extract_text_from_website(url):
    try:
        response = requests.get(url.strip(), timeout=10)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, "html.parser")
        for tag in soup(["script", "style"]):
            tag.decompose()
        text = soup.get_text(separator="\n").strip()
        if len(text.split()) < 50:
            raise ValueError("Website doesn't contain enough readable text.")
        return text
    except Exception as e:
        raise RuntimeError(f"Website fetch failed: {e}")

# Split large text into chunks
def split_text(text):
    st.write("✂️ Splitting text into chunks...")
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_text(text)
    st.write(f"📦 Created {len(chunks)} text chunks.")
    return chunks

# Create vector store
def create_vector_store(chunks):
    st.write("🔢 Generating embeddings and creating vector store...")
    embeddings = HuggingFaceEmbeddings()
    return FAISS.from_texts(chunks, embeddings)

# Ask question using Ollama
def ask_question(vectorstore, query):
    st.write("🤖 Asking model via Ollama...")
    retriever = vectorstore.as_retriever()
    docs = retriever.get_relevant_documents(query)
    llm = Ollama(model="llama3")
    chain = load_qa_chain(llm, chain_type="stuff")
    return chain.run(input_documents=docs, question=query)

# UI Inputs
col1, col2 = st.columns([3, 1])
with col1:
    url = st.text_input("Enter a PDF or website URL").strip()
with col2:
    uploaded_pdf = st.file_uploader("Or upload a PDF", type="pdf")

question = st.text_input("Ask your question:")

chunks = []

try:
    if url:
        st.info("📥 Fetching content from URL...")
        if url.lower().endswith(".pdf"):
            response = requests.get(url, timeout=15)
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                tmp.write(response.content)
                pdf_text = extract_text_from_pdf(tmp.name)
                chunks = split_text(pdf_text)
        else:
            website_text = extract_text_from_website(url)
            chunks = split_text(website_text)
        st.success("✅ URL content loaded.")

    elif uploaded_pdf:
        st.info("📄 Reading uploaded PDF...")
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(uploaded_pdf.read())
            pdf_text = extract_text_from_pdf(tmp.name)
            chunks = split_text(pdf_text)
        st.success("✅ PDF file processed.")

    else:
        st.warning("📂 Please upload a PDF or enter a valid URL.")

except Exception as e:
    st.error(f"❌ Error during content extraction: {e}")

# Answer section
if chunks and question:
    with st.spinner("💬 Thinking..."):
        try:
            vectorstore = create_vector_store(chunks)
            answer = ask_question(vectorstore, question)
            st.success("🧠 Answer:")
            st.write(answer)
        except Exception as e:
            st.error(f"❌ Failed to answer: {e}")