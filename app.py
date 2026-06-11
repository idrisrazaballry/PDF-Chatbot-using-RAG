"""
RAG PDF Chatbot — Groq Compatible Fix
Fixes groq.BadRequestError by reducing context size and using correct imports.
"""

import streamlit as st
import os
import tempfile
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings          # free, no API key
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_groq import ChatGroq                               # FIX 3: correct import

st.set_page_config(page_title="RAG PDF Chatbot (Groq)", layout="wide")

if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None
if "messages" not in st.session_state:
    st.session_state.messages = []

# FIX 1: smaller chunks so retrieved context stays within Groq's token limit
@st.cache_resource
def process_pdf(_pdf_path):
    loader = PyPDFLoader(_pdf_path)
    docs = loader.load()
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,      # was 1000 — halved to reduce context size
        chunk_overlap=50     # was 150
    )
    chunks = splitter.split_documents(docs)
    # Free local embeddings — no OpenAI key needed
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    vectorstore = FAISS.from_documents(chunks, embeddings)
    return vectorstore

# Sidebar: Groq API key (free at console.groq.com)
with st.sidebar:
    st.header("🔑 Groq API Key")
    api_key = st.text_input(
        "Groq API Key", type="password",
        help="Get a free key at https://console.groq.com"
    )
    if not api_key:
        st.warning("⚠️ Enter your Groq API key to continue.")
        st.stop()
    os.environ["GROQ_API_KEY"] = api_key

    model_choice = st.selectbox(
        "Model",
        ["llama3-8b-8192", "llama3-70b-8192", "mixtral-8x7b-32768"],
        index=0
    )

def create_groq_rag(vectorstore):
    # FIX 1 cont: retrieve only 3 chunks (was 4)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

    # FIX 3: use ChatGroq instead of ChatOpenAI
    llm = ChatGroq(model=model_choice, temperature=0)

    prompt = ChatPromptTemplate.from_template("""
Use ONLY the following context to answer the question.
If the answer is not in the context, say "Not found in document."

Context: {context}

Question: {question}

Answer:""")

    # FIX 2: truncate context to ~2000 chars to stay within Groq token limits
    def format_docs(docs):
        combined = "\n\n".join(doc.page_content for doc in docs)
        return combined[:2000]   # hard cap — prevents BadRequestError

    chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    return chain

# UI
st.title("📚 RAG PDF Chatbot (Groq)")
st.markdown("*Upload PDF → Process → Chat with Groq LLMs for free!*")

col1, col2 = st.columns([3, 1])

with col1:
    uploaded_file = st.file_uploader("Choose PDF", type="pdf")
    if uploaded_file:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(uploaded_file.getvalue())
            pdf_path = tmp.name

        if st.button("🔄 Process PDF", type="primary"):
            with st.spinner("Processing (local embeddings, no cost)..."):
                st.session_state.vectorstore = process_pdf(pdf_path)
                st.session_state.messages = []
                st.success("✅ Ready to chat!")

with col2:
    if st.session_state.vectorstore:
        st.success("📄 PDF Loaded!")
    else:
        st.info("👆 Upload PDF first")

# Chat
if st.session_state.vectorstore:
    chain = create_groq_rag(st.session_state.vectorstore)

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])

    if prompt := st.chat_input("Ask about your PDF..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.write(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    response = chain.invoke(prompt)
                    st.write(response)
                    st.session_state.messages.append(
                        {"role": "assistant", "content": response}
                    )
                except Exception as e:
                    st.error(f"Error: {e}")

    if st.button("🗑️ Clear Chat"):
        st.session_state.messages = []
        st.rerun()

else:
    st.info("👆 Upload & process a PDF first!")
dfghp[
';lkjjkl;'
';lkjjkl;'
';lk
