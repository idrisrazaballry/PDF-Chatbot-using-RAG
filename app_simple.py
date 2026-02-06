import streamlit as st
import os, tempfile
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS

# Config
st.set_page_config(layout="wide")
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None

# API Key (sidebar)
with st.sidebar:
    api_key = st.text_input("🔑 OpenAI API Key", type="password")
    if not api_key:
        st.warning("Enter API key first!")
        st.stop()
    os.environ["OPENAI_API_KEY"] = api_key

st.title("📚 RAG PDF Chatbot")

# File upload
uploaded_file = st.file_uploader("Upload PDF", type="pdf")
if uploaded_file:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(uploaded_file.getvalue())
        pdf_path = tmp.name
    
    if st.button("🔄 Process PDF", type="primary"):
        with st.spinner("Processing..."):
            # Load & split
            loader = PyPDFLoader(pdf_path)
            docs = loader.load()
            splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
            chunks = splitter.split_documents(docs)
            
            # Vector store
            embeddings = OpenAIEmbeddings()
            st.session_state.vectorstore = FAISS.from_documents(chunks, embeddings)
            st.success("✅ Ready to chat!")

# Chat
if st.session_state.vectorstore:
    retriever = st.session_state.vectorstore.as_retriever()
    llm = ChatOpenAI(model="gpt-4o-mini")
    
    for msg in st.session_state.get("messages", []):
        with st.chat_message(msg["role"]):
            st.write(msg["content"])
    
    if question := st.chat_input("Ask about your PDF..."):
        st.session_state.setdefault("messages", []).append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.write(question)
        
        with st.chat_message("assistant"):
            # Retrieve + Generate
            docs = retriever.invoke(question)
            context = "\n\n".join([d.page_content for d in docs])
            response = llm.invoke(f"Context: {context}\n\nQ: {question}\nA:")
            
            st.write(response.content)
            st.session_state["messages"].append({"role": "assistant", "content": response.content})

else:
    st.info("👆 Upload PDF → Process → Chat!")
