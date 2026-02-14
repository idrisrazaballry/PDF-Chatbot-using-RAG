"""
FIXED RAG PDF Chatbot - LangChain 0.3+ Compatible (Feb 2026)
"""

import streamlit as st
import os
import tempfile
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

st.set_page_config(page_title="RAG PDF Chatbot", layout="wide")

if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None
if "messages" not in st.session_state:
    st.session_state.messages = []

@st.cache_resource
def process_pdf(_pdf_path):
    loader = PyPDFLoader(_pdf_path)
    docs = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    chunks = splitter.split_documents(docs)
    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.from_documents(chunks, embeddings)
    return vectorstore

# Sidebar API Key
with st.sidebar:
    st.header("API Key")
    api_key = st.text_input("OpenAI API Key", type="password", 
                          help="https://platform.openai.com/api-keys")
    if not api_key:
        st.warning("Enter OpenAI API key!")
        st.stop()
    os.environ["OPENAI_API_KEY"] = api_key

# SIMPLIFIED RAG CHAIN (No deprecated imports!)
def create_simple_rag(vectorstore):
    retriever = vectorstore.as_retriever(search_kwargs={"k": 4})
    
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    
    prompt = ChatPromptTemplate.from_template("""
    Use ONLY the following context to answer the question.
    If answer not in context, say "Not found in document."
    
    Context: {context}
    
    Question: {question}
    
    Answer:""")
    
    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)
    
    chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    return chain

# UI
st.title(" RAG PDF Chatbot")
st.markdown("*Upload PDF → Process → Chat instantly!*")

col1, col2 = st.columns([3, 1])

with col1:
    uploaded_file = st.file_uploader("Choose PDF", type="pdf")
    if uploaded_file:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(uploaded_file.getvalue())
            pdf_path = tmp.name
        
        if st.button("Process PDF", type="primary"):
            with st.spinner("Processing..."):
                st.session_state.vectorstore = process_pdf(pdf_path)
                st.session_state.messages = []
                st.success("Ready to chat!")

with col2:
    if st.session_state.vectorstore:
        st.success("PDF Loaded!")
    else:
        st.info("Upload PDF first")

# Chat
if st.session_state.vectorstore:
    chain = create_simple_rag(st.session_state.vectorstore)
    
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])
    
    if prompt := st.chat_input("Ask about your PDF..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.write(prompt)
        
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                response = chain.invoke(prompt)
                st.write(response)
                st.session_state.messages.append({"role": "assistant", "content": response})
    
    if st.button("Clear Chat"):
        st.session_state.messages = []
        st.rerun()

else:
    st.info(" Upload & process PDF first!")


