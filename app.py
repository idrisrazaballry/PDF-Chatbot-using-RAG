"""
🚀 RAG PDF Chatbot - LangChain 0.3+ Compatible
✅ 100% FREE: HuggingFace embeddings + Groq LLM (no OpenAI billing needed)
"""

import streamlit as st
import os
import tempfile
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings      # ✅ Free embeddings
from langchain_groq import ChatGroq                           # ✅ Free LLM via Groq
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
def load_embeddings():
    """
    HuggingFace sentence-transformers — runs locally, completely free.
    """
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )


@st.cache_resource
def process_pdf(_pdf_path):
    loader = PyPDFLoader(_pdf_path)
    docs = loader.load()
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, chunk_overlap=150
    )
    chunks = splitter.split_documents(docs)
    embeddings = load_embeddings()
    vectorstore = FAISS.from_documents(chunks, embeddings)
    return vectorstore


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("🔑 Groq API Key")
    st.caption("Free forever — get yours at groq.com/keys")
    groq_api_key = st.text_input(
        "Groq API Key",
        type="password",
        help="Sign up free at https://console.groq.com/keys"
    )
    if not groq_api_key:
        st.warning("⚠️ Enter your Groq API key to start chatting.")
        st.markdown("[🔗 Get a free Groq key](https://console.groq.com/keys)")
        st.stop()
    os.environ["GROQ_API_KEY"] = groq_api_key

    st.divider()
    st.markdown("**💡 100% Free Stack**")
    st.caption("📄 Embeddings: HuggingFace all-MiniLM-L6-v2 (local)")
    st.caption("🤖 Chat: Groq + LLaMA3 (free API)")


def create_rag_chain(vectorstore, api_key):
    retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

    # ✅ Free Groq LLM — LLaMA3 is fast and accurate
    llm = ChatGroq(
        model="llama3-8b-8192",
        temperature=0,
        groq_api_key=api_key
    )

    prompt = ChatPromptTemplate.from_template("""
    Use ONLY the following context to answer the question.
    If the answer is not found in the context, say "Not found in the document."

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


# ── UI ────────────────────────────────────────────────────────────────────────
st.title("📚 RAG PDF Chatbot")
st.markdown("*Upload any PDF → Process for free → Chat instantly — 100% free!*")

col1, col2 = st.columns([3, 1])

with col1:
    uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")
    if uploaded_file:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(uploaded_file.getvalue())
            pdf_path = tmp.name

        if st.button("🔄 Process PDF", type="primary"):
            with st.spinner("Processing with free HuggingFace embeddings..."):
                st.session_state.vectorstore = process_pdf(pdf_path)
                st.session_state.messages = []
                st.success("✅ PDF processed — ready to chat!")

with col2:
    if st.session_state.vectorstore:
        st.success("📄 PDF Loaded!")
    else:
        st.info("👆 Upload a PDF first")

# ── Chat Interface ─────────────────────────────────────────────────────────────
if st.session_state.vectorstore:
    chain = create_rag_chain(st.session_state.vectorstore, groq_api_key)

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])

    if prompt := st.chat_input("Ask anything about your PDF..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.write(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                response = chain.invoke(prompt)
                st.write(response)
                st.session_state.messages.append(
                    {"role": "assistant", "content": response}
                )

    if st.button("🗑️ Clear Chat"):
        st.session_state.messages = []
        st.rerun()

else:
    st.info("👆 Upload & process a PDF to start chatting!")
