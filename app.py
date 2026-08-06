"""
RAG PDF Chatbot — Groq

Fixes:
  1. @st.cache_resource(process_pdf) never re-ran for a new PDF, because args
     prefixed with "_" are excluded from Streamlit's cache key. The cache is now
     keyed on the file's content hash.
  2. Uploading a new PDF without clicking "Process" left the old index active.
     A file-hash check now invalidates the index and chat log automatically.
  3. Deprecated Groq model IDs (llama3-8b-8192, llama3-70b-8192,
     mixtral-8x7b-32768) return HTTP 400 model_decommissioned — the likely
     original source of the BadRequestError. Replaced with current IDs.
  4. Chunk size / k / context cap restored to sane values (the 500-char chunks
     and 2000-char cap were a workaround for a problem that wasn't token limits).
  5. Temp files are now cleaned up instead of leaking on every rerun.
"""

import hashlib
import os
import tempfile

import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

st.set_page_config(page_title="RAG PDF Chatbot (Groq)", layout="wide")

st.session_state.setdefault("vectorstore", None)
st.session_state.setdefault("doc_id", None)     # content hash of indexed PDF
st.session_state.setdefault("doc_name", None)
st.session_state.setdefault("messages", [])

MAX_CONTEXT_CHARS = 8000   # llama-3.1-8b-instant has a large window; 2000 was overkill


# ------------------------------------------------------------- Processing
@st.cache_resource(show_spinner=False)
def load_embeddings():
    """Cached separately so MiniLM isn't reloaded on every PDF."""
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )


@st.cache_resource(show_spinner=False)
def process_pdf(_pdf_path: str, file_id: str):
    """
    THE FIX: `_pdf_path` is underscore-prefixed so Streamlit skips hashing it,
    but `file_id` is a plain arg and IS hashed. A different PDF therefore gets a
    different cache entry instead of silently reusing the previous index.
    """
    docs = PyPDFLoader(_pdf_path).load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    chunks = splitter.split_documents(docs)
    if not chunks:
        return None
    return FAISS.from_documents(chunks, load_embeddings())


def file_fingerprint(uploaded) -> str:
    return hashlib.md5(uploaded.getvalue()).hexdigest()


def reset_document_state():
    st.session_state.vectorstore = None
    st.session_state.doc_id = None
    st.session_state.doc_name = None
    st.session_state.messages = []


# ----------------------------------------------------------------- Sidebar
with st.sidebar:
    st.header("Groq API Key")
    api_key = st.text_input(
        "Groq API Key", type="password",
        help="Get a free key at https://console.groq.com",
    )
    if not api_key:
        st.warning("Enter your Groq API key to continue.")
        st.stop()
    os.environ["GROQ_API_KEY"] = api_key

    model_choice = st.selectbox(
        "Model",
        ["llama-3.1-8b-instant", "llama-3.3-70b-versatile"],
        index=0,
    )

    if st.session_state.doc_name:
        st.divider()
        st.caption(f"Indexed: **{st.session_state.doc_name}**")


# -------------------------------------------------------------- RAG chain
def create_groq_rag(vectorstore):
    retriever = vectorstore.as_retriever(search_kwargs={"k": 4})
    llm = ChatGroq(model=model_choice, temperature=0)

    prompt = ChatPromptTemplate.from_template(
        """Use ONLY the following context to answer the question.
If the answer is not in the context, say "Not found in document."

Context:
{context}

Question: {question}

Answer:"""
    )

    def format_docs(docs):
        combined = "\n\n---\n\n".join(d.page_content for d in docs)
        return combined[:MAX_CONTEXT_CHARS]

    return (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )


# --------------------------------------------------------------------- UI
st.title("RAG PDF Chatbot (Groq)")
st.markdown("*Upload PDF -> Process -> Chat with Groq LLMs for free!*")

col1, col2 = st.columns([3, 1])

with col1:
    uploaded_file = st.file_uploader("Choose PDF", type="pdf")

    if uploaded_file is None:
        if st.session_state.vectorstore is not None:
            reset_document_state()
    else:
        current_id = file_fingerprint(uploaded_file)

        # A different file invalidates the previous index and chat history.
        if st.session_state.doc_id is not None and st.session_state.doc_id != current_id:
            reset_document_state()
            st.warning(
                "New PDF detected — previous document cleared. "
                "Click **Process PDF** to index this one."
            )

        already_indexed = st.session_state.doc_id == current_id

        if st.button("Process PDF", type="primary", disabled=already_indexed):
            pdf_path = None
            try:
                with st.spinner("Processing (local embeddings, no cost)..."):
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                        tmp.write(uploaded_file.getvalue())
                        pdf_path = tmp.name

                    vs = process_pdf(pdf_path, current_id)

                if vs is None:
                    st.error(
                        "No extractable text found — this PDF is likely a scan. "
                        "It would need OCR before it can be indexed."
                    )
                else:
                    st.session_state.vectorstore = vs
                    st.session_state.doc_id = current_id
                    st.session_state.doc_name = uploaded_file.name
                    st.session_state.messages = []
                    st.success("Ready to chat!")
                    st.rerun()
            except Exception as e:
                st.error(f"Failed to process PDF: {e}")
            finally:
                if pdf_path and os.path.exists(pdf_path):
                    os.unlink(pdf_path)

with col2:
    if st.session_state.vectorstore is not None:
        st.success("PDF Loaded!")
        st.caption(st.session_state.doc_name)
    else:
        st.info("Upload PDF first")


# ------------------------------------------------------------------- Chat
if st.session_state.vectorstore is not None:
    chain = create_groq_rag(st.session_state.vectorstore)
    retriever = st.session_state.vectorstore.as_retriever(search_kwargs={"k": 4})

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])

    if question := st.chat_input("Ask about your PDF..."):
        st.session_state.messages.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.write(question)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    response = chain.invoke(question)
                    st.write(response)

                    # Verify retrieval is pulling from the CURRENT document.
                    with st.expander("Sources"):
                        for i, d in enumerate(retriever.invoke(question), 1):
                            page = d.metadata.get("page", "?")
                            src = os.path.basename(str(d.metadata.get("source", "")))
                            st.markdown(f"**Chunk {i} — page {page}** `{src}`")
                            st.caption(d.page_content[:400] + "...")

                    st.session_state.messages.append(
                        {"role": "assistant", "content": response}
                    )
                except Exception as e:
                    st.error(f"Error: {e}")

    if st.button("Clear Chat"):
        st.session_state.messages = []
        st.rerun()

else:
    st.info("Upload & process a PDF first!")
