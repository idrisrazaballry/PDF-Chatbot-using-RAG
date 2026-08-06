"""
RAG PDF Chatbot — Groq

Two answering modes:
  - Document mode (default): answers strictly from the uploaded PDF.
  - Hybrid mode (toggle):    falls back to the model's general knowledge when
                             the PDF doesn't cover the question, with the source
                             of every answer clearly labelled.

Earlier fixes retained:
  1. @st.cache_resource(process_pdf) never re-ran for a new PDF, because args
     prefixed with "_" are excluded from Streamlit's cache key. Now keyed on the
     file's content hash.
  2. A new upload without clicking "Process" left the old index active.
  3. Deprecated Groq model IDs return HTTP 400 model_decommissioned.
  4. Sane chunk size / k / context cap.
  5. Temp file cleanup.
"""

import hashlib
import os
import re
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

MAX_CONTEXT_CHARS = 8000
NOT_IN_CONTEXT = "NOT_IN_CONTEXT"   # sentinel the grounded prompt returns on a miss

# --------------------------------------------------------------- Smalltalk
# Social messages are not document queries. Sending them to the retriever
# guarantees a miss, which is why "thank you" came back as "Not found in
# document". These are matched and answered locally — no retrieval, no API call.
SMALLTALK = {
    "thanks": (
        "thanks", "thank you", "thanks a lot", "thank you so much", "thankyou",
        "thx", "ty", "thanks so much", "many thanks", "appreciate it",
        "thanks a ton", "cheers",
    ),
    "greeting": (
        "hi", "hello", "hey", "yo", "hii", "hiya", "good morning",
        "good afternoon", "good evening", "hello there", "hey there",
    ),
    "farewell": (
        "bye", "goodbye", "see you", "see ya", "good night", "gn", "cya",
    ),
    "affirmation": (
        "ok", "okay", "k", "cool", "nice", "great", "got it", "understood",
        "perfect", "awesome", "sure", "alright", "fine", "yes", "yep", "no",
    ),
}

SMALLTALK_REPLIES = {
    "thanks": "You're welcome! Ask me anything else about the document.",
    "greeting": "Hello! Upload a PDF and ask me anything about it.",
    "farewell": "Goodbye! Your document stays loaded if you come back.",
    "affirmation": "Got it — let me know what else you'd like to look up.",
}


def classify_smalltalk(text: str):
    """Return a smalltalk category, or None if this looks like a real question.

    Only fires when the WHOLE message is social. "thanks, now what does
    section 3 say?" must still reach the retriever, so anything longer than a
    few words, or containing a question word, is treated as a real query.
    """
    norm = re.sub(r"[^a-z\s]", " ", text.lower())
    norm = re.sub(r"\s+", " ", norm).strip()

    if not norm or len(norm.split()) > 4:
        return None
    if re.search(r"\b(what|why|how|when|where|who|which|explain|summar|list|tell)\b", norm):
        return None

    for category, phrases in SMALLTALK.items():
        if norm in phrases:
            return category
    return None


# ------------------------------------------------------------- Processing
@st.cache_resource(show_spinner=False)
def load_embeddings():
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )


@st.cache_resource(show_spinner=False)
def process_pdf(_pdf_path: str, file_id: str):
    """`_pdf_path` is skipped by the cache key; `file_id` is hashed, so a
    different PDF gets a different cache entry instead of reusing the old one."""
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

    st.divider()
    st.subheader("Answering mode")
    allow_general = st.toggle(
        "Answer beyond the PDF",
        value=False,
        help=(
            "Off: answers come only from the uploaded document.\n\n"
            "On: if the document doesn't cover the question, the model answers "
            "from its own training knowledge instead — labelled as such."
        ),
    )
    if allow_general:
        st.caption(
            "General-knowledge answers are **not** grounded in your PDF and "
            "may be wrong or out of date. Check the label on each reply."
        )

    if st.session_state.doc_name:
        st.divider()
        st.caption(f"Indexed: **{st.session_state.doc_name}**")


# ----------------------------------------------------------------- Chains
def build_llm():
    return ChatGroq(model=model_choice, temperature=0)


def build_grounded_chain(vectorstore):
    """Strict RAG. Returns the NOT_IN_CONTEXT sentinel when the PDF can't answer."""
    retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

    prompt = ChatPromptTemplate.from_template(
        """Answer the question using ONLY the context below.

If the context does not contain enough information to answer, reply with
exactly this token and nothing else: """ + NOT_IN_CONTEXT + """

Do not use any outside knowledge. Do not guess.

Context:
{context}

Question: {question}

Answer:"""
    )

    def format_docs(docs):
        return "\n\n---\n\n".join(d.page_content for d in docs)[:MAX_CONTEXT_CHARS]

    return (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | build_llm()
        | StrOutputParser()
    )


def build_general_chain():
    """No retrieval at all — the model's own knowledge, used only as a fallback."""
    prompt = ChatPromptTemplate.from_template(
        """Answer the question clearly and accurately from your own knowledge.
If you are uncertain or the topic may have changed recently, say so plainly.

Question: {question}

Answer:"""
    )
    return {"question": RunnablePassthrough()} | prompt | build_llm() | StrOutputParser()


def answer_question(question, vectorstore, allow_general):
    """Returns (answer_text, source_label, retrieved_docs)."""
    # Gate social messages before they ever touch the retriever.
    category = classify_smalltalk(question)
    if category:
        return SMALLTALK_REPLIES[category], "chat", []

    if vectorstore is not None:
        grounded = build_grounded_chain(vectorstore)
        result = grounded.invoke(question).strip()

        if NOT_IN_CONTEXT not in result.upper():
            docs = vectorstore.as_retriever(search_kwargs={"k": 4}).invoke(question)
            return result, "document", docs

        if not allow_general:
            return (
                "Not found in document. Turn on **Answer beyond the PDF** in the "
                "sidebar if you want me to answer from general knowledge instead.",
                "refused",
                [],
            )

    if not allow_general:
        return "Upload and process a PDF first.", "refused", []

    return build_general_chain().invoke(question).strip(), "general", []


SOURCE_BADGE = {
    "document": ("From your document", "green"),
    "general": ("From the model's general knowledge — not from your PDF", "orange"),
    "refused": (None, None),
    "chat": (None, None),   # social reply — no source badge needed
}


def render_badge(source):
    label, colour = SOURCE_BADGE.get(source, (None, None))
    if label:
        st.markdown(f":{colour}-badge[{label}]")


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
    elif allow_general:
        st.info("No PDF — general mode")
    else:
        st.info("Upload PDF first")


# ------------------------------------------------------------------- Chat
chat_enabled = st.session_state.vectorstore is not None or allow_general

if chat_enabled:
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            if msg["role"] == "assistant":
                render_badge(msg.get("source"))
            st.write(msg["content"])

    if question := st.chat_input("Ask a question..."):
        st.session_state.messages.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.write(question)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    answer, source, docs = answer_question(
                        question, st.session_state.vectorstore, allow_general
                    )
                except Exception as e:
                    st.error(f"Error: {e}")
                    answer, source, docs = None, None, []

            if answer is not None:
                render_badge(source)
                st.write(answer)

                if docs:
                    with st.expander("Sources"):
                        for i, d in enumerate(docs, 1):
                            page = d.metadata.get("page", "?")
                            src = os.path.basename(str(d.metadata.get("source", "")))
                            st.markdown(f"**Chunk {i} — page {page}** `{src}`")
                            st.caption(d.page_content[:400] + "...")

                st.session_state.messages.append(
                    {"role": "assistant", "content": answer, "source": source}
                )

    if st.button("Clear Chat"):
        st.session_state.messages = []
        st.rerun()

else:
    st.info(
        "Upload & process a PDF — or turn on **Answer beyond the PDF** in the "
        "sidebar to chat without one."
    )
