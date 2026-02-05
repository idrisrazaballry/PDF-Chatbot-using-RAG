import streamlit as st
import tempfile
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_huggingface import HuggingFacePipeline
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
import torch

# Config
st.set_page_config(layout="wide")
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None

st.title("📚 FREE RAG PDF Chatbot")
st.markdown("**Zero cost, local models only!**")

# File upload
uploaded_file = st.file_uploader("Upload PDF", type="pdf")
if uploaded_file:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(uploaded_file.getvalue())
        pdf_path = tmp.name
    
    if st.button("🔄 Process PDF", type="primary"):
        with st.spinner("Processing PDF (local)..."):
            # Load & split
            loader = PyPDFLoader(pdf_path)
            docs = loader.load()
            splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
            chunks = splitter.split_documents(docs)
            
            # FREE local embeddings
            embeddings = HuggingFaceEmbeddings(
                model_name="sentence-transformers/all-MiniLM-L6-v2"
            )
            
            # Create vector store
            st.session_state.vectorstore = FAISS.from_documents(chunks, embeddings)
            st.success("✅ FREE local processing complete!")

# Chat (FREE local LLM)
if st.session_state.vectorstore:
    st.subheader("💬 Chat with your PDF")
    
    # FREE local model (Phi-3 mini - fast & good)
    @st.cache_resource
    def load_llm():
        model_id = "microsoft/Phi-3-mini-4k-instruct"
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        model = AutoModelForCausalLM.from_pretrained(
            model_id, 
            torch_dtype=torch.float16, 
            device_map="auto", 
            trust_remote_code=True
        )
        pipe = pipeline(
            "text-generation", 
            model=model, 
            tokenizer=tokenizer, 
            max_new_tokens=256,
            temperature=0.1,
            do_sample=True
        )
        return HuggingFacePipeline(pipeline=pipe)

    retriever = st.session_state.vectorstore.as_retriever()
    llm = load_llm()
    
    # Chat history
    for msg in st.session_state.get("messages", []):
        with st.chat_message(msg["role"]):
            st.write(msg["content"])
    
    if question := st.chat_input("Ask about your PDF..."):
        st.session_state.setdefault("messages", []).append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.write(question)
        
        with st.chat_message("assistant"):
            # Retrieve relevant docs
            docs = retriever.invoke(question)
            context = "\n\n".join([d.page_content for d in docs[:3]])
            
            # Prompt for local LLM
            prompt = f"""Use only this context to answer the question.
            
Context: {context}
            
Question: {question}
            
Answer:"""
            
            response = llm.invoke(prompt)
            answer = response.strip()
            
            st.write(answer)
            st.session_state["messages"].append({"role": "assistant", "content": answer})

else:
    st.info("👆 Upload PDF → Process → Chat FREE!")
    
    st.markdown("""
    ### 🌟 Why This is Perfect for You:
    - ✅ **₹0 cost forever**
    - ✅ Offline after first download  
    - ✅ Portfolio-ready (shows local AI skills)
    - ✅ Fast on laptop (Phi-3 mini)
    - ✅ Real RAG implementation
    """)
