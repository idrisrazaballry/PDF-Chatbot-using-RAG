# PDF-Chatbot-using-RAG

This is a 100% free, local-first document Q&A system that lets you upload any PDF and chat with its content instantly. Perfect portfolio project for CSE students - demonstrates production-ready RAG skills without spending a single rupee on APIs.

WorkFlow
1. PDF UPLOAD
   User drags PDF (research paper/textbook/notes)

2. TEXT EXTRACTION  
   PyPDFLoader → Raw text from all pages
   
3. CHUNKING
   Recursive splitter → 1000-char chunks (100-char overlap)
   Why? AI can't process entire PDF at once

4. FREE EMBEDDINGS
   sentence-transformers/all-MiniLM-L6-v2
   Each chunk → 384 numbers (vector representation)

5. VECTOR STORAGE
   FAISS → Stores all vectors for lightning-fast search

6. QUESTION → ANSWER
   User: "What is main topic?"
   → Retrieve TOP 3 most similar chunks
   → Show relevant PDF sections instantly!
