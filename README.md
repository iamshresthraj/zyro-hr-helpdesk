# Zyro Dynamics — HR Help Desk

An AI-powered HR Help Desk chatbot built using Retrieval-Augmented Generation (RAG).

## Tech Stack
- LangChain + LCEL
- FAISS vector store
- HuggingFace Embeddings (all-MiniLM-L6-v2)
- Groq LLM (llama-3.3-70b-versatile)
- Streamlit

## How it works
1. Loads 11 Zyro Dynamics HR policy documents
2. Chunks and embeds them into a FAISS vector store
3. Retrieves relevant chunks using MMR search
4. Generates grounded answers using Groq LLM
5. Refuses out-of-scope questions gracefully

## Run locally
pip install -r requirements.txt
streamlit run app.py
