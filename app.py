import os
import streamlit as st
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_groq import ChatGroq

GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")
CORPUS_PATH  = "hr_docs/"

st.set_page_config(page_title="Zyro Dynamics HR Help Desk", page_icon="🏢", layout="centered")
st.title("🏢 Zyro Dynamics — HR Help Desk")
st.caption("Powered by RAG + Groq | Ask anything about HR policies")

@st.cache_resource(show_spinner="Loading HR documents...")
def load_pipeline():
    loader = PyPDFDirectoryLoader(CORPUS_PATH)
    docs = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=150)
    chunks = splitter.split_documents(docs)
    emb = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2",
                                 encode_kwargs={"normalize_embeddings": True})
    vs = FAISS.from_documents(chunks, emb)
    retriever = vs.as_retriever(search_type="mmr",
                                 search_kwargs={"k": 5, "fetch_k": 20, "lambda_mult": 0.7})
    llm = ChatGroq(api_key=GROQ_API_KEY, model="llama-3.3-70b-versatile",
                   temperature=0.1, max_tokens=512)

    rag_prompt = ChatPromptTemplate.from_messages([
        ("system", "You are ZyroHR, the official HR assistant for Zyro Dynamics. "
                   "Answer ONLY from the provided context. Be concise and cite the policy document. "
                   "If not found say: contact hr.helpdesk@zyrodynmics.com"),
        ("human", "Context:\n{context}\n\nQuestion: {question}\n\nAnswer:"),
    ])
    oos_prompt = ChatPromptTemplate.from_messages([
        ("system", "Classify as HR-RELATED or OUT-OF-SCOPE. "
                   "HR topics: leave, salary, payroll, PF, insurance, attendance, WFH, "
                   "performance, POSH, notice period, onboarding, travel, Zyro Dynamics policies. "
                   "Reply ONE word only: HR-RELATED or OUT-OF-SCOPE."),
        ("human", "{question}"),
    ])

    def format_docs(docs):
        return "\n\n".join(
            f"[{d.metadata.get('filename','Policy')} p{d.metadata.get('page',0)+1}]\n{d.page_content.strip()}"
            for d in docs
        )

    def ask(question):
        label = (oos_prompt | llm | StrOutputParser()).invoke({"question": question}).upper()
        if "HR-RELATED" not in label:
            return {"answer": "I can only answer Zyro Dynamics HR policy questions. "
                              "Your question is outside my scope.", "sources": [], "is_hr": False}
        rdocs = retriever.invoke(question)
        ctx = format_docs(rdocs)
        chain = ({"context": lambda _: ctx, "question": RunnablePassthrough()}
                 | rag_prompt | llm | StrOutputParser())
        answer = chain.invoke(question)
        sources = sorted({d.metadata.get("filename", "?") for d in rdocs})
        return {"answer": answer, "sources": sources, "is_hr": True}

    return ask

if "messages" not in st.session_state:
    st.session_state.messages = []

ask = load_pipeline()

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if prompt := st.chat_input("Ask about leave, salary, benefits, WFH, performance..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    with st.chat_message("assistant"):
        with st.spinner("Searching HR policies..."):
            result = ask(prompt)
        st.markdown(result["answer"])
        if result["sources"]:
            st.caption("Sources: " + " | ".join(result["sources"]))
        if not result["is_hr"]:
            st.warning("Outside HR scope.")
    st.session_state.messages.append({"role": "assistant", "content": result["answer"]})