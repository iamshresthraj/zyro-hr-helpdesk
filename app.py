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

GROQ_API_KEY = st.secrets.get("GROQ_API_KEY", os.environ.get("GROQ_API_KEY", ""))
CORPUS_PATH  = "hr_docs/"

st.set_page_config(
    page_title="Zyro Dynamics HR Help Desk",
    page_icon="🏢",
    layout="centered",
)

st.title("Zyro Dynamics — HR Help Desk")
st.caption("Powered by RAG + Groq | Ask anything about HR policies")

@st.cache_resource(show_spinner="Loading HR policy documents...")
def load_pipeline():
    # Load PDFs
    loader = PyPDFDirectoryLoader(CORPUS_PATH)
    docs = loader.load()

    # Fix filename metadata
    for doc in docs:
        src = doc.metadata.get("source", "")
        doc.metadata["filename"] = os.path.basename(src)

    # Chunk
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=150,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_documents(docs)

    # Embed
    emb = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        encode_kwargs={"normalize_embeddings": True},
    )

    # Vector store
    vs = FAISS.from_documents(chunks, emb)
    retriever = vs.as_retriever(
        search_type="mmr",
        search_kwargs={"k": 5, "fetch_k": 20, "lambda_mult": 0.7},
    )

    # LLM
    llm = ChatGroq(
        api_key=GROQ_API_KEY,
        model="llama-3.3-70b-versatile",
        temperature=0.1,
        max_tokens=512,
    )

    # RAG prompt
    rag_prompt = ChatPromptTemplate.from_messages([
        ("system",
         "You are ZyroHR, the official HR Help Desk assistant for Zyro Dynamics Pvt. Ltd. "
         "Answer employee questions using ONLY the provided HR policy context. "
         "Rules:\n"
         "- Be concise and factual.\n"
         "- Always mention which document your answer comes from.\n"
         "- If the context does not contain the answer, say: "
         "I could not find this in the HR policy documents. "
         "Please contact hr.helpdesk@zyrodynmics.com for assistance.\n"
         "- Never fabricate information."),
        ("human", "HR Policy Context:\n{context}\n\nEmployee Question: {question}\n\nAnswer:"),
    ])

    # Classifier prompt
    oos_prompt = ChatPromptTemplate.from_messages([
        ("system",
         "You are a query classifier for the Zyro Dynamics HR Help Desk.\n"
         "Classify the question as HR-RELATED or OUT-OF-SCOPE.\n\n"
         "HR-RELATED includes anything about: leave, salary, CTC, payroll, bonus, PF, "
         "gratuity, insurance, ESOP, attendance, working hours, WFH, remote work, "
         "performance review, appraisal, PIP, promotion, demotion, termination, firing, "
         "resignation, notice period, onboarding, probation, separation, F&F settlement, "
         "exit, retirement, travel, expense, reimbursement, code of conduct, POSH, "
         "harassment, IT policy, data security, company profile, grade levels, benefits, "
         "perks, L&D budget, wellness, Zyro Dynamics policies.\n\n"
         "OUT-OF-SCOPE: cooking, sports scores, movies, general coding, weather, "
         "stock markets, news, anything unrelated to Zyro Dynamics HR.\n\n"
         "Reply with ONE word only: HR-RELATED or OUT-OF-SCOPE."),
        ("human", "{question}"),
    ])

    def format_docs(docs):
        return "\n\n".join(
            f"[{d.metadata.get('filename', 'HR Policy')} - Page {d.metadata.get('page', 0) + 1}]\n"
            f"{d.page_content.strip()}"
            for d in docs
        )

    def ask(question):
        # Classify
        label = (oos_prompt | llm | StrOutputParser()).invoke({"question": question}).strip().upper()
        if "HR-RELATED" not in label:
            return {
                "answer": "I can only answer questions related to Zyro Dynamics HR policies "
                          "such as leave, salary, benefits, attendance, conduct, and separation. "
                          "Your question is outside my scope. Please contact the relevant department directly.",
                "sources": [],
                "is_hr": False,
            }

        # Retrieve
        rdocs = retriever.invoke(question)
        context = format_docs(rdocs)

        # Generate
        chain = (
            {"context": lambda _: context, "question": RunnablePassthrough()}
            | rag_prompt
            | llm
            | StrOutputParser()
        )
        answer = chain.invoke(question)

        # Collect unique source filenames
        sources = sorted({d.metadata.get("filename", "HR Policy") for d in rdocs})

        return {"answer": answer, "sources": sources, "is_hr": True}

    return ask

# Session state
if "messages" not in st.session_state:
    st.session_state.messages = []

if not GROQ_API_KEY:
    st.error("GROQ_API_KEY not found. Add it in Streamlit Secrets.")
    st.stop()

ask = load_pipeline()

# Render chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("sources"):
            st.caption("Sources: " + " | ".join(msg["sources"]))

# Suggested questions on empty state
if not st.session_state.messages:
    st.markdown("**Suggested questions:**")
    suggestions = [
        "How many earned leaves do I get per year?",
        "What is the WFH policy for L3 employees?",
        "What is the notice period if I resign at L5?",
        "What health insurance coverage is provided?",
        "How does the annual performance review work?",
        "What are the travel expense limits for L4?",
    ]
    col1, col2 = st.columns(2)
    for i, s in enumerate(suggestions):
        with (col1 if i % 2 == 0 else col2):
            if st.button(s, key=f"s{i}", use_container_width=True):
                st.session_state.messages.append({"role": "user", "content": s})
                st.rerun()

# Handle suggestion button press
if (st.session_state.messages
        and st.session_state.messages[-1]["role"] == "user"
        and len(st.session_state.messages) % 2 == 1):
    last_q = st.session_state.messages[-1]["content"]
    with st.chat_message("user"):
        st.markdown(last_q)
    with st.chat_message("assistant"):
        with st.spinner("Searching HR policies..."):
            result = ask(last_q)
        st.markdown(result["answer"])
        if result["sources"]:
            st.caption("Sources: " + " | ".join(result["sources"]))
    st.session_state.messages.append({
        "role": "assistant",
        "content": result["answer"],
        "sources": result["sources"],
        "is_hr": result["is_hr"],
    })

# Chat input
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
    st.session_state.messages.append({
        "role": "assistant",
        "content": result["answer"],
        "sources": result["sources"],
        "is_hr": result["is_hr"],
    })
