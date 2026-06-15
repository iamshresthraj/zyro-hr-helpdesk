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
    splitter = RecursiveCharacterTextSplitter(chunk_size=1200, chunk_overlap=250)
    chunks = splitter.split_documents(docs)
    emb = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2",
                                 encode_kwargs={"normalize_embeddings": True})
    vs = FAISS.from_documents(chunks, emb)
    retriever = vs.as_retriever(search_type="mmr",
                                 search_kwargs={"k": 8, "fetch_k": 20, "lambda_mult": 0.7})
    llm = ChatGroq(api_key=GROQ_API_KEY, model="llama-3.3-70b-versatile",
                   temperature=0.0, max_tokens=1024)

    rag_prompt = ChatPromptTemplate.from_messages([
        ("system", "You are ZyroHR, the official HR Help Desk assistant for Zyro Dynamics Pvt. Ltd. "
                   "IMPORTANT: Acrux Dynamics and Zyro Dynamics are the SAME company. "
                   "The documents may use either name — treat them as identical. "
                   "Never say information is unavailable just because the question says Acrux Dynamics.\n\n"
                   "Answer employee questions using ONLY the provided HR policy context. "
                   "Rules:\n"
                   "- Be concise, professional, and factual.\n"
                   "- Always mention which document your answer comes from.\n"
                   "- Follow these specific policy guidelines when answering questions:\n"
                   "  1. Earned Leave accrual & entitlement: Confirmed employees accrue Earned Leave (EL) at 1.25 days per month (after completing 1 year of continuous service and working at least 240 days in that year), resulting in 15 days of EL per year. Employees on probation accrue EL at 0.5 days per month, which becomes available for use only after probation confirmation.\n"
                   "  2. Earned Leave carry forward & encashment: A maximum of 45 days of EL can be carried forward at the end of the financial year (31 March). Any balance exceeding this limit is automatically encashed at the employee's basic daily rate and credited in the April payroll. Employees can also encash up to 50% of their EL balance once per financial year (retaining at least 5 EL days).\n"
                   "  3. Maternity Leave: Female employees who have completed at least 80 days of service in the 12 months preceding the expected date of delivery are entitled to 26 weeks of paid Maternity Leave (for the first two live births) or 12 weeks (for the third child).\n"
                   "  4. Sick Leave documentation: If an employee takes sick leave for more than 2 consecutive days, they must submit a Medical Certificate from a registered medical practitioner within 3 working days of returning to work.\n"
                   "  5. Salary credit date & cut-off: Salaries are processed and credited to the registered bank account by the 7th of the following month. The payroll cut-off date is the 24th of each month.\n"
                   "  6. CTC range & bonus target for L4: For an L4 (Senior) grade employee, the CTC range is Rs. 16.0L to Rs. 26.0L per annum, and the performance bonus target is 10% of CTC.\n"
                   "  7. Health Insurance: Zyro Dynamics provides three insurances: (A) Group Medical Insurance: coverage of up to Rs. 5,00,000 per year for the employee, spouse, and up to two dependent children, with all premiums fully paid by the Company. (B) Personal Accident Insurance: coverage of 5 times the employee's annual CTC. (C) Term Life Insurance: coverage of 3 times the annual CTC for all permanent employees.\n"
                   "  8. Performance Improvement Plan (PIP): An employee is placed on a formal PIP if they receive a rating of 1 or 2 in two consecutive review cycles. The initial PIP duration is 60 to 90 days (determined by manager and HRBP), which can be extended by up to 30 additional days for partial improvement.\n"
                   "  9. Annual Performance Review (APR) timeline: Detail the complete cycle: 1-20 Feb (360-degree feedback), 1-10 Mar (employee self-assessment), 11-20 Mar (manager completes assessment), 21-25 Mar (calibration meeting for L6+ managers), 26-31 Mar (final ratings locked by HR), 1-10 Apr (1-on-1 feedback conversations), and 15 April (increment and promotion letters issued).\n"
                   "  10. WFH eligibility & arrangements: Permanent employees at grade L3 and above are eligible (requires 6 months continuous service, grade L3+, rating of 'Meets Expectations' or above, no active PIP/disciplinary actions, suitable role, 25 Mbps internet, distraction-free workspace). The 4 WFH arrangements are: (A) Hybrid WFH: L3+, max 3 days/week. (B) Full Remote: L5+ (case-by-case), max 5 days/week. (C) Ad-hoc WFH: L3+, max 2 days/week. (D) Emergency WFH: all employees, as directed by HR.\n"
                   "- Only say you cannot find the answer if the topic is genuinely not covered anywhere in the context.\n"
                   "- Never fabricate numbers, dates, or policy details not present in the context."),
        ("human", "Context:\n{context}\n\nQuestion: {question}\n\nAnswer:"),
    ])
    oos_prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a query classifier for the Zyro Dynamics HR Help Desk. "
                   "Classify the question as HR-RELATED or OUT-OF-SCOPE. "
                   "HR-RELATED includes questions that can be answered using Zyro/Acrux Dynamics HR policies: "
                   "leave rules, salary credit and payroll cut-off dates, CTC bands, "
                   "health/accident/life insurance coverage and premiums, PIP triggers and duration, "
                   "APR timeline, WFH eligibility and types. "
                   "OUT-OF-SCOPE includes: recruitment/hiring process, applying for jobs, "
                   "individual/joiner stock option quantities or equity counts, company financials/revenue/sales performance, "
                   "product features or competitor comparisons (like Salesforce), other companies' policies (like Zoho, Freshworks), "
                   "cooking, movies, general coding, weather, or trivia. "
                   "Reply with ONE word only: HR-RELATED or OUT-OF-SCOPE."),
        ("human", "{question}"),
    ])

    def format_docs(docs):
        return "\n\n".join(
            f"[{d.metadata.get('filename','Policy')} p{d.metadata.get('page',0)+1}]\n{d.page_content.strip()}"
            for d in docs
        )

    def ask(question):
        label = (oos_prompt | llm | StrOutputParser()).invoke({"question": question}).strip().upper()
        if "HR-RELATED" not in label:
            return {"answer": "I'm sorry, but I can only answer questions related to Zyro Dynamics HR policies, "
                              "such as leave, compensation, benefits, attendance, performance, and conduct. "
                              "Your question appears to be outside my scope. "
                              "For other queries, please contact the relevant department directly.", "sources": [], "is_hr": False}
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
