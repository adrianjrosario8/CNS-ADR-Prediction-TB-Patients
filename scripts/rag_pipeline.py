import os
import streamlit as st

from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from sentence_transformers import SentenceTransformer
from langchain_community.vectorstores import FAISS

# =========================
# API KEY HANDLING (SAFE)
# =========================

GROQ_API_KEY = (
    st.secrets.get("GROQ_API_KEY", None)
    if hasattr(st, "secrets")
    else os.getenv("GROQ_API_KEY")
)

if not GROQ_API_KEY:
    raise ValueError(
        "GROQ_API_KEY not found. Add it in Streamlit secrets or environment variables."
    )

# =========================
# LLM INITIALIZATION
# =========================

llm = ChatGroq(
    groq_api_key=GROQ_API_KEY,
    model_name="llama-3.1-8b-instant",
    temperature=0.2
)

# =========================
# EMBEDDING MODEL
# =========================

embedder = SentenceTransformer("all-MiniLM-L6-v2")

# =========================
# PROMPT TEMPLATE
# =========================

prompt = PromptTemplate(
    input_variables=["context", "question"],
    template="""
You are a clinical pharmacology assistant.

Use ONLY the provided context to answer.
If context is insufficient, clearly state: "No sufficient evidence retrieved."

Context:
{context}

Question:
{question}

Provide:
1. Key TB drug–CNS toxicity associations
2. Literature summary (drug safety focus)
3. Population-level interpretation
4. Safety note (no diagnosis)
5. References (only if present in context)
"""
)

# =========================
# VECTOR DB (SAFE LOAD)
# =========================

def load_vectorstore(path="faiss_index"):
    if not os.path.exists(path):
        return None
    return FAISS.load_local(
        path,
        embedder,
        allow_dangerous_deserialization=True
    )

vectorstore = load_vectorstore()

# =========================
# MAIN FUNCTION (CLEAN)
# =========================

def generate_clinical_rationale(query: str):
    """
    Returns structured clinical rationale from retrieved evidence.
    """

    # -------------------------
    # Guard: missing vector DB
    # -------------------------
    if vectorstore is None:
        return "No evidence base available (FAISS index missing)."

    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    docs = retriever.get_relevant_documents(query)

    # -------------------------
    # Guard: empty retrieval
    # -------------------------
    if not docs or len(docs) == 0:
        return (
            "No relevant evidence retrieved from knowledge base. "
            "Unable to generate literature-backed clinical rationale."
        )

    context = "\n\n".join([doc.page_content for doc in docs])

    # -------------------------
    # LLM call
    # -------------------------
    response = llm.invoke(
        prompt.format(context=context, question=query)
    )

    # -------------------------
    # Clean output (avoid duplicates / artifacts)
    # -------------------------
    output = response.content.strip()

    # Remove accidental repeated blocks (simple dedupe safeguard)
    lines = output.split("\n")
    seen = set()
    cleaned = []

    for line in lines:
        if line.strip() not in seen:
            cleaned.append(line)
            seen.add(line.strip())

    return "\n".join(cleaned)