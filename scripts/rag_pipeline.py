import os
from dotenv import load_dotenv

from sentence_transformers import SentenceTransformer
from langchain_community.vectorstores import FAISS
from langchain.embeddings.base import Embeddings
try:
    from langchain_groq import ChatGroq
except ImportError:
    from langchain_groq.chat_models import ChatGroq
from langchain_core.prompts import PromptTemplate


# =========================
# ENV SETUP
# =========================

load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")


# =========================
# EMBEDDINGS MODEL
# =========================

model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")


class EmbeddingWrapper(Embeddings):
    def embed_documents(self, texts):
        return model.encode(texts).tolist()

    def embed_query(self, text):
        return model.encode(text).tolist()


embedding_function = EmbeddingWrapper()


# =========================
# LOAD VECTOR DB
# =========================

vectorstore = FAISS.load_local(
    "vector_store/faiss_index",
    embedding_function,
    allow_dangerous_deserialization=True
)

retriever = vectorstore.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 3}
)


# =========================
# LLM
# =========================

llm = ChatGroq(
    groq_api_key=GROQ_API_KEY,
    model_name="llama-3.1-8b-instant",
    temperature=0.2
)


# =========================
# PROMPT (CONTROLLED DOMAIN)
# =========================

prompt = PromptTemplate(
    template="""
You are a biomedical literature assistant specialized in:

- Tuberculosis treatment
- CNS adverse drug reactions
- TB pharmacovigilance literature

RULES:
- Use ONLY retrieved evidence
- Focus on TB drug-related CNS toxicity
- Avoid unrelated epidemiology or HIV-only studies unless drug-related
- Do not hallucinate missing studies

Patient:
{question}

Evidence:
{context}

OUTPUT:
1. Key TB drug–CNS toxicity associations
2. Literature summary (drug safety focus)
3. Population-level interpretation
4. Safety note (no diagnosis)
""",
    input_variables=["question", "context"]
)


# =========================
# MAIN RAG FUNCTION
# =========================

def generate_clinical_rationale(patient_summary):

    raw_docs = retriever.invoke(patient_summary)

    # =========================
    # BALANCED FILTER (FIXED)
    # =========================

    KEYWORDS = [
        "adverse",
        "toxicity",
        "reaction",
        "neurolog",
        "nervous",
        "cns",
        "central",
        "tb",
        "tuberculosis",
        "isoniazid",
        "rifampicin",
        "ethambutol",
        "pyrazinamide",
        "drug",
        "treatment",
        "pharmac"
    ]

    docs = [
        d for d in raw_docs
        if any(k in d.page_content.lower() for k in KEYWORDS)
    ]

    # =========================
    # FALLBACK (IMPORTANT FIX)
    # =========================

    if not docs:
        docs = raw_docs[:3]

    # =========================
    # BUILD CONTEXT
    # =========================

    context = "\n\n".join(
        d.page_content for d in docs
    ) if docs else "No relevant literature retrieved."

    # =========================
    # LLM CALL
    # =========================

    response = llm.invoke(
        prompt.format(question=patient_summary, context=context)
    )

    # =========================
    # SOURCES
    # =========================

    sources = [
        {
            "title": d.metadata.get("title", "Unknown"),
            "pmid": d.metadata.get("pmid", "Unknown")
        }
        for d in docs
    ]

    return response.content, sources