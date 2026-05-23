import streamlit as st
from langchain_community.vectorstores import FAISS
from langchain.embeddings.base import Embeddings
from langchain_groq import ChatGroq
from langchain.schema import HumanMessage
from sentence_transformers import SentenceTransformer

GROQ_API_KEY = st.secrets["GROQ_API_KEY"]
VECTOR_STORE_PATH = "vector_store/faiss_index"

# EMBEDDING MODEL

_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

class SentenceTransformerEmbeddings(Embeddings):
    def embed_documents(self, texts):
        return _model.encode(texts).tolist()
    def embed_query(self, text):
        return _model.encode(text).tolist()

embedding_function = SentenceTransformerEmbeddings()

# LOAD VECTOR STORE

@st.cache_resource
def load_vectorstore():
    return FAISS.load_local(
        VECTOR_STORE_PATH,
        embedding_function,
        allow_dangerous_deserialization=True
    )

# PROMPT — kept short deliberately to stay within token budget

PROMPT_TEMPLATE = """You are a clinical pharmacovigilance assistant.

Patient profile:
{question}

Retrieved literature:
{context}

Using only the retrieved literature above, write a brief clinical interpretation with:
1. Key TB-CNS associations (include statistics if present in the literature)
2. Literature summary (drug safety focus)
3. Population-level interpretation
4. Safety note (monitoring recommendations only, no diagnosis)

If the literature does not contain relevant information, say so clearly. Do not fabricate data."""


def generate_clinical_rationale(patient_summary: str):

    vectorstore = load_vectorstore()

    llm = ChatGroq(
        api_key=GROQ_API_KEY,
        model_name="llama-3.3-70b-versatile",
        temperature=0.2,
        max_tokens=512
    )

    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 2}
    )

    # Retrieve and deduplicate by PMID
    raw_docs = retriever.get_relevant_documents(patient_summary)

    seen_pmids = set()
    unique_docs = []
    sources = []

    for doc in raw_docs:
        pmid = doc.metadata.get("pmid", "Unknown")
        if pmid not in seen_pmids:
            seen_pmids.add(pmid)
            unique_docs.append(doc)
            sources.append({
                "pmid": pmid,
                "title": doc.metadata.get("title", "")
            })

    # Hard cap on context length
    context = "\n\n".join([doc.page_content for doc in unique_docs])
    context = context[:2000]

    # Build and send prompt
    filled_prompt = PROMPT_TEMPLATE.format(
        question=patient_summary.strip(),
        context=context
    )

    response = llm.invoke([HumanMessage(content=filled_prompt)])
    result = response.content

    return result, sources