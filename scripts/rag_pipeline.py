import streamlit as st

from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from sentence_transformers import SentenceTransformer
from langchain_community.vectorstores import FAISS



# API KEY


GROQ_API_KEY = st.secrets["GROQ_API_KEY"]



# LLM


llm = ChatGroq(
    groq_api_key=GROQ_API_KEY,
    model_name="llama-3.1-8b-instant",
    temperature=0.3
)



# EMBEDDINGS


embedder = SentenceTransformer(
    "all-MiniLM-L6-v2"
)



# VECTOR STORE


vectorstore = FAISS.load_local(
    "vector_store/faiss_index",
    embedder,
    allow_dangerous_deserialization=True
)



# PROMPT TEMPLATE


prompt = PromptTemplate(
    input_variables=["context", "question"],
    template="""
You are a clinical pharmacology and TB drug safety assistant.

Use ONLY the provided literature context.

Generate a concise evidence-based clinical interpretation.

Focus on:
- TB drug-related CNS toxicity
- HIV-associated TB complications
- Neurotoxicity risk factors
- Treatment complexity
- Population-level safety interpretation

Do NOT mention missing evidence.
Do NOT say "insufficient context."
Do NOT invent studies or PMIDs.

Context:
{context}

Patient Case:
{question}

Provide output EXACTLY in this structure:

Key TB-CNS Associations:
1.
2.
3.

Literature Summary:

Population-Level Interpretation:

Safety Note (No Diagnosis):
"""
)



# MAIN FUNCTION


def generate_clinical_rationale(query):

    retriever = vectorstore.as_retriever(
        search_kwargs={"k": 3}
    )

    docs = retriever.invoke(query)

    context = "\n\n".join([
        doc.page_content for doc in docs
    ])

    response = llm.invoke(
        prompt.format(
            context=context,
            question=query
        )
    )

    answer = response.content

    sources = []

    for doc in docs:

        metadata = doc.metadata

        pmid = metadata.get("pmid", "Unknown")
        title = metadata.get("title", "Unknown Title")

        sources.append(
            f"PMID {pmid} — {title}"
        )

    return answer, sources