import os
import streamlit as st

from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from sentence_transformers import SentenceTransformer
from langchain_community.vectorstores import FAISS

# API KEY HANDLING

GROQ_API_KEY = (
    st.secrets.get("GROQ_API_KEY", None)
    if hasattr(st, "secrets")
    else os.getenv("GROQ_API_KEY")
)

if not GROQ_API_KEY:
    raise ValueError(
        "GROQ_API_KEY not found. Add it in Streamlit secrets or environment variables."
    )

# LLM INITIALIZATION


llm = ChatGroq(
    groq_api_key=GROQ_API_KEY,
    model_name="llama-3.1-8b-instant",
    temperature=0.2
)


# EMBEDDING MODEL


embedder = SentenceTransformer("all-MiniLM-L6-v2")

# PROMPT TEMPLATE


prompt = PromptTemplate(
    input_variables=["context", "question"],
    template="""
You are a clinical pharmacology assistant.

Use ONLY the provided context to answer.
If context is insufficient, clearly state:
"No sufficient evidence retrieved."

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

Do NOT invent PMIDs, studies, citations, or statistics.
"""
)

# VECTOR DB LOAD


def load_vectorstore(path="faiss_index"):
    if not os.path.exists(path):
        return None

    return FAISS.load_local(
        path,
        embedder,
        allow_dangerous_deserialization=True
    )

vectorstore = load_vectorstore()

# MAIN FUNCTION


def generate_clinical_rationale(query: str):


    # Missing vector DB


    if vectorstore is None:
        return (
            "No evidence base available (FAISS index missing).",
            []
        )

    # Retrieve documents


    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

    docs = retriever.get_relevant_documents(query)

    # Empty retrieval guard
  
    if not docs or len(docs) == 0:
        return (
            "No relevant evidence retrieved from knowledge base. "
            "Unable to generate literature-backed clinical rationale.",
            []
        )

    # Build context


    context = "\n\n".join([
        doc.page_content for doc in docs
    ])

  
    # Generate response
  

    response = llm.invoke(
        prompt.format(
            context=context,
            question=query
        )
    )

    output = response.content.strip()

  
    # Remove duplicate lines


    lines = output.split("\n")

    cleaned = []
    seen = set()

    for line in lines:
        stripped = line.strip()

        if stripped and stripped not in seen:
            cleaned.append(line)
            seen.add(stripped)

    final_output = "\n".join(cleaned)

    # Extract references safely


    references = []

    for doc in docs:

        metadata = getattr(doc, "metadata", {})

        if "source" in metadata:
            references.append(metadata["source"])

        elif "pmid" in metadata:
            references.append(f"PMID {metadata['pmid']}")

    # Remove duplicates
    
    references = list(set(references))

    return final_output, references