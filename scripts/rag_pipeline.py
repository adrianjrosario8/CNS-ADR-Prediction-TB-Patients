import os
import streamlit as st
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain_groq import ChatGroq
from langchain.prompts import PromptTemplate
from langchain.embeddings.base import Embeddings
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

# PROMPT

PROMPT_TEMPLATE = """
You are a clinical pharmacovigilance assistant. Using only the retrieved literature below,
provide a concise evidence-based clinical interpretation for a TB patient with the following profile:

{question}

Structure your response as:
1. Key TB-CNS Associations (cite specific findings from the literature with statistics if available)
2. Literature Summary (drug safety focus)
3. Population-Level Interpretation
4. Safety Note (no diagnosis, monitoring recommendations only)

Base your response strictly on the provided context. Do not fabricate statistics or references.

Context:
{context}
"""

prompt = PromptTemplate(
    template=PROMPT_TEMPLATE,
    input_variables=["context", "question"]
)

# RAG CHAIN

def generate_clinical_rationale(patient_summary: str):
    
    vectorstore = load_vectorstore()

    llm = ChatGroq(
        api_key=GROQ_API_KEY,
        model_name="llama3-8b-8192",
        temperature=0.2
    )

    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 5}
    )

    # Retrieve docs and deduplicate by PMID
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

    # Build context from deduplicated docs
    context = "\n\n".join([doc.page_content for doc in unique_docs])

    # Build chain with deduplicated context
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vectorstore.as_retriever(search_kwargs={"k": 1}),
        chain_type_kwargs={"prompt": prompt},
        return_source_documents=False
    )

    # Run with manually constructed context
    from langchain.chains.question_answering import load_qa_chain
    
    doc_chain = load_qa_chain(
        llm,
        chain_type="stuff",
        prompt=prompt
    )

    result = doc_chain.run(
        input_documents=unique_docs,
        question=patient_summary
    )

    return result, sources