import os
import streamlit as st

from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from sentence_transformers import SentenceTransformer
from langchain_community.vectorstores import FAISS
from langchain.embeddings.base import Embeddings



# API KEY


GROQ_API_KEY = None

try:
    GROQ_API_KEY = st.secrets["GROQ_API_KEY"]
except Exception:
    GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if not GROQ_API_KEY:
    raise ValueError(
        "GROQ_API_KEY not found. Add it in Streamlit secrets."
    )



# LLM


llm = ChatGroq(
    groq_api_key=GROQ_API_KEY,
    model_name="llama-3.1-8b-instant",
    temperature=0.2
)


# EMBEDDINGS


embedding_model = SentenceTransformer(
    "sentence-transformers/all-MiniLM-L6-v2"
)


class SentenceTransformerEmbeddings(Embeddings):

    def embed_documents(self, texts):
        embeddings = embedding_model.encode(texts)
        return embeddings.tolist()

    def embed_query(self, text):
        embedding = embedding_model.encode(text)
        return embedding.tolist()


embedding_function = SentenceTransformerEmbeddings()



# PROMPT


prompt = PromptTemplate(
    input_variables=["context", "question"],
    template="""
You are a clinical pharmacology assistant.

Use ONLY the provided evidence context.

If evidence is limited, explicitly state:
"No sufficient evidence retrieved."

DO NOT invent:
- PMIDs
- citations
- studies
- statistics
- conclusions not present in context

Context:
{context}

Question:
{question}

Provide:

1. Key TB drug–CNS toxicity associations

2. Literature summary (drug safety focus)

3. Population-level interpretation

4. Safety note (no diagnosis)

5. References
"""
)



# VECTOR STORE


VECTOR_DB_PATH = "vector_store/faiss_index"


def load_vectorstore():

    if not os.path.exists(VECTOR_DB_PATH):
        return None

    try:
        db = FAISS.load_local(
            VECTOR_DB_PATH,
            embedding_function,
            allow_dangerous_deserialization=True
        )

        return db

    except Exception as e:
        print(f"Vector DB loading error: {e}")
        return None


vectorstore = load_vectorstore()



# MAIN FUNCTION


def generate_clinical_rationale(query: str):

    try:

        if vectorstore is None:
            return (
                "No evidence base available (FAISS index missing).",
                []
            )

        retriever = vectorstore.as_retriever(
            search_kwargs={"k": 4}
        )

        docs = retriever.invoke(query)

        if not docs:
            return (
                "No relevant literature retrieved from the evidence base.",
                []
            )

        context = "\n\n".join(
            [doc.page_content for doc in docs]
        )

        response = llm.invoke(
            prompt.format(
                context=context,
                question=query
            )
        )

        output = response.content.strip()

        # Remove duplicated lines

        lines = output.split("\n")

        cleaned = []
        seen = set()

        for line in lines:

            stripped = line.strip()

            if stripped and stripped not in seen:
                cleaned.append(line)
                seen.add(stripped)

        final_output = "\n".join(cleaned)

        # References

        references = []

        for doc in docs:

            metadata = getattr(doc, "metadata", {})

            if "pmid" in metadata:
                references.append(
                    f"PMID {metadata['pmid']}"
                )

            elif "title" in metadata:
                references.append(
                    metadata["title"]
                )

        references = list(set(references))

        return final_output, references

    except Exception as e:

        return (
            f"RAG pipeline error: {str(e)}",
            []
        )