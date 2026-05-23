import os
import json
import pickle

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

from sentence_transformers import SentenceTransformer

from langchain_community.vectorstores import FAISS
from langchain.embeddings.base import Embeddings


# PATHS

CORPUS_PATH = "data/pubmed_tb_adr.json"
VECTOR_STORE_PATH = "vector_store/faiss_index"
METADATA_PATH = "vector_store/metadata.pkl"


# EMBEDDING MODEL

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


# LOAD CORPUS

with open(CORPUS_PATH, "r", encoding="utf-8") as f:
    papers = json.load(f)

print(f"\nLoaded {len(papers)} papers.")


# DEDUPLICATE BY PMID

seen_pmids = set()
unique_papers = []

for paper in papers:
    pmid = paper.get("pmid", "Unknown")
    if pmid not in seen_pmids:
        seen_pmids.add(pmid)
        unique_papers.append(paper)

duplicates_removed = len(papers) - len(unique_papers)
print(f"Removed {duplicates_removed} duplicate PMIDs. {len(unique_papers)} unique papers retained.")


# CONVERT TO DOCUMENTS

documents = []

for paper in unique_papers:
    title = paper.get("title", "")
    abstract = paper.get("abstract", "")
    pmid = paper.get("pmid", "Unknown")

    # Skip papers with no usable abstract
    if not abstract.strip():
        print(f"Skipping PMID {pmid} — no abstract.")
        continue

    text = f"""Title: {title}

Abstract:
{abstract}"""

    doc = Document(
        page_content=text,
        metadata={
            "pmid": pmid,
            "title": title
        }
    )

    documents.append(doc)

print(f"Converted {len(documents)} papers into documents.")


# CHUNKING

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=100
)

split_docs = text_splitter.split_documents(documents)

print(f"Created {len(split_docs)} chunks.")


# CREATE VECTOR STORE

vectorstore = FAISS.from_documents(
    split_docs,
    embedding_function
)

print("FAISS vector store created.")


# SAVE VECTOR STORE

os.makedirs("vector_store", exist_ok=True)

vectorstore.save_local(VECTOR_STORE_PATH)

print(f"Saved vector DB to: {VECTOR_STORE_PATH}")


# SAVE METADATA

metadata = [
    {
        "content": doc.page_content,
        "metadata": doc.metadata
    }
    for doc in split_docs
]

with open(METADATA_PATH, "wb") as f:
    pickle.dump(metadata, f)

print(f"Metadata saved to: {METADATA_PATH}")