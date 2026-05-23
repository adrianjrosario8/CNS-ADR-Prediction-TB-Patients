import os
import pickle
import json

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

from sentence_transformers import SentenceTransformer

from langchain_community.vectorstores import FAISS
from langchain.embeddings.base import Embeddings

# File Paths

CORPUS_PATH = "data/pubmed_tb_adr.json"

VECTOR_STORE_PATH = "vector_store/faiss_index"

METADATA_PATH = "vector_store/metadata.pkl"

# Load Embedding Model

embedding_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

# Custom Langchain Embedding Class

class SentenceTransformerEmbeddings(Embeddings):
    
    def embed_documents(self, texts):
        
        embeddings = embedding_model.encode(texts)
        
        return embeddings.tolist()
    
    def embed_query(self, text):
        
        embedding = embedding_model.encode(text)
        
        return embedding.tolist()
    
embedding_function = SentenceTransformerEmbeddings()

# Load Corpus

with open(CORPUS_PATH, "r", encoding="utf-8") as f:
    
    papers = json.load(f)
    
print(f"\nLoaded {len(papers)} papers.")

# Conversion to documents

documents = []

for paper in papers:
    
    text = f"""
    
    Title: {paper['title']}
    
    Abstract: {paper['abstract']}
    
    """
    doc = Document(

        page_content=text,

        metadata={
            "pmid": paper["pmid"],
            "title": paper["title"]
        }
    )

    documents.append(doc)
    
print(f"Converted {len(documents)} papers into Langchain documents.")

# Chunk Documents

text_splitter = RecursiveCharacterTextSplitter(
    
    chunk_size=500,
    
    chunk_overlap = 100
)

split_docs = text_splitter.split_documents(documents)

print(f"Created {len(split_docs)} text chunks.")

# Create FAISS vector store

vectorstore = FAISS.from_documents(
    split_docs,
    
    embedding_function
)

print("FAISS vector store created")

# Create Output Directory

os.makedirs("../vector_store", exist_ok=True)

# Save Vector Store

vectorstore.save_local(VECTOR_STORE_PATH)

print(f"FAISS index saved to: {VECTOR_STORE_PATH}")

# Save metadta

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
                   
                   
                  
                   
                   
        
        
            

