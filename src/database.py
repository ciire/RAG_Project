import os
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from rank_bm25 import BM25Okapi

# Define paths relative to this file
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PERSIST_DIR = os.path.join(BASE_DIR, "..", "chroma_db")

def get_embedding_function():
    return HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

def save_to_chroma(chunks):
    print(f"--- Saving {len(chunks)} chunks to {PERSIST_DIR} ---")
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=get_embedding_function(),
        persist_directory=PERSIST_DIR
    )
    print("--- Database successfully created/updated ---")
    return vectorstore

def load_chroma():
    return Chroma(
        persist_directory=PERSIST_DIR,
        embedding_function=get_embedding_function()
    )

def query_db(query_text, k=3):
    db = load_chroma()
    results = db.similarity_search_with_score(query_text, k=k)
    return results

def get_all_documents():
    """Fetch all documents from ChromaDB for BM25 indexing."""
    db = load_chroma()
    raw = db._collection.get(include=["documents", "metadatas"])
    docs = [
        Document(page_content=text, metadata=meta)
        for text, meta in zip(raw["documents"], raw["metadatas"])
    ]
    return db, docs

class HybridRetriever:
    """Combines dense (Chroma) + sparse (BM25) retrieval."""

    def __init__(self, chroma_db, documents):
        self.chroma_db = chroma_db
        self.documents = documents
        tokenized = [doc.page_content.lower().split() for doc in documents]
        self.bm25 = BM25Okapi(tokenized)

    def query(self, query_text, k=3):
        # Dense retrieval from Chroma
        dense_results = self.chroma_db.similarity_search_with_score(query_text, k=k)
        dense_docs = {doc.page_content: (doc, score) for doc, score in dense_results}

        # Sparse BM25 retrieval
        tokens = query_text.lower().split()
        bm25_scores = self.bm25.get_scores(tokens)
        top_indices = sorted(range(len(bm25_scores)), key=lambda i: bm25_scores[i], reverse=True)[:k]
        sparse_docs = {
            self.documents[i].page_content: (self.documents[i], bm25_scores[i])
            for i in top_indices
        }

        # Merge, deduplicate, return top-k (dense wins on collision)
        merged = {**sparse_docs, **dense_docs}
        return list(merged.values())[:k]