import os
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder

# Define paths relative to this file
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PERSIST_DIR = os.path.join(BASE_DIR, "..", "chroma_db")

def get_embedding_function():
    """Returns the embedding model for the vector database."""
    return HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

def save_to_chroma(chunks):
    """Saves structural chunks to the Chroma vector store."""
    print(f"--- Saving {len(chunks)} chunks to {PERSIST_DIR} ---")
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=get_embedding_function(),
        persist_directory=PERSIST_DIR
    )
    print("--- Database successfully created/updated ---")
    return vectorstore

def load_chroma():
    """Loads the existing Chroma vector store from disk."""
    return Chroma(
        persist_directory=PERSIST_DIR,
        embedding_function=get_embedding_function()
    )

def get_all_documents():
    """Fetches all documents from ChromaDB to build the BM25 index."""
    db = load_chroma()
    raw = db._collection.get(include=["documents", "metadatas"])
    docs = [
        Document(page_content=text, metadata=meta)
        for text, meta in zip(raw["documents"], raw["metadatas"])
    ]
    return db, docs

class HybridRetriever:
    """
    Two-Stage Retriever:
    1. Retrieval: Hybrid search (Dense Vector + Sparse BM25) pulls top 10 candidates.
    2. Re-ranking: Cross-Encoder evaluates query/doc pairs for high-precision sorting.
    """

    def __init__(self, chroma_db, documents):
        self.chroma_db = chroma_db
        self.documents = documents
        
        # Initialize BM25 (Sparse)
        tokenized = [doc.page_content.lower().split() for doc in documents]
        self.bm25 = BM25Okapi(tokenized)
        
        # Initialize Cross-Encoder (Reranker)
        print("--- Loading Cross-Encoder Reranker (ms-marco-MiniLM-L-6-v2) ---")
        self.reranker = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')

    def query(self, query_text, k=3):
        # STAGE 1: Pull 10 candidates (more than k) to give the reranker options
        initial_pool_size = 10 

        # Dense Retrieval (Chroma)
        dense_results = self.chroma_db.similarity_search_with_score(query_text, k=initial_pool_size)
        dense_docs = {doc.page_content: doc for doc, score in dense_results}

        # Sparse Retrieval (BM25)
        tokens = query_text.lower().split()
        bm25_scores = self.bm25.get_scores(tokens)
        top_indices = sorted(range(len(bm25_scores)), key=lambda i: bm25_scores[i], reverse=True)[:initial_pool_size]
        sparse_docs = {self.documents[i].page_content: self.documents[i] for i in top_indices}

        # Deduplicate candidates from both methods
        candidate_docs = list({**sparse_docs, **dense_docs}.values())

        # STAGE 2: Rerank candidates with Cross-Encoder
        # Cross-Encoders score [Query, Document] pairs together for higher accuracy
        pairs = [[query_text, doc.page_content] for doc in candidate_docs]
        rerank_scores = self.reranker.predict(pairs)

        # Pair docs with scores and sort descending
        scored_results = sorted(zip(candidate_docs, rerank_scores), key=lambda x: x[1], reverse=True)

        # Return the top k original Document objects
        return [doc for doc, score in scored_results[:k]]