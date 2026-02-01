import os
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

# Define paths relative to this file
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PERSIST_DIR = os.path.join(BASE_DIR, "..", "chroma_db")

def get_embedding_function():
    """Centralized embedding model so it's consistent across the app."""
    return HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

def save_to_chroma(chunks):
    """Takes chunks and saves them to the local ChromaDB."""
    print(f"--- Saving {len(chunks)} chunks to {PERSIST_DIR} ---")
    
    vectorstore = Chroma.from_documents(
        documents=chunks, 
        embedding=get_embedding_function(),
        persist_directory=PERSIST_DIR
    )
    print("--- Database successfully created/updated ---")
    return vectorstore

def load_chroma():
    """Loads the existing database from disk for querying."""
    return Chroma(
        persist_directory=PERSIST_DIR,
        embedding_function=get_embedding_function()
    )