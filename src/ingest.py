import os
from langchain_community.document_loaders import DirectoryLoader, UnstructuredMarkdownLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter, Language
from database import save_to_chroma

def ingest_react_repo():
    content_path = "./react.dev/src/content"
    print("--- Loading local React documents ---")
    loader = DirectoryLoader(
        content_path,
        glob="**/*.md*",
        loader_cls=UnstructuredMarkdownLoader
    )
    docs = loader.load()

    print("--- Chunking with JS-aware splitter ---")
    splitter = RecursiveCharacterTextSplitter.from_language(
        language=Language.JS,
        chunk_size=1000,
        chunk_overlap=200
    )
    chunks = splitter.split_documents(docs)

    print(f"Successfully processed {len(chunks)} chunks from React docs.")
    return chunks

if __name__ == "__main__":
    chunks = ingest_react_repo()
    save_to_chroma(chunks)