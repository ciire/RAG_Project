import os
from langchain_community.document_loaders import DirectoryLoader, UnstructuredMarkdownLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter, Language

def ingest_react_repo():
    # 1. Define the path to the 'content' directory
    content_path = "./react.dev/src/content"
    
    # 2. Use DirectoryLoader to find all Markdown/MDX files
    # show_progress=True is great for long loads so you know it's working
    print("--- Loading local React documents ---")
    loader = DirectoryLoader(
        content_path, 
        glob="**/*.md*", # Finds both .md and .mdx
        loader_cls=UnstructuredMarkdownLoader 
    )
    docs = loader.load()
    
    # 3. Code-Aware Splitting
    # Since these are technical docs, we use the JS Language splitter 
    # so the AI doesn't break code blocks in half.
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
    ingest_react_repo()