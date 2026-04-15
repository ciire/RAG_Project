import os
from langchain_community.document_loaders import DirectoryLoader, UnstructuredMarkdownLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter, MarkdownHeaderTextSplitter
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

    # 1. Define the headers to split on
    headers_to_split_on = [
        ("#", "Header 1"),
        ("##", "Header 2"),
        ("###", "Header 3"),
    ]

    # 2. Initialize the Markdown Splitter
    markdown_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on)

    # 3. Initialize the Recursive Splitter as a fallback for massive sections
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, 
        chunk_overlap=100
    )

    final_chunks = []

    print("--- Splitting by Markdown Structure ---")
    for doc in docs:
        # Split by header
        header_splits = markdown_splitter.split_text(doc.page_content)
        
        # Further split any chunks that are still too big
        # We also carry over the metadata so the AI knows which header it's reading
        for split in header_splits:
            split.metadata.update(doc.metadata) # Keep original file info
            if len(split.page_content) > 1000:
                sub_splits = text_splitter.split_documents([split])
                final_chunks.extend(sub_splits)
            else:
                final_chunks.append(split)

    print(f"Successfully processed {len(final_chunks)} structural chunks.")
    return final_chunks

if __name__ == "__main__":
    chunks = ingest_react_repo()
    save_to_chroma(chunks)