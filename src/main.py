import os
from ingest import ingest_react_repo
from database import save_to_chroma, load_chroma, PERSIST_DIR

def main():
    # 1. Check if the database folder exists in your project root
    if not os.path.exists(PERSIST_DIR):
        print("--- No existing database found. Initializing ingestion... ---")
        
        # Call your existing function from ingest.py
        chunks = ingest_react_repo()
        
        # Save them using the logic in database.py
        save_to_chroma(chunks)
        print("--- Database created successfully. ---")
    else:
        print("--- Found existing database. Loading... ---")

    # 2. Load the vector store using your database.py function
    db = load_chroma()

    # 3. Interactive Query Loop
    print("\nReact Docs AI is ready. (Type 'quit' or 'exit' to stop)")
    
    while True:
        query_text = input("\nAsk a question about React: ")
        
        if query_text.lower() in ['quit', 'exit']:
            print("Goodbye!")
            break

        if not query_text.strip():
            continue

        # 4. Search the database
        # k=3 retrieves the top 3 most relevant segments
        print(f"--- Searching for: {query_text} ---")
        results = db.similarity_search(query_text, k=3)

        # 5. Display the results
        print("\n--- Top Relevant Chunks ---")
        for i, doc in enumerate(results):
            source = doc.metadata.get("source", "Unknown")
            # Print a snippet of the content
            content_snippet = doc.page_content[:400].replace('\n', ' ')
            print(f"[{i+1}] Source: {source}")
            print(f"    Content: {content_snippet}...\n")

if __name__ == "__main__":
    main()