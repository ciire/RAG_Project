import os
from ingest import ingest_react_repo
from database import save_to_chroma, PERSIST_DIR
from app import generate_answer # Import the "Brain"


def main():
    # 1. Check if the database folder exists

    if not os.path.exists(PERSIST_DIR):
        print("--- No existing database found. Initializing ingestion... ---")
        chunks = ingest_react_repo()
        save_to_chroma(chunks)
        print("--- Database created successfully. ---")
    else:
        print("--- Found existing database. ---")

    # 2. Interactive Query Loop
    print("\nReact Docs AI is ready. (Type 'quit' or 'exit' to stop)")
    
    while True:
        query_text = input("\nAsk a question about React: ")
        
        if query_text.lower() in ['quit', 'exit']:
            print("Goodbye!")
            break

        if not query_text.strip():
            continue

        # 3. Call the AI generation logic from app.py
        # This one function handles retrieval AND the LLM response
        print(f"--- Generating answer for: {query_text} ---")
        answer = generate_answer(query_text)

        # 4. Display the finalized answer
        print("\n--- AI Response ---")
        print(answer)
        print("\n" + "="*50)

if __name__ == "__main__":
    main()