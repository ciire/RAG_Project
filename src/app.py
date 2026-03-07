import os
from database import query_db  # Import your existing search logic
from dotenv import load_dotenv
from groq import Groq

print("here")
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
dotenv_path = os.path.join(BASE_DIR, "..", ".env")
load_dotenv(dotenv_path, override=True)

# 1. Setup the Public LLM
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

def generate_answer(question):
    # 2. Retrieval: Get the "Ground Truth" from your local Chroma DB
    # We call the function you already wrote!
    context_chunks = query_db(question, k=3)
    
    # 3. Prompting: Build the message for the LLM
    context_text = "\n\n".join([doc.page_content for doc, _ in context_chunks])
    
    prompt = f"Use this React documentation to answer: {question}\n\nContext: {context_text}"
    
    # 4. Generation: Send to the Public LLM
    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}]
    )
    
    return response.choices[0].message.content