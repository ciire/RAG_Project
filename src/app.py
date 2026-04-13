import os
from database import get_all_documents, HybridRetriever
from dotenv import load_dotenv
from groq import Groq

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
dotenv_path = os.path.join(BASE_DIR, "..", ".env")
load_dotenv(dotenv_path, override=True)

client = Groq(api_key=os.getenv("GROQ_API_KEY"))

# --- CRITICAL FIX: INITIALIZE ONCE AT STARTUP ---
# This prevents the script from reloading the DB and embeddings for every question.
print("--- Initializing Retriever and Database ---")
chroma_db, all_docs = get_all_documents()
retriever = HybridRetriever(chroma_db, all_docs)

def assess_context_quality(context_chunks):
    if not context_chunks:
        return 0.0
    scores = [score for _, score in context_chunks]
    return sum(scores) / len(scores)

def generate_answer(question, return_context=False):
    # Use the pre-loaded retriever
    context_chunks = retriever.query(question, k=3)

    context_list = [doc.page_content for doc, _ in context_chunks]
    context_text = "\n\n".join(context_list)
    
    avg_score = assess_context_quality(context_chunks)
    CONFIDENCE_THRESHOLD = 0.8 

    if avg_score > CONFIDENCE_THRESHOLD:
        prompt = f"""Answer this React question using BOTH the provided documentation 
AND your own knowledge. Clearly indicate which parts come from the docs vs your training.

Question: {question}

React Docs Context (may be incomplete):
{context_text}"""
    else:
        prompt = f"""Use this React documentation to answer the question.
Stick closely to the provided context.

Question: {question}

Context:
{context_text}"""

    response = client.chat.completions.create(
        model="meta-llama/llama-4-scout-17b-16e-instruct",
        messages=[{"role": "user", "content": prompt}]
    )

    answer = response.choices[0].message.content

    if return_context:
        return answer, context_list
    
    return answer