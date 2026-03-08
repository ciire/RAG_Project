import os
from database import get_all_documents, HybridRetriever
from dotenv import load_dotenv
from groq import Groq

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
dotenv_path = os.path.join(BASE_DIR, "..", ".env")
load_dotenv(dotenv_path, override=True)

client = Groq(api_key=os.getenv("GROQ_API_KEY"))

def assess_context_quality(context_chunks):
    """Check if retrieved context is actually relevant."""
    if not context_chunks:
        return 0.0
    scores = [score for _, score in context_chunks]
    return sum(scores) / len(scores)

def generate_answer(question):
    # Retrieval: build hybrid retriever and search
    chroma_db, all_docs = get_all_documents()
    retriever = HybridRetriever(chroma_db, all_docs)
    context_chunks = retriever.query(question, k=3)

    context_text = "\n\n".join([doc.page_content for doc, _ in context_chunks])
    avg_score = assess_context_quality(context_chunks)

    CONFIDENCE_THRESHOLD = 0.8  # Tune this based on your data

    if avg_score > CONFIDENCE_THRESHOLD:
        # Weak local context — let LLM supplement with its own knowledge
        prompt = f"""Answer this React question using BOTH the provided documentation 
AND your own knowledge. Clearly indicate which parts come from the docs vs your training.

Question: {question}

React Docs Context (may be incomplete):
{context_text}"""
    else:
        # Strong local context — ground the answer strictly in docs
        prompt = f"""Use this React documentation to answer the question.
Stick closely to the provided context.

Question: {question}

Context:
{context_text}"""

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}]
    )

    return response.choices[0].message.content