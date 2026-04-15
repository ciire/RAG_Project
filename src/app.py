import os
from database import get_all_documents, HybridRetriever
from dotenv import load_dotenv
from groq import Groq

# Setup paths and environment
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
dotenv_path = os.path.join(BASE_DIR, "..", ".env")
load_dotenv(dotenv_path, override=True)

client = Groq(api_key=os.getenv("GROQ_API_KEY"))

# --- INITIALIZATION ---
# This runs once when the server/script starts
print("--- Initializing Retriever and Database ---")
chroma_db, all_docs = get_all_documents()
retriever = HybridRetriever(chroma_db, all_docs)

def assess_context_quality(context_chunks):
    """
    Evaluates if we found useful information. 
    Since the Reranker now returns a list of Documents (without raw scores),
    we verify quality by checking if the list is non-empty.
    """
    if not context_chunks:
        return 0.0
    # In a reranked system, if chunks made it through the filter, 
    # we treat them as high-quality (1.0).
    return 1.0

def generate_answer(question, return_context=False):
    """
    Retrieves context using the Hybrid + Reranker pipeline 
    and generates a response using Llama 3.3 70B.
    """
    # 1. Get reranked documents from database.py
    # context_chunks is now a list of Document objects
    context_chunks = retriever.query(question, k=3)

    # 2. Extract text from documents (fixed the unpacking error here)
    context_list = [doc.page_content for doc in context_chunks]
    context_text = "\n\n".join(context_list)
    
    # 3. Determine prompt strategy
    avg_score = assess_context_quality(context_chunks)
    CONFIDENCE_THRESHOLD = 0.8 

    if avg_score > CONFIDENCE_THRESHOLD:
        # High confidence: Blend docs with model knowledge
        prompt = f"""Answer this React question using BOTH the provided documentation 
AND your own knowledge. Clearly indicate which parts come from the docs vs your training.

Question: {question}

React Docs Context:
{context_text}"""
    else:
        # Low confidence/No results: Stick strictly to what is available
        prompt = f"""Use this React documentation to answer the question.
If the answer is not in the context, say you don't know. 
Stick closely to the provided context.

Question: {question}

Context:
{context_text}"""

    # 4. Generate completion with the stable 70B model
    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}]
    )

    answer = response.choices[0].message.content

    if return_context:
        return answer, context_list
    
    return answer