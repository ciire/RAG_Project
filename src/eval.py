import os
import json
import pandas as pd
from app import generate_answer
from deepeval.test_case import LLMTestCase
from deepeval.metrics import FaithfulnessMetric, AnswerRelevancyMetric, ContextualPrecisionMetric
# CORRECTED IMPORT PATH
from deepeval.models.base_model import DeepEvalBaseLLM 
from groq import Groq

class GroqLLM(DeepEvalBaseLLM):
    def __init__(self, model_name="meta-llama/llama-4-scout-17b-16e-instruct"):
        self.model_name = model_name
        self.client = Groq(api_key=os.getenv("GROQ_API_KEY"))

    def load_model(self):
        return self.client

    def generate(self, prompt: str) -> str:
        chat_completion = self.client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model=self.model_name,
        )
        return chat_completion.choices[0].message.content

    async def a_generate(self, prompt: str) -> str:
        # DeepEval uses this for concurrent evaluations
        return self.generate(prompt)

    def get_model_name(self):
        return self.model_name

def run_evaluation():
    judge_llm = GroqLLM()

    # Load your existing test_set.json
    base_path = os.path.dirname(os.path.abspath(__file__))
    test_set_path = os.path.join(base_path, "test_set.json")
    with open(test_set_path, "r") as f: 
        test_data = json.load(f)

    # Threshold 0.7 means anything lower is a "fail"
    faithfulness = FaithfulnessMetric(threshold=0.7, model=judge_llm)
    relevancy = AnswerRelevancyMetric(threshold=0.7, model=judge_llm)
    precision = ContextualPrecisionMetric(threshold=0.7, model=judge_llm)

    results = []

    for item in test_data:
        print(f"Evaluating: {item['question'][:50]}...")
        
        # Get answer from your app.py
        answer, contexts = generate_answer(item["question"], return_context=True)

        test_case = LLMTestCase(
            input=item["question"],
            actual_output=answer,
            expected_output=item["ground_truth"],
            retrieval_context=contexts
        )

        # Run the judge
        faithfulness.measure(test_case)
        relevancy.measure(test_case)
        precision.measure(test_case)

        results.append({
            "question": item["question"],
            "score_faithfulness": faithfulness.score,
            "score_relevancy": relevancy.score,
            "score_precision": precision.score,
            "reasoning": faithfulness.reason # DeepEval gives you 'why' it scored this way
        })

    df = pd.DataFrame(results)
    df.to_csv("deepeval_results.csv", index=False)
    print("\n--- Done! Results saved to deepeval_results.csv ---")

if __name__ == "__main__":
    run_evaluation()