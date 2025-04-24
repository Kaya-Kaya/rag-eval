import json

from deepeval import evaluate
from deepeval.metrics import AnswerRelevancyMetric, FaithfulnessMetric
from deepeval.test_case import LLMTestCase

from rag_pipeline import LLM

class Evaluator():
    def __init__(self, llm: LLM, eval_json_path: str = "tests.json"):
        self.eval_json_path = eval_json_path
        self.llm = llm

    def load_evaluations(self, json_path: str) -> list:
        with open(json_path, 'r') as f:
            return json.load(f)

    def run_evaluations(self):
        evaluations = self.load_evaluations(self.eval_json_path)

        test_cases = []
        metrics = [
            AnswerRelevancyMetric(threshold=0.7),
            FaithfulnessMetric(threshold=0.7)
        ]

        for idx, eval_item in enumerate(evaluations):
            try:
                # Get LLM response
                response = self.llm.chat(eval_item['Q'])

                # Create test case
                test_case = LLMTestCase(
                    input=eval_item['Q'],
                    actual_output=response,
                    expected_output=eval_item['A']
                )

                test_cases.append(test_case)

                # Print progress
                print(f"Processed {idx + 1}/{len(evaluations)} questions")

            except Exception as e:
                print(f"Error processing question {idx + 1}: {str(e)}")

        # Run evaluations
        evaluate(test_cases, metrics=metrics)
        print("\nEvaluation Complete!")
