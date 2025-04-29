import json

from deepeval import evaluate
from deepeval.metrics import AnswerRelevancyMetric
from deepeval.test_case import LLMTestCase

from .rag_pipeline import LLM, RAGPipeline

class Evaluator():
    def __init__(self, llm: LLM, pipeline: RAGPipeline, tests_json_path: str = "tests.json", responses_path: str = "responses.json"):
        self.tests_json_path = tests_json_path
        self.llm = llm
        self.pipeline = pipeline
        self.responses_path = responses_path
        
    def get_llm_responses(self):
        with open(self.tests_json_path, 'r') as f:
            tests = json.load(f)

        test_cases = []

        for idx, eval_item in enumerate(tests):
            try:
                # Get LLM response
                response = self.llm.chat(self.pipeline.obtain_query_with_documents(eval_item['Q']))

                test_case = {
                    "input": eval_item['Q'],
                    "actual_output": response,
                    "expected_output": eval_item['A']
                }

                test_cases.append(test_case)

                # Print progress
                print(f"Processed {idx + 1}/{len(tests)} questions")

            except Exception as e:
                print(f"Error processing question {idx + 1}: {str(e)}")

        # Save test cases to a file
        with open(self.responses_path, "w") as f:
            json.dump(test_cases, f, indent=4)

    def run_evaluations(self):
        with open(self.responses_path, 'r') as f:
            test_cases = json.load(f)

        test_cases = [
            LLMTestCase(
                input=tc["input"],
                actual_output=tc["actual_output"],
                expected_output=tc["expected_output"]
            ) for tc in test_cases
        ]

        metrics = [
            AnswerRelevancyMetric(threshold=0.7, model="gpt-4o-mini"),
        ]

        # Run evaluations
        evaluate(test_cases, metrics=metrics)
        print("\nEvaluation Complete!")
