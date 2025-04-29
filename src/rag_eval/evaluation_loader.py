import json

from deepeval import evaluate
from deepeval.metrics import AnswerRelevancyMetric, GEval
from deepeval.test_case import LLMTestCase, LLMTestCaseParams

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
            GEval(
                name="Correctness",
                evaluation_steps=[
                    "If 'actual output' sufficiently answer the 'query' then it's a good output. 'Actual output' doesn't need to strictly repeat the information from 'expected output'"
                    "Only use 'expected ouput' to check if the facts in 'actual output' contradicts any facts in 'expected output'",
                    "If there is information in 'expected output' that isn't mentioned in the 'actual output', check it's relevant to the query to evaluate that information",
                    "do not penalize the omission of insignificant words and relevant information",
                    "actual output's having differnt writing style and grammar from the expected output's is OK"
                ],
                #criteria="Determine if the 'actual output' is factually correct based on the 'expected output'. Do not penalize for different format, structure, wording or unncessary information as long as the fact provided is correct"
                model="gpt-4o-mini",
                evaluation_params=[LLMTestCaseParams.ACTUAL_OUTPUT, LLMTestCaseParams.EXPECTED_OUTPUT],
                threshold=0.5
            )
        ]

        # Run evaluations
        evaluate(test_cases, metrics=metrics)
        print("\nEvaluation Complete!")
