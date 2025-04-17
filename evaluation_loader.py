
import json
from deepeval import evaluate
from deepeval.metrics import AnswerRelevancyMetric, FaithfulnessMetric
from deepeval.test_case import LLMTestCase
from dotenv import load_dotenv

load_dotenv()
# Configuration
EVAL_JSON_PATH = "shorttestjsonthing.json" #choose json here
LLM_CHOICE = "openai"  # choose llm here


# insert different llm query function here

def query_openai(question: str) -> str:
    """OpenAI implementation using ChatCompletion"""
    from openai import OpenAI
    try:
        client = OpenAI()
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": question}]
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"OpenAI Error: {str(e)}"


def query_anthropic(question: str) -> str:

    from anthropic import Anthropic
    try:
        client = Anthropic(api_key="your-api-key-here")
        response = client.messages.create(
            model="claude-3-opus-20240229",
            max_tokens=1000,
            messages=[{"role": "user", "content": question}]
        )
        return response.content[0].text
    except Exception as e:
        return f"Anthropic Error: {str(e)}"


def query_llama(question: str) -> str:

    from transformers import pipeline
    try:
        generator = pipeline('text-generation', model='meta-llama/Llama-2-7b-chat-hf')
        response = generator(question, max_length=200)
        return response[0]['generated_text']
    except Exception as e:
        return f"Llama Error: {str(e)}"


def get_llm_response(question: str) -> str:
    """Switch between LLM implementations"""
    if LLM_CHOICE == "openai":
        return query_openai(question)
    elif LLM_CHOICE == "anthropic":
        return query_anthropic(question)
    elif LLM_CHOICE == "llama":
        return query_llama(question)
    else:
        raise ValueError(f"Invalid LLM choice: {LLM_CHOICE}")


# eval

def load_evaluations(json_path: str) -> list:
    #load json
    with open(json_path, 'r') as f:
        return json.load(f)


def run_evaluations():

    evaluations = load_evaluations(EVAL_JSON_PATH)

    test_cases = []
    metrics = [
        AnswerRelevancyMetric(threshold=0.7),
        FaithfulnessMetric(threshold=0.7)
    ]

    for idx, eval_item in enumerate(evaluations):
        try:
            # Get LLM response
            response = get_llm_response(eval_item['Q'])

            # Create test case
            test_case = LLMTestCase(
                input=eval_item['Q'],
                actual_output=response,
                expected_output=eval_item['A'],
                context=["Course Policy Questions"]  # TODO: Add context for each question in the JSON??
            )

            test_cases.append(test_case)

            # Print progress
            print(f"Processed {idx + 1}/{len(evaluations)} questions")

        except Exception as e:
            print(f"Error processing question {idx + 1}: {str(e)}")

    # Run evaluations
    evaluate(test_cases, metrics=metrics)
    print("\nEvaluation Complete!")


if __name__ == "__main__":
    run_evaluations()