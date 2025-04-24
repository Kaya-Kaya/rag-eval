# RAG Evaluation Framework

This project provides a framework for evaluating Retrieval-Augmented Generation (RAG) pipelines. It includes tools for integrating RAG pipelines, running evaluations, and assessing the quality of responses using answer relevancy and faithfulness metrics.

## Features

- Abstract base classes for defining RAG pipelines and LLMs.
- Evaluation loader to run tests and measure performance.
- Metrics for assessing answer relevancy and faithfulness.

## Project Structure

- `src/rag_pipeline.py`: Defines abstract base classes for RAG pipelines and LLMs.
- `src/evaluation_loader.py`: Contains the evaluation loader and logic for running evaluations.

## Installation

Install the package using pip:
```bash
pip install git+https://github.com/Kaya-Kaya/rag-eval
```

## Usage

1. Define your custom RAG pipeline and LLM by implementing the abstract methods in `RAGPipeline` and `LLM` classes.

2. Prepare a `tests.json` file with the following structure:
   ```json
   [
       {
           "Q": "What is the capital of France?",
           "A": "Paris"
       },
       {
           "Q": "Who wrote '1984'?",
           "A": "George Orwell"
       }
   ]
   ```

3. Use the `Evaluator` class to run the evaluation:
   ```python
   from rag_eval.evaluation_loader import Evaluator
   from rag_eval.rag_pipeline import LLM

   class MyLLM(LLM):
       def chat(self, user_message: dict):
           # Implement your LLM logic here
           return "Example response"

   llm = MyLLM()
   evaluator = Evaluator(llm, eval_json_path="tests.json")
   evaluator.run_evaluations()
   ```

## Example

Here is an example of how to implement a custom LLM:
```python
from src.rag_pipeline import LLM

class MyLLM(LLM):
    def chat(self, user_message: dict):
        # Implement your LLM logic here
        return "Example response"
```

## License

This project is licensed under the MIT License.
