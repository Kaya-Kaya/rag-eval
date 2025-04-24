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

1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/rag-eval.git
   cd rag-eval
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
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

3. Run the evaluation:
   ```bash
   python -m src.evaluation_loader
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
