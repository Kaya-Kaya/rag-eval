from abc import ABC, abstractmethod
from typing import List, Any, Dict

class RAGPipeline(ABC):
    @abstractmethod
    def search(self, query: str, k: int = 3) -> List[Any]:
        """
        Obtains the top k most relevant documents to the query.
        """
        pass

    @abstractmethod
    def obtain_query_with_documents(self, query: str, k: int = 3) -> Dict:
        """
        Creates a user message with the given query and relevant
        documents.
        """
        pass
  
class LLM(ABC):
    @abstractmethod
    def chat(self, user_message: Dict) -> str:
        pass