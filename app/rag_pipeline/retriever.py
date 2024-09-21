from typing import Dict

class Retriever:
    def __init__(self, models: Dict):
        self.models = models

    def retrieve_documents(self, query: str, model_name: str, top_k: int = 5) -> Dict:
        if model_name not in self.models:
            raise ValueError(f"Invalid model name: {model_name}. Available models are: {list(self.models.keys())}")
        return self.models[model_name].retrieve_documents(query, top_k)
