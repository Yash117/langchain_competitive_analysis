from typing import List, Dict
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_community.retrievers import DPRRetriever

class DprModel:
    def __init__(self, dpr_model_name: str, index_path: str):
        self.embeddings = HuggingFaceEmbeddings(model_name=dpr_model_name)
        self.retriever = DPRRetriever.from_pretrained(dpr_model_name, index_path)

    def retrieve_documents(self, query: str, top_k: int = 5) -> List[Dict]:
        docs = self.retriever.get_relevant_documents(query, top_k=top_k)
        return [{"content": doc.page_content, "metadata": doc.metadata} for doc in docs]
