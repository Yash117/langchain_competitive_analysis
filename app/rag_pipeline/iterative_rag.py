from typing import List, Dict
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_community.retrievers import ReformerRetriever

class ReformerModel:
    def __init__(self, reformer_model_name: str, index_path: str):
        self.embeddings = HuggingFaceEmbeddings(model_name=reformer_model_name)
        self.retriever = ReformerRetriever.from_pretrained(reformer_model_name, index_path)

    def retrieve_documents(self, query: str, top_k: int = 5) -> List[Dict]:
        docs = self.retriever.get_relevant_documents(query, top_k=top_k)
        return [{"content": doc.page_content, "metadata": doc.metadata} for doc in docs]
