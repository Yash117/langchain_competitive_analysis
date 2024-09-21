from typing import List, Dict
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.retrievers import BM25Retriever, EnsembleRetriever

class SpladeModel:
    def __init__(self, 
                 bert_model_name: str, 
                 vector_store: str = "faiss",
                 ):
        self.embeddings = HuggingFaceEmbeddings(model_name=bert_model_name)
        self.bm25_retriever = BM25Retriever.from_texts(
            # Load your actual data for BM25 initialization
            ["Document 1", "Document 2", "Document 3"] 
        )
        self.dense_retriever = EnsembleRetriever(
            retrievers=[self.bm25_retriever],
            weights=[1.0],  # Adjust weights if using multiple retrievers
            embedding=self.embeddings,
        )

    def retrieve_documents(self, query: str, top_k: int = 5) -> List[Dict]:
        docs = self.dense_retriever.get_relevant_documents(query, top_k=top_k)
        return [{"content": doc.page_content, "metadata": doc.metadata} for doc in docs]
