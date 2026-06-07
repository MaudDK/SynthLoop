from langchain_classic.retrievers import EnsembleRetriever
from synthloop.rag.store import VectorStore
from collections import defaultdict


class HybridRetriever:
    def __init__(self, vector_store: VectorStore, k = 3):
       faiss_retriever = vector_store.vector_store.as_retriever(
           search_kwargs={"k": k}
       )
       vector_store.bm25_retriever.k = k

       self.ensemble = EnsembleRetriever(
           retrievers=[faiss_retriever, vector_store.bm25_retriever],
           weights=[0.5, 0.5]
       )
    
    def retrieve(self, query):
        docs = self.ensemble.invoke(query)
        return docs

    def multi_retrieve(self, queries):
        results = defaultdict(list)
        for query in queries:
            docs = self.retrieve(query)
            for doc in docs:
                tag = doc.metadata['name']
                results[query].append(tag)
        return results
