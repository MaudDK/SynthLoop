import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
from langchain_community.vectorstores import FAISS
from langchain_community.retrievers import BM25Retriever
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document
from langchain_community.document_loaders import CSVLoader, PyPDFLoader
import pandas as pd
import pickle

class VectorStore:
    def __init__(self, embeddings = None, save_path = "./vector_store"):
        self.embeddings = embeddings or HuggingFaceEmbeddings(
            model_name="Qwen/Qwen3-Embedding-0.6B",
            model_kwargs={"device": "cuda"},
            encode_kwargs={
                "batch_size": 16,
                "normalize_embeddings": True
            }
        )
        self.documents = []
        self.vector_store = None
        self.save_path = save_path
        self.bm25_retriever = None

    def build(self, data, content_columns=None, metadata_columns=None, separator="\n", source=None):
        if isinstance(data, pd.DataFrame):
            self.documents = self._df_to_documents(data, content_columns, metadata_columns, separator, source)
        elif isinstance(data, str):
            if data.endswith(".csv"):
                self.documents = self._csv_to_documents(data, content_columns, metadata_columns)
            elif data.endswith(".pdf"):
                self.documents = self._pdf_to_documents(data)
            else:
                raise ValueError("Unsupported file type", data.split(".")[-1])
        else:
            raise ValueError("Unsupported data type")
        return self

    def index(self):
        if not self.documents:
            raise ValueError("No documents to index")

        print(f"Indexing {len(self.documents)} documents ...")
        self.embeddings.show_progress = True 
        self.vector_store = FAISS.from_documents(self.documents, self.embeddings)
        print("Building BM25 index...")
        self.bm25_retriever = BM25Retriever.from_documents(self.documents)
        self.save()
        return self

    def save(self):
        if not self.vector_store:
            raise ValueError("No vector store to save")

        self.vector_store.save_local(self.save_path)
        with open(f"{self.save_path}/bm25.pkl", "wb") as f:
            pickle.dump(self.bm25_retriever, f)
        
        return self

    @classmethod
    def load(cls, path):
        vector_store = cls()
        vector_store.vector_store = FAISS.load_local(
            path,
            embeddings=vector_store.embeddings,
            allow_dangerous_deserialization=True
        )
        with open(f"{path}/bm25.pkl", "rb") as f:
            vector_store.bm25_retriever = pickle.load(f)
        return vector_store

    def search(self, query: str, k: int = 10) -> list[tuple]:
        if not self.vector_store:
            raise ValueError("No vector store to search")
        return self.vector_store.similarity_search_with_score(query, k=k)


    @staticmethod
    def _csv_to_documents(csv_path, content_columns, metadata_columns):
        loader = CSVLoader(
            file_path=csv_path,
            metadata_columns=metadata_columns,
            content_columns=content_columns
        )

        return loader.load()

    @staticmethod
    def _df_to_documents(df, content_columns, metadata_columns, separator = "\n", source = None):
        content = df[content_columns].apply(
            lambda row: separator.join(
                f"{col}: {row[col]}" for col in content_columns
            ), axis=1
        )

        extra = {"row": range(len(df))}
        if source:
            extra["source"] = source
    
        metadata = df[metadata_columns].assign(**extra).to_dict(orient="records")

        return [
            Document(page_content=text, metadata=meta)
            for text, meta in zip(content, metadata)
        ]
    
    @staticmethod
    def _pdf_to_documents(pdf_path):
        loader = PyPDFLoader(
            file_path=pdf_path,
        )

        return loader.load()