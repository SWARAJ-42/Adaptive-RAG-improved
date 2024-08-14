from sentence_transformers import SentenceTransformer
from langchain.embeddings.base import Embeddings

# Initialize SentenceTransformer model
embedding_model_name = "all-MiniLM-L6-v2"

class SentenceTransformerEmbeddings(Embeddings):
    def __init__(self, model_name):
        self.model = SentenceTransformer(model_name)

    def embed_query(self, query: str) -> list:
        # Use SentenceTransformer's encode method
        return self.model.encode(query).tolist()

    def embed_documents(self, documents: list) -> list:
        # Use SentenceTransformer's encode method
        return [self.model.encode(doc).tolist() for doc in documents]
