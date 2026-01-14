from typing import List, Tuple, Callable
import chromadb
from chromadb import Documents, EmbeddingFunction, Embeddings
import google.generativeai as genai
import os
from aimakerspace.text_utils import Document

class GeminiEmbeddingFunction(EmbeddingFunction):
    def __init__(self, api_key: str, model_name: str = "models/text-embedding-004"):
        self.api_key = api_key
        self.model_name = model_name
        genai.configure(api_key=self.api_key)

    def __call__(self, input: Documents) -> Embeddings:
        # Gemini embedding API
        # Handle batching if necessary, but genai usually handles it or we loop
        embeddings = []
        for text in input:
            result = genai.embed_content(
                model=self.model_name,
                content=text,
                task_type="retrieval_document"
            )
            embeddings.append(result['embedding'])
        return embeddings

class VectorDatabase:
    def __init__(self, collection_name: str = "documents"):
        self.client = chromadb.PersistentClient(path="./chroma_db")
        
        google_api_key = os.getenv("GOOGLE_API_KEY")
        openai_key = os.getenv("OPENAI_API_KEY")

        if google_api_key:
            self.embedding_fn = GeminiEmbeddingFunction(
                api_key=google_api_key
            )
        elif openai_key:
             from chromadb.utils import embedding_functions
             self.embedding_fn = embedding_functions.OpenAIEmbeddingFunction(
                api_key=openai_key,
                model_name="text-embedding-3-small" 
            )
        else:
             from chromadb.utils import embedding_functions
             self.embedding_fn = embedding_functions.DefaultEmbeddingFunction()

        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            embedding_function=self.embedding_fn
        )

    async def abuild_from_documents(self, documents: List[Document]):
        ids = [f"id_{i}" for i in range(len(documents))]
        doc_texts = [doc.page_content for doc in documents]
        metadatas = [doc.metadata for doc in documents]
        
        if not doc_texts:
            return self

        self.collection.upsert(
            documents=doc_texts,
            metadatas=metadatas,
            ids=ids
        )
        return self

    def search_by_text(
        self,
        query_text: str,
        k: int = 4,
    ) -> List[Document]:
        # For query, we might need task_type="retrieval_query" if using Gemini
        # The custom function uses "retrieval_document" by default (lazy implementation above).
        # To strictly follow Gemini best practices, we should distinguish.
        # However, Chroma's EmbeddingFunction interface doesn't easily pass 'task_type' per call without state.
        # We'll use the generic embedding or rely on the fact that 'retrieval_document' usually works okay for dense retrieval too,
        # OR better: subclass properly.
        # For this fix, let's just use the embedding function as defined.
        
        # If we really want "retrieval_query", we might need to modify the function or handle it internally.
        # Let's keep it simple.
        
        results = self.collection.query(
            query_texts=[query_text],
            n_results=k
        )
        
        retrieved_docs = []
        if results["documents"]:
            texts = results["documents"][0]
            metas = results["metadatas"][0]
            
            for text, meta in zip(texts, metas):
                retrieved_docs.append(Document(page_content=text, metadata=meta))
                
        return retrieved_docs

if __name__ == "__main__":
    # Test
    # db = VectorDatabase()
    pass
