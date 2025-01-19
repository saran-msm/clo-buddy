import chromadb
from chromadb.config import Settings
from chromadb.utils import embedding_functions
import os
from typing import List, Dict, Optional

class VectorStore:
    def __init__(self):
        # Initialize ChromaDB with new configuration
        self.client = chromadb.PersistentClient(
            path="data/vectordb"
        )
        
        # Create embedding function using sentence-transformers
        self.embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
        
        # Create or get collection
        self.collection = self.client.get_or_create_collection(
            name="legal_documents",
            embedding_function=self.embedding_function
        )

    def add_document(self, 
                    document_id: str, 
                    text_chunks: List[str], 
                    metadata: Optional[Dict] = None) -> None:
        """
        Add document chunks to the vector store
        """
        try:
            # Generate IDs for chunks
            chunk_ids = [f"{document_id}_chunk_{i}" for i in range(len(text_chunks))]
            
            # Add metadata to each chunk
            metadatas = []
            for i in range(len(text_chunks)):
                chunk_metadata = {
                    "document_id": document_id,
                    "chunk_index": i,
                }
                if metadata:
                    chunk_metadata.update(metadata)
                metadatas.append(chunk_metadata)
            
            # Add to collection
            self.collection.add(
                ids=chunk_ids,
                documents=text_chunks,
                metadatas=metadatas
            )
            
        except Exception as e:
            print(f"Error adding document to vector store: {e}")
            raise

    def search(self, 
               query: str, 
               n_results: int = 5, 
               metadata_filter: Optional[Dict] = None) -> List[Dict]:
        """
        Search for relevant document chunks
        """
        try:
            results = self.collection.query(
                query_texts=[query],
                n_results=n_results,
                where=metadata_filter
            )
            
            # Format results
            formatted_results = []
            for i in range(len(results['documents'][0])):
                formatted_results.append({
                    'text': results['documents'][0][i],
                    'metadata': results['metadatas'][0][i],
                    'distance': results['distances'][0][i] if 'distances' in results else None
                })
            
            return formatted_results
            
        except Exception as e:
            print(f"Error searching vector store: {e}")
            raise

    def delete_document(self, document_id: str) -> None:
        """
        Delete all chunks of a document
        """
        try:
            self.collection.delete(
                where={"document_id": document_id}
            )
        except Exception as e:
            print(f"Error deleting document from vector store: {e}")
            raise 