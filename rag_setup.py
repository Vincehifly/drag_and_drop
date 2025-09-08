#!/usr/bin/env python3
"""
RAG Setup - PDF Processing and Vector Database
Handles PDF chunking and vector database creation for RAG functionality.
"""

import os
import chromadb
from chromadb.config import Settings
from pypdf import PdfReader
from langchain_openai import AzureOpenAIEmbeddings
from typing import List, Dict
import hashlib

# Load environment variables from .env file
try:
    from dotenv import load_dotenv
    load_dotenv()
    print("✅ Loaded environment variables from .env file")
except ImportError:
    print("⚠️ python-dotenv not installed. Install with: pip install python-dotenv")
except Exception as e:
    print(f"⚠️ Could not load .env file: {e}")

class RAGSetup:
    def __init__(self, pdf_path: str = "cheat_sheet.pdf", db_path: str = "./chroma_db"):
        self.pdf_path = pdf_path
        self.db_path = db_path
        
        # Use Azure OpenAI embeddings
        self.embedding_model = AzureOpenAIEmbeddings(
            azure_deployment=os.getenv("EMBEDDING_MODEL_NAME", "AA-TEXTEMBEDDING3LARGE"),
            azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
            api_key=os.getenv("AZUREOPENAIAPIKEY"),
            api_version=os.getenv("AZUREOPENAIAPIVERSION", "2024-02-15-preview")
        )
        
        self.chunk_size = 1000
        self.chunk_overlap = 200
        
        # Initialize ChromaDB
        self.client = chromadb.PersistentClient(
            path=self.db_path,
            settings=Settings(anonymized_telemetry=False)
        )
        
        # Get or create collection
        self.collection = self.client.get_or_create_collection(
            name="cheat_sheet",
            metadata={"hnsw:space": "cosine"}
        )
    
    def extract_text_from_pdf(self) -> str:
        """Extract text from the PDF file."""
        try:
            reader = PdfReader(self.pdf_path)
            text = ""
            for page in reader.pages:
                text += page.extract_text() + "\n"
            return text
        except Exception as e:
            print(f"Error reading PDF: {e}")
            return ""
    
    def chunk_text(self, text: str) -> List[Dict[str, str]]:
        """Split text into overlapping chunks."""
        chunks = []
        start = 0
        
        while start < len(text):
            end = start + self.chunk_size
            chunk_text = text[start:end]
            
            # Find the last complete sentence or word boundary
            if end < len(text):
                last_period = chunk_text.rfind('.')
                last_space = chunk_text.rfind(' ')
                if last_period > last_space and last_period > 0:
                    end = start + last_period + 1
                    chunk_text = text[start:end]
                elif last_space > 0:
                    end = start + last_space
                    chunk_text = text[start:end]
            
            if chunk_text.strip():
                chunk_id = hashlib.md5(chunk_text.encode()).hexdigest()
                chunks.append({
                    'id': chunk_id,
                    'text': chunk_text.strip(),
                    'metadata': {
                        'start_pos': start,
                        'end_pos': end,
                        'chunk_size': len(chunk_text)
                    }
                })
            
            start = end - self.chunk_overlap
            if start >= len(text):
                break
        
        return chunks
    
    def embed_chunks(self, chunks: List[Dict[str, str]]) -> List[List[float]]:
        """Generate embeddings for text chunks using Azure OpenAI."""
        texts = [chunk['text'] for chunk in chunks]
        # Azure OpenAI embeddings returns a list of floats directly
        embeddings = self.embedding_model.embed_documents(texts)
        return embeddings
    
    def setup_vector_db(self) -> bool:
        """Process PDF and create vector database."""
        print(f"📄 Processing PDF: {self.pdf_path}")
        
        # Check if PDF exists
        if not os.path.exists(self.pdf_path):
            print(f"❌ PDF file not found: {self.pdf_path}")
            return False
        
        # Extract text from PDF
        text = self.extract_text_from_pdf()
        if not text.strip():
            print("❌ No text extracted from PDF")
            return False
        
        print(f"✅ Extracted {len(text)} characters from PDF")
        
        # Chunk the text
        chunks = self.chunk_text(text)
        print(f"✅ Created {len(chunks)} text chunks")
        
        # Generate embeddings
        print("🔄 Generating embeddings...")
        embeddings = self.embed_chunks(chunks)
        
        # Store in ChromaDB
        print("🔄 Storing in vector database...")
        try:
            # Clear existing data - delete all documents
            try:
                self.collection.delete(where={"$exists": "chunk_size"})
            except:
                # If delete fails, try to get all IDs and delete them
                try:
                    all_docs = self.collection.get()
                    if all_docs['ids']:
                        self.collection.delete(ids=all_docs['ids'])
                except:
                    pass  # Continue if delete fails
            
            # Add new chunks
            self.collection.add(
                embeddings=embeddings,
                documents=[chunk['text'] for chunk in chunks],
                metadatas=[chunk['metadata'] for chunk in chunks],
                ids=[chunk['id'] for chunk in chunks]
            )
            
            print(f"✅ Successfully stored {len(chunks)} chunks in vector database")
            return True
            
        except Exception as e:
            print(f"❌ Error storing in vector database: {e}")
            return False
    
    def search(self, query: str, n_results: int = 5) -> List[Dict[str, any]]:
        """Search the vector database for relevant chunks."""
        try:
            # Generate query embedding using Azure OpenAI
            query_embedding = self.embedding_model.embed_query(query)
            
            # Search in ChromaDB
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=n_results
            )
            
            # Format results
            formatted_results = []
            for i in range(len(results['documents'][0])):
                formatted_results.append({
                    'text': results['documents'][0][i],
                    'metadata': results['metadatas'][0][i],
                    'distance': results['distances'][0][i]
                })
            
            return formatted_results
            
        except Exception as e:
            print(f"Error searching vector database: {e}")
            return []


def setup_rag() -> RAGSetup:
    """Initialize and setup RAG system."""
    rag = RAGSetup()
    
    # Check if database already exists and has data
    try:
        count = rag.collection.count()
        if count > 0:
            print(f"✅ Vector database already exists with {count} chunks")
            return rag
    except:
        pass
    
    # Setup vector database
    if rag.setup_vector_db():
        print("✅ RAG system ready!")
        return rag
    else:
        print("❌ Failed to setup RAG system")
        return None


if __name__ == "__main__":
    # Test the RAG setup
    rag = setup_rag()
    if rag:
        # Test search
        test_query = "What is machine learning?"
        results = rag.search(test_query, n_results=3)
        print(f"\n🔍 Test search for: '{test_query}'")
        for i, result in enumerate(results, 1):
            print(f"\nResult {i} (distance: {result['distance']:.3f}):")
            print(f"Text: {result['text'][:200]}...")
