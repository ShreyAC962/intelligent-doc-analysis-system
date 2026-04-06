"""
RAG (Retrieval-Augmented Generation) System
"""

from typing import List, Dict
import logging
import os

# Vertex AI imports
from google.cloud import aiplatform
from vertexai.preview.generative_models import GenerativeModel
from vertexai.language_models import TextEmbeddingModel

# Load settings from your config.py
from config import settings

logging.basicConfig(level=settings.log_level)
logger = logging.getLogger(__name__)


class RAGSystem:
    """Retrieval-Augmented Generation System"""

    def __init__(self, vector_search_index_endpoint: str):
        """
        Initialize RAG system
        """
        try:
            # Initialize Vertex AI
            aiplatform.init(
                project=settings.project_id,
                location=settings.vertex_ai_location
            )

            # Initialize Gemini model
            try:
                self.model = GenerativeModel(model_name=settings.gemini_model)
                logger.info("Gemini model initialized")
            except Exception as e:
                logger.warning(f"Gemini model init failed: {e}. Using fallback 'gemini-1.5-flash'")
                self.model = GenerativeModel(model_name="gemini-1.5-flash")

            # Initialize embedding model
            self.embedding_model = TextEmbeddingModel.from_pretrained(
                settings.embedding_model
            )

            # Vector Search endpoint
            self.index_endpoint = vector_search_index_endpoint

            logger.info("RAG System initialized")

        except Exception as e:
            logger.error(f"RAG System initialization failed: {e}")
            raise

    def retrieve_relevant_documents(self, query: str, top_k: int = 5) -> List[Dict]:
        """Retrieve top-k relevant documents"""
        try:
            # Convert query to embedding
            query_embedding = self.embedding_model.get_embeddings([query])[0].values

            # MOCK vector search response (replace with Vertex AI MatchingEngine in prod)
            relevant_docs = [
                {
                    'chunk_id': 'doc1_chunk1',
                    'text': 'Machine learning models require training data...',
                    'score': 0.92,
                    'metadata': {'source': 'ml_guide.pdf', 'page': 5}
                },
                {
                    'chunk_id': 'doc2_chunk3',
                    'text': 'To train a model, first prepare your dataset...',
                    'score': 0.88,
                    'metadata': {'source': 'tutorial.pdf', 'page': 12}
                }
            ]

            logger.info(f"Retrieved {len(relevant_docs)} relevant documents")
            return relevant_docs

        except Exception as e:
            logger.error(f"Error retrieving documents: {e}")
            return []

    def generate_answer(self, query: str, context_documents: List[Dict]) -> Dict:
        """Generate answer using Gemini AI"""
        if not context_documents:
            return {
                'query': query,
                'answer': "I couldn't find relevant documents to answer your question.",
                'sources': [],
                'confidence': 'none',
                'num_sources_used': 0
            }

        try:
            # Build context from documents
            context = "\n\n".join([
                f"Document {i+1} (Source: {doc['metadata']['source']}):\n{doc['text']}"
                for i, doc in enumerate(context_documents)
            ])

            prompt = f"""You are a helpful AI assistant. Answer the question based ONLY on the provided documents.
            If the documents don't contain enough information, say "I don't have enough information to answer that question."

            Documents:
            {context}

            Question: {query}

            Instructions:
            1. Answer based only on the provided documents
            2. Cite sources by mentioning the document name
            3. Be concise but thorough
            4. If uncertain, acknowledge it

            Answer:"""

            response = self.model.generate_content(prompt)
            answer_text = response.text

            return {
                'query': query,
                'answer': answer_text,
                'sources': [doc['metadata']['source'] for doc in context_documents],
                'confidence': 'high' if len(context_documents) >= 3 else 'medium',
                'num_sources_used': len(context_documents)
            }

        except Exception as e:
            logger.error(f"Error generating answer: {e}")
            return {
                'query': query,
                'answer': "Error generating answer.",
                'sources': [],
                'confidence': 'none',
                'num_sources_used': 0
            }

    def query(self, question: str, top_k: int = 5) -> Dict:
        """Full RAG pipeline"""
        docs = self.retrieve_relevant_documents(question, top_k)
        return self.generate_answer(question, docs)

    def summarize_document(self, document_id: str) -> Dict:
        """Summarize a specific document"""
        try:
            # MOCK: retrieve document chunks
            document_chunks = [
                "Chapter 1: Introduction to machine learning...",
                "Chapter 2: Data preprocessing techniques...",
                "Chapter 3: Model training and evaluation..."
            ]
            full_text = "\n\n".join(document_chunks)

            prompt = f"""Summarize the following document in 2-3 paragraphs.
                Focus on key points, main findings, and actionable insights.

                Document:
                {full_text}

                Summary:"""

            response = self.model.generate_content(prompt)

            return {
                'document_id': document_id,
                'summary': response.text,
                'original_length': len(full_text),
                'summary_length': len(response.text),
                'compression_ratio': len(response.text) / len(full_text)
            }

        except Exception as e:
            logger.error(f"Error summarizing document: {e}")
            return {
                'document_id': document_id,
                'summary': "Error summarizing document",
                'original_length': 0,
                'summary_length': 0,
                'compression_ratio': 0
            }


# --------------------------
# Example Usage
# --------------------------
if __name__ == "__main__":
    rag = RAGSystem(vector_search_index_endpoint="projects/xxx/locations/xxx/indexEndpoints/xxx")

    # Query example
    result = rag.query("What are the key features of machine learning?")
    print("Question:", result['query'])
    print("Answer:", result['answer'])
    print("Sources:", result['sources'])

    # Summarize document example
    summary = rag.summarize_document("doc123")
    print("\nSummary:", summary['summary'])
    print(f"Compression: {summary['compression_ratio']:.2%}")