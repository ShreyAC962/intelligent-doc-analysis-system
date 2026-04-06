"""
RAG (Retrieval-Augmented Generation) System
 
WHY THIS EXISTS:
- Allows users to ask questions about their documents
- Ensures AI answers are based on actual documents (not hallucinated)
- Combines search + generation for accurate responses
 
WHAT IT DOES:
1. Receives user question
2. Searches for relevant documents using Vector Search
3. Sends question + relevant documents to Gemini AI
4. Returns answer with citations

WITHOUT RAG: AI might make up an answer
WITH RAG: AI bases answer on your actual documents

TECHNOLOGIES USED:
- Vertex AI Vector Search: Fast similarity search
- Gemini AI: Generates human-like answers
- Embeddings: Converts questions to vectors

"""

from typing import List, Dict, Optional
from google.cloud import aiplatform
from vertexai.preview.generative_models import GenerativeModel, Part
from vertexai.language_models import TextEmbeddingModel
from config import settings
import logging

logging.basicConfig(level=settings.log_level)
logger = logging.getLogger(__name__)

class RAGSystem:
    # Combines search(retrieval) with AI text generation

    def __init__(self, vector_search_index_endpoint : str):
        # vector_serach_index_endpoint -> Where to search for documents

        # Intialize Vertex AI
        aiplatform.init(
            project=settings.project_id,
            location=settings.vertex_ai_location
        )

        # Initialize Gemini model 
        self.model = GenerativeModel(settings.gemini_model)

        # Initialize embedding model
        self.embedding_model = TextEmbeddingModel.from_pretrained(settings.embedding_model)

        # Vector Search endpoint
        self.index_endpoint = vector_search_index_endpoint

        logger.info("RAG System Initialized")
    

    def retrieve_relevant_documents(self, query : str, top_k: int = 5) -> List[Dict]:
        try:
            # Convert query to embedding
            query_embeding = self.embedding_model.get_embeddings([query])[0].values

            # Search vector index

            # In production this would call Vertex AI Vector Search API
            # ACTUAL IMPLEMENTATION (uncomment in production)

            # from google.cloud.aiplatform import MatchingEngineIndexEndpoint
            # endpoint = MatchingEngineIndexEndpoint(self.index_endpoint)
            # response = endpoint.find_neighbors(
            #     deployed_index_id="deployed_index",
            #     queries=[query_embeding],
            #     num_neighbors=top_k
            # )

            # Mock response, in production this comes from vector search
            relevant_docs = [
                {
                    'chunk_id' : 'doc1_chunk1',
                    'text' : 'Machine learning models require training data ...',
                    'score' : 0.92,
                    'metadata' : {'source' : 'ml_guide.pdf', 'page' : 5}
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
            raise

    
    def generate_answer(self, query : str, context_documents : List[Dict]) -> Dict:
        # Generate answer using Gemini AI
        try:
            # Build context from retrieved documents
            context = "\n\n".join([
                f"Document {i+1} (Source: {doc['metadata']['source']}):\n{doc['text']}"
                for i, doc in enumerate(context_documents)
            ])
            # Create a prompt - This is the instruction we give to Gemini AI
            prompt = f"""You are a helpful AI assistant. Answer the question based ONLY on the provided documents.
            If the documents don't contain enough information to answer, say "I don't have enough information to answer that question."
            
            Documents:
            {context}
            
            Question: {query}
            
            Instructions:
            1. Answer based only on the provided documents
            2. Cite sources by mentioning the document name
            3. Be concise but thorough
            4. If uncertain, acknowledge it
            
            Answer:"""

            # Call Gemini API
            response = self.model.generate_content(prompt)

            # Extract answer
            answer_text = response.text

            # Prepare result with metadata
            result = {
                'query' : query,
                'answer' : answer_text,
                'sources' : [doc['metadata']['source'] for doc in context_documents],
                'confidence' : 'high' if len(context_documents) >= 3 else 'medium',
                'num_sources_used' : len(context_documents)
            }

            logger.info(f"Generated answer for query: {query[:50]}..")
            return result
        
        except Exception as e:
            logger.error(f"Error generating answer: {e}")
            raise


    def query(self, question: str, top_k : int = 5) -> Dict:
        # Complete RAG pipeline : Retrieve(Vector Search) + Generate (Gemini AI)
        try:
            # Retrieve relevant documets
            relevant_docs = self.retrieve_relevant_documents(question, top_k)
            if not relevant_docs:
                return {
                    'query' : question,
                    'answer' : "I couldm't find any relevant documents to answer your question.",
                    'sources' : [],
                    'confidence' : 'none',
                    'num_sources_used' : 0
                }
            
            # Generate answer
            result = self.generate_answer(question, relevant_docs)
            return result
    
        except Exception as e:
            logger.error(f"Error in RAG query : {e}")
            raise

    
    def summarize_documents(self, document_id : str) -> Dict:
        # Summarize a specific document

        # Retrieve all chunks of the document
        # Send to Gemini with summarization prompt 
        # Return concise summary
        try:
            # In production, retrieve document chunks from Vector Search
            # Using Mock data
            document_chunks = [
                "Chapter 1: Introduction to machine learning...",
                "Chapter 2: Data preprocessing techniques...",
                "Chapter 3: Model training and evaluation..."
            ]
            # Combine chunks
            full_text = "\n\n".join(document_chunks)

            # Create summarization prompt
            prompt = f"""Summarize the following document in 2-3 paragraphs.
            Focus on key points, main findings, and actionable insights.
            
            Document:
            {full_text}
            
            Summary:"""

            # Generate Summary
            response = self.model.generate_content(prompt)

            result = {
                'document_id' : document_id,
                'summary' : response.text,
                'original_length' : len(full_text),
                'summary_length' : len(response.text),
                'compression_ratio' : len(response.text)/len(full_text)
            }

            logger.info(f"Summarized document: {document_id}")
            return result
        
        except Exception as e:
            logger.info(f"Error summarizing document: {e}")
            raise




# Example Usage

if __name__ == "__main__":
    # Initialize RAG system
    # In production, you'd pass the actual Vector Search endpoint
    rag = RAGSystem(vector_search_index_endpoint="projects/xxx/locations/xxx/indexEndpoints/xxx")

    # Ask a question
    result = rag.query("What are the key features of machine learning?")
    print("Question:", result['query'])
    print("Answer:", result['answer'])
    print("Sources:", result['sources'])

    # Summarize a document
    summary = rag.summarize_documents("doc123")
    print("\nSummary:", summary['summary'])
    print(f"Compression: {summary['compression_ratio']:.2%}")
 

    
