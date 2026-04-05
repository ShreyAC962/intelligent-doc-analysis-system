"""
Document Processing Module
 
WHY THIS EXISTS:
- Handles uploading, extracting, and processing documents
- Converts documents into searchable embeddings
- Manages the document lifecycle
 
WHAT IT DOES:
1. Upload documents to Cloud Storage
2. Extract text from PDFs, DOCX, TXT
3. Split text into chunks (for better search)
4. Create embeddings (vector representations)
5. Store embeddings in Vector Search
 
TECHNOLOGIES USED:
- Cloud Storage: Stores raw documents
- Vertex AI Embeddings: Converts text to numbers (vectors)
- Vector Search: Fast similarity search
"""

from typing import List, Dict, Optional
from google.cloud import storage
from google.cloud import aiplatform
from vertexai.language_models import TextEmbeddingModel
import PyPDF2
from docx import Document
import io
import hashlib
from datetime import datetime
from config import settings
import logging

logging.basicConfig(level=settings.log_level)
logger = logging.getLogger(__name__)

class DocumentProcessor:
    def __init__(self):
        # Intialize google cloud clients
        self.storage_client = storage.Client(project = settings.project_id)
        self.bucket = self.storage_client.bucket(settings.documents_bucket)

        # Intialize VertexAI
        aiplatform.init(
            project = settings.project_id,
            location = settings.vertex_ai_location
        )

        # Intialize embedding model
        self.embedding_model = TextEmbeddingModel.from_pretrained(settings.embedding_model)
        
        logger.info("DocumentProcessor initialized")
    

    def upload_document(self, file_path : str, file_content : bytes, metadata : Optional[Dict] = None) -> str:
        # Upload a document to Cloud Storage (centrailized storage for all documents)
        # Uploads to Google cloud storage bucket

        # file_path : Where file to be stored
        # metadata : Extra info about the file
        # file_content : The actual file bytes

        # Returns GCS URI : gs://bucket_name/docs/report.pdf

        try :
            blob = self.bucket.blob(file_path)
            if metadata:
                blob.metadata = metadata
            blob.upload_from_string(file_content)
            gcs_uri = f"gs://{settings.documents_bucket}/{file_path}"
            logger.info(f"Uploaded document : {gcs_uri}")
            return gcs_uri
        except Exception as e:
            logger.info(f"Eroor uploading document : {e}")
            raise

    def extract_text(self, file_content : bytes, file_type : str) -> str:
            # Extract text from any supported file type
            if file_type.lower() == 'pdf':
                return self.extract_text_from_pdf(file_content)
            elif file_type.lower() in ['docx', 'doc']:
                return self.extract_text_from_docx(file_content)
            elif file_type.lower() == 'txt':  
                return file_content.decode('utf-8')
            else:
                raise ValueError("Unsupported file type : {file_type}") 


    def chunk_text(self, text : str, chunk_size: int = 1000, overlap : int = 200) -> List[str]:
        # Split long text into smaller chunks
        # Long documents are hard to seearch - AI models have token limits
        # Smaller chunks = more precise search results
    
        # Split text every chunk_size characters - keep overlap characters between chunks(so context is not lost)
        chunks = []
        start = 0
        text_length = len(text)

        while start < text_length:
            end = start + chunk_size
            chunk = text[start : end]
            chunks.append(chunk)
            start = end - overlap
        
        logger.info(f"Split text into {len(chunks)} chunks")
        return chunks


           
                      



