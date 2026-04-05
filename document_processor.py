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


    def extract_text_from_pdf(self, file_content : bytes) -> str:
        # Extract text from PDF files
        try:
            pdf_file = io.BytesIO(file_content)
            pdf_reader = PyPDF2.PdfFileReader(pdf_file)
            text = ""
            for page in pdf_reader.pages:
                text += page.extract_text() + "\n"

            logger.info(f"Extracted {len(text)} characters from PDF")
            return text
        except Exception as e:
            logger.error(f"Error extracting PDF text : {e}")
            raise

    
    def extract_text_from_docx(self, file_content : bytes) -> str:
        # Word files have complex formatting : we need just the text
        try:
            docx_file = io.BytesIO(file_content)
            doc = Document(docx_file) 

            text = ""
            for paragraph in doc.paragraphs:
                text += paragraph.text + "\n"

            logger.info(f"Extracted {len(text)} characters from DOCX")
            return text
        
        except Exception as e:
            logger.error(f"Error extracting DOCX text : {e}")
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
    
    def create_embeddings(self, texts: List[str]) -> List[List[float]]:
        # Convert text into vector embeddings
        # Similar texts have similar numeric vectors

        # Using Google's embedding model(Trained on billions of documents)
        # Each text becomes a list of 768 numbers
        try:
            # Process in batch - API has limits
            embeddings = []
            batch_size = settings.embedding_batch_size
            for i in range(0, len(texts), batch_size):
                batch = texts[i : i + batch_size]

                # Call Vertex AI Embedding API
                embeddings_response = self.embedding_model.get_embeddings(batch)

                # Extract the vectors
                batch_embeddings = [emb.values for emb in embeddings_response]
                embeddings.append(batch_embeddings)
            
            logger.info(f"Created {len(embeddings)} embeddings")
            return embeddings
        
        except Exception as e:
            logger.error(f"Error creating embeddings : {e}")
            raise
    
    def process_document(self, file_path : str, file_content : bytes, file_type: str) -> Dict:

    # Complete document processing pipeline
    # Upload to Cloud Storage -> Extract text -> Chunk text -> Create embeddings -> Return results for indexing
        try:
            # Generate unique document ID
            doc_id = hashlib.md5(file_content).hexdigest()

            # Upload the document
            gcs_uri = self.upload_document(file_path, file_content, {
                'doc_id' : doc_id,
                'file_type' : file_type,
                'processed_at' : datetime.now().isoformat()
            })

            # Extract text
            text = self.extract_text(file_content, file_type)

            # Chunk text
            chunks = self.chunk_text(text)

            # Create embeddings
            emeddings = self.create_embeddings(chunks)

            # Prepare result
            result = {
                'doc_id' : doc_id,
                'gcs_uri' : gcs_uri,
                'file_type' : file_type,
                'num_chunks' : len(chunks),
                'embeddings' : emeddings,
                'processed_at' : datetime.now().isoformat()
            }

            logger.info(f"Processed document : {doc_id}")
            return result
        
        except Exception as e:
            logger.error(f"Error processing documents: {e}")
            raise

# Example usage for testing
if __name__ == "__main__":
    # Initialize processor
    processor = DocumentProcessor()

    sample_text = b"This is a sample document about machine learning and AI."

    result = processor.process_document(
        file_path = "samples/test.txt",
        file_content = sample_text,
        file_type = "txt" 
    )

    print(f"Document processed: {result['doc_id']}")
    print(f"Number of chunks: {result['num_chunks']}")
    print(f"Embedding dimensions: {len(result['embeddings'][0])}")
 
                

        


           

                      



