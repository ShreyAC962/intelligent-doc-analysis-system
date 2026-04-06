"""
Configuration Module

WHY THIS FILE EXISTS:
- Keeps all settings in one place
- Loads secrets from environment variables (more secure than hardcoding)
- Validates configuration on startup

WHAT IT DOES:
- Reads .env file
- Sets up Google Cloud credentials
- Provides configuration to all other modules
"""

from pydantic_settings import BaseSettings
from pydantic import Field
from typing import Optional
import os


class Settings(BaseSettings):
    """
    Application Settings
    
    Think of this like a control panel for your entire system.
    All the knobs and dials are defined here.
    """
    
    # Google Cloud Project Settings
    project_id: str = Field(..., env="PROJECT_ID")
    region: str = Field(default="us-central1", env="REGION")
    
    # Storage Configuration
    bucket_name: str = Field(..., env="BUCKET_NAME")
    documents_bucket: str = Field(..., env="DOCUMENTS_BUCKET")
    models_bucket: str = Field(..., env="MODELS_BUCKET")
    data_bucket: str = Field(..., env="DATA_BUCKET")
    
    # BigQuery Configuration
    dataset_id: str = Field(..., env="DATASET_ID")
    table_id: str = Field(default="events", env="TABLE_ID")
    anomaly_table_id: str = Field(default="detected_anomalies", env="ANOMALY_TABLE_ID")
    
    # Vertex AI Configuration
    vertex_ai_location: str = Field(default="us-central1", env="VERTEX_AI_LOCATION")
    gemini_model: str = Field(default="gemini-1.5-pro-001", env="GEMINI_MODEL")
    embedding_model: str = Field(default="textembedding-gecko@003", env="EMBEDDING_MODEL")
    
    # Vector Search Configuration
    index_endpoint_name: str = Field(..., env="INDEX_ENDPOINT_NAME")
    index_name: str = Field(..., env="INDEX_NAME")
    index_dimensions: int = Field(default=768, env="INDEX_DIMENSIONS")
    vector_search_neighbors: int = Field(default=10, env="VECTOR_SEARCH_NEIGHBORS")
    
    # Service Account
    service_account_key_path: Optional[str] = Field(None, env="SERVICE_ACCOUNT_KEY_PATH")
    
    # API Configuration
    api_host: str = Field(default="0.0.0.0", env="API_HOST")
    api_port: int = Field(default=8080, env="API_PORT")
    max_workers: int = Field(default=4, env="MAX_WORKERS")
    
    # Model Training Configuration
    batch_size: int = Field(default=32, env="BATCH_SIZE")
    learning_rate: float = Field(default=0.001, env="LEARNING_RATE")
    epochs: int = Field(default=10, env="EPOCHS")
    anomaly_threshold: float = Field(default=0.95, env="ANOMALY_THRESHOLD")
    
    # Monitoring Configuration
    enable_monitoring: bool = Field(default=True, env="ENABLE_MONITORING")
    monitoring_interval: int = Field(default=300, env="MONITORING_INTERVAL")
    log_level: str = Field(default="INFO", env="LOG_LEVEL")
    
    # Feature Flags
    enable_document_summarization: bool = Field(default=True, env="ENABLE_DOCUMENT_SUMMARIZATION")
    enable_anomaly_detection: bool = Field(default=True, env="ENABLE_ANOMALY_DETECTION")
    enable_real_time_processing: bool = Field(default=True, env="ENABLE_REAL_TIME_PROCESSING")
    
    # Performance Tuning
    max_document_length: int = Field(default=10000, env="MAX_DOCUMENT_LENGTH")
    embedding_batch_size: int = Field(default=100, env="EMBEDDING_BATCH_SIZE")
    
    # Deployment Configuration
    deployment_environment: str = Field(default="production", env="DEPLOYMENT_ENVIRONMENT")
    min_instances: int = Field(default=1, env="MIN_INSTANCES")
    max_instances: int = Field(default=10, env="MAX_INSTANCES")
    memory_limit: str = Field(default="4Gi", env="MEMORY_LIMIT")
    cpu_limit: str = Field(default="2", env="CPU_LIMIT")
    
    # Cloud Run specific settings (NEW - these were missing!)
    timeout: int = Field(default=300, env="TIMEOUT")
    concurrency: int = Field(default=80, env="CONCURRENCY")
    
    # Rate Limiting (NEW - these were missing!)
    max_requests_per_minute: int = Field(default=60, env="MAX_REQUESTS_PER_MINUTE")
    max_gemini_calls_per_minute: int = Field(default=10, env="MAX_GEMINI_CALLS_PER_MINUTE")
    max_embedding_calls_per_minute: int = Field(default=20, env="MAX_EMBEDDING_CALLS_PER_MINUTE")
    
    # Cost Optimization (NEW - these were missing!)
    use_gemini_flash: bool = Field(default=True, env="USE_GEMINI_FLASH")
    enable_embedding_cache: bool = Field(default=True, env="ENABLE_EMBEDDING_CACHE")
    storage_class: str = Field(default="STANDARD", env="STORAGE_CLASS")
    auto_delete_old_data: bool = Field(default=True, env="AUTO_DELETE_OLD_DATA")
    data_retention_days: int = Field(default=30, env="DATA_RETENTION_DAYS")
    
    # Development flags (NEW - these were missing!)
    debug: bool = Field(default=False, env="DEBUG")
    use_mock_data: bool = Field(default=False, env="USE_MOCK_DATA")
    skip_expensive_ops: bool = Field(default=False, env="SKIP_EXPENSIVE_OPS")
    
    class Config:
        # This tells Pydantic to look for a .env file
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False


def get_settings() -> Settings:
    """
    Get application settings
    
    WHY: Creates a single instance of settings that's reused everywhere
    HOW: Reads from .env file, validates all required fields exist
    """
    return Settings()


def setup_google_credentials(settings: Settings) -> None:
    """
    Setup Google Cloud authentication
    
    WHY: Google Cloud needs to know who you are before allowing API calls
    HOW: Points to the service account key file
    
    SIMPLE EXPLANATION:
    Like showing your ID card before entering a building. The service
    account key is your digital ID card for Google Cloud.
    """
    if settings.service_account_key_path and os.path.exists(settings.service_account_key_path):
        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = settings.service_account_key_path
        print(f"Google Cloud credentials configured: {settings.service_account_key_path}")
    else:
        print("No service account key found. Using default credentials.")


# Create global settings instance
settings = get_settings()
setup_google_credentials(settings)