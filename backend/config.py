"""
Configuration for Clinical Notes RAG System

Key settings:
- EMBEDDING_MODEL: text-embedding-3-large for better medical/legal document understanding
- CHUNK_SIZE: 2048 for larger context windows (keeps sections together)
- CHUNK_OVERLAP: 400 for better context preservation
"""
import os
from dotenv import load_dotenv

load_dotenv()


class Settings:
    # OpenAI Models
    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY")
    LLM_MODEL: str = os.getenv("LLM_MODEL", "gpt-4o-mini")
    
    # ✅ UPGRADED: text-embedding-3-large for better medical/legal document understanding
    # 3072 dimensions, 64.6% MTEB score vs 62.3% for small
    # Cost: $0.13/1M tokens (vs $0.02 for small) - worth it for accuracy
    EMBEDDING_MODEL: str = os.getenv("EMBEDDING_MODEL", "text-embedding-3-large")
    
    # Dynamically set dimensions based on model
    @property
    def EMBEDDING_DIMENSIONS(self) -> int:
        """Get correct embedding dimensions for the selected model"""
        model = self.EMBEDDING_MODEL
        if "large" in model:
            return 3072
        elif "small" in model:
            return 1536
        elif "ada" in model:
            return 1536
        else:
            # Default to large model dimensions
            return 3072
    
    # Qdrant Cloud
    QDRANT_URL: str = os.getenv("QDRANT_URL")
    QDRANT_API_KEY: str = os.getenv("QDRANT_API_KEY")
    
    # MongoDB (local or Atlas)
    MONGODB_URI: str = os.getenv("MONGODB_URI", "mongodb://localhost:27017")
    
    # ✅ IMPROVED: Chunking settings for legal/medical documents
    # Larger chunks keep sections together (e.g., "8. Relocation Allowance" stays intact)
    CHUNK_SIZE: int = 2048  # Was 512, increased for better context
    CHUNK_OVERLAP: int = 400  # Was 50, increased to preserve context across boundaries
    
    # Collection name
    COLLECTION_NAME: str = "clinical_notes"


settings = Settings()

# Print configuration for debugging
print(f"📋 Config: Embedding={settings.EMBEDDING_MODEL}, Dim={settings.EMBEDDING_DIMENSIONS}, Chunk={settings.CHUNK_SIZE}")

# Validate API keys
if not settings.OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY not set in .env file")

if not settings.QDRANT_URL or not settings.QDRANT_API_KEY:
    raise ValueError("QDRANT_URL and QDRANT_API_KEY must be set in .env file")
