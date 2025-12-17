"""Qdrant Cloud Vector Store with Document Tracking"""
import uuid
from qdrant_client import QdrantClient
from qdrant_client.http import models
from llama_index.core import VectorStoreIndex, StorageContext, Settings as LlamaSettings
from llama_index.vector_stores.qdrant import QdrantVectorStore
from llama_index.embeddings.openai import OpenAIEmbedding
from config import settings
from typing import List, Dict, Optional
from llama_index.core import Document


class VectorStoreManager:
    """
    Manages Qdrant Cloud vector store with document tracking.
    
    Key features:
    - Tracks which Qdrant points belong to which document
    - Enables deletion of specific documents without clearing everything
    - Links to MongoDB document IDs
    """
    
    def __init__(self):
        # Qdrant Cloud client
        self.qdrant_client = QdrantClient(
            url=settings.QDRANT_URL,
            api_key=settings.QDRANT_API_KEY,
        )
        
        # OpenAI embeddings
        self.embed_model = OpenAIEmbedding(
            model=settings.EMBEDDING_MODEL,
            api_key=settings.OPENAI_API_KEY
        )
        LlamaSettings.embed_model = self.embed_model
        
        # Create collection if it doesn't exist
        self._ensure_collection()
        
        # Vector store
        self.vector_store = QdrantVectorStore(
            client=self.qdrant_client,
            collection_name=settings.COLLECTION_NAME,
        )
        
        # In-memory document cache (for BM25 and quick access)
        self._documents: Dict[str, List[Document]] = {}  # document_id -> documents
    
    def _ensure_collection(self):
        """Create collection if it doesn't exist"""
        collections = self.qdrant_client.get_collections().collections
        collection_names = [c.name for c in collections]
        
        if settings.COLLECTION_NAME not in collection_names:
            self.qdrant_client.create_collection(
                collection_name=settings.COLLECTION_NAME,
                vectors_config=models.VectorParams(
                    size=settings.EMBEDDING_DIMENSIONS,  # 3072 for text-embedding-3-large
                    distance=models.Distance.COSINE
                )
            )
            print(f"Created collection: {settings.COLLECTION_NAME} (dim: {settings.EMBEDDING_DIMENSIONS})")
    
    def index_documents(
        self, 
        documents: List[Document], 
        document_id: str = None
    ) -> tuple[VectorStoreIndex, List[str]]:
        """
        Index documents into Qdrant Cloud
        
        Args:
            documents: List of LlamaIndex documents
            document_id: MongoDB document ID to track these vectors
        
        Returns:
            Tuple of (VectorStoreIndex, list of Qdrant point IDs)
        """
        # Generate unique point IDs for tracking
        point_ids = [str(uuid.uuid4()) for _ in documents]
        
        # Add document_id to metadata for filtering
        for doc, point_id in zip(documents, point_ids):
            doc.metadata["document_id"] = document_id or "unknown"
            doc.metadata["point_id"] = point_id
        
        # Cache documents
        if document_id:
            self._documents[document_id] = documents
        
        # Index into Qdrant
        storage_context = StorageContext.from_defaults(
            vector_store=self.vector_store
        )
        
        # ✅ Convert Documents to TextNodes directly (bypass chunking)
        from llama_index.core.schema import TextNode
        from tqdm import tqdm
        
        nodes = []
        print(f"Creating nodes from {len(documents)} chunks...")
        for doc in tqdm(documents, desc="Parsing nodes"):
            node = TextNode(
                text=doc.text,
                metadata=doc.metadata,
                id_=doc.doc_id or str(uuid.uuid4())
            )
            nodes.append(node)
        
        # Create index from nodes (no additional chunking)
        index = VectorStoreIndex(
            nodes=nodes,
            storage_context=storage_context,
            embed_model=self.embed_model,
            show_progress=True
        )
        
        print(f"Indexed {len(documents)} chunks to Qdrant Cloud (doc_id: {document_id})")
        return index, point_ids
    
    def get_index(self) -> VectorStoreIndex:
        """Get existing index for queries"""
        return VectorStoreIndex.from_vector_store(
            vector_store=self.vector_store,
            embed_model=self.embed_model
        )
    
    def get_index_for_document(self, document_id: str) -> Optional[VectorStoreIndex]:
        """
        Get index filtered to a specific document.
        This is useful when you want to search only within one document.
        """
        # For now, we return the full index
        # Filtering happens at query time via metadata
        return self.get_index()
    
    def get_all_documents(self) -> List[Document]:
        """Get all cached documents (for BM25)"""
        all_docs = []
        for docs in self._documents.values():
            all_docs.extend(docs)
        return all_docs
    
    def get_documents_for_document_id(self, document_id: str) -> List[Document]:
        """
        Get documents for a specific MongoDB document ID.
        
        Note: Since document IDs in Qdrant might not match MongoDB IDs,
        we also try to match by source filename.
        """
        # Try direct match first
        if document_id in self._documents:
            return self._documents[document_id]
        
        # Fallback: search by source filename
        # This handles cases where MongoDB ID != Qdrant document_id
        all_docs = self.get_all_documents()
        matching_docs = []
        
        for doc in all_docs:
            # Check if this document's metadata contains the document_id
            if doc.metadata.get("document_id") == document_id:
                matching_docs.append(doc)
        
        return matching_docs
    
    def get_documents_by_source(self, source_filename: str) -> List[Document]:
        """Get all documents from a specific source file"""
        all_docs = self.get_all_documents()
        return [doc for doc in all_docs if doc.metadata.get("source") == source_filename]
    
    def load_documents_from_qdrant(self, document_ids: List[str] = None, force_clear: bool = False):
        """
        Load documents from Qdrant into memory cache.
        Called on server startup to restore state.
        
        Args:
            document_ids: Optional list of specific document IDs to load
            force_clear: If True, clear existing cache before loading
        """
        try:
            print(f"🔄 Loading documents from Qdrant collection: {settings.COLLECTION_NAME}")
            
            # Clear cache if requested to avoid duplicates
            if force_clear:
                print(f"   Clearing existing cache...")
                self._documents = {}
            
            # Scroll through all points in the collection
            offset = None
            all_points = []
            
            while True:
                response = self.qdrant_client.scroll(
                    collection_name=settings.COLLECTION_NAME,
                    limit=100,
                    offset=offset,
                    with_payload=True,
                    with_vectors=False
                )
                
                points, offset = response
                if not points:
                    break
                    
                all_points.extend(points)
                print(f"   Loaded batch of {len(points)} points...")
                
                if offset is None:
                    break
            
            print(f"📦 Total points retrieved: {len(all_points)}")
            
            # Reconstruct documents from Qdrant payloads
            loaded_count = 0
            sample_shown = False
            
            for point in all_points:
                payload = point.payload or {}
                doc_id = payload.get("document_id", "unknown")
                
                # Skip if we're filtering and this doesn't match
                if document_ids and doc_id not in document_ids:
                    continue
                
                # Get text from payload
                text = ""
                
                # Try direct text field first
                if "text" in payload and payload["text"]:
                    text = payload["text"]
                # Try _node_content (llama-index serialized format)
                elif "_node_content" in payload:
                    node_content = payload["_node_content"]
                    if isinstance(node_content, str):
                        try:
                            import json
                            node_data = json.loads(node_content)
                            text = node_data.get("text", "")
                        except:
                            text = node_content
                    elif isinstance(node_content, dict):
                        text = node_content.get("text", "")
                # Fallback to content field
                if not text and "content" in payload:
                    text = str(payload.get("content", ""))
                
                # Create Document from payload
                doc = Document(
                    text=text,
                    metadata={
                        "source": payload.get("source", "Unknown"),
                        "page_number": payload.get("page_number", 0),
                        "document_id": doc_id,
                        "point_id": str(point.id)
                    }
                )
                
                # Show first chunk as sample
                if not sample_shown and text:
                    print(f"\n   📄 Sample chunk (first loaded):")
                    print(f"      Source: {payload.get('source', 'Unknown')}")
                    print(f"      Page: {payload.get('page_number', 0)}")
                    print(f"      Text length: {len(text)} chars")
                    print(f"      Text preview: {text[:200]}...")
                    sample_shown = True
                
                if doc_id not in self._documents:
                    self._documents[doc_id] = []
                self._documents[doc_id].append(doc)
                loaded_count += 1
            
            total = sum(len(docs) for docs in self._documents.values())
            print(f"✅ Loaded {loaded_count} new chunks (total: {total}) from Qdrant for {len(self._documents)} documents")
            print(f"   Document IDs in cache: {list(self._documents.keys())[:5]}...")
            
        except Exception as e:
            print(f"❌ Error loading documents from Qdrant: {e}")
            import traceback
            traceback.print_exc()
    
    def delete_document_vectors(self, document_id: str, point_ids: List[str] = None) -> bool:
        """Delete all vectors for a specific document using point IDs"""
        try:
            if point_ids:
                # Delete by point IDs (reliable, no index required)
                self.qdrant_client.delete(
                    collection_name=settings.COLLECTION_NAME,
                    points_selector=models.PointIdsList(
                        points=point_ids
                    )
                )
                print(f"✅ Deleted {len(point_ids)} vectors for document: {document_id}")
            else:
                print(f"⚠️  No point IDs provided for document {document_id}, skipping Qdrant delete")
            
            # Remove from cache
            if document_id in self._documents:
                del self._documents[document_id]
            
            # Reload cache from Qdrant to ensure consistency
            self.load_documents_from_qdrant()
            
            return True
        except Exception as e:
            print(f"❌ Error deleting document vectors: {e}")
            return False
    
    def get_stats(self) -> dict:
        """Get collection statistics"""
        try:
            collection_info = self.qdrant_client.get_collection(settings.COLLECTION_NAME)
            docs_in_mem = len(self._documents)
            chunks_in_mem = sum(len(docs) for docs in self._documents.values())
            
            # Get vector count - API varies by Qdrant version
            vectors_count = 0
            if hasattr(collection_info, 'vectors_count'):
                vectors_count = collection_info.vectors_count
            elif hasattr(collection_info, 'points_count'):
                vectors_count = collection_info.points_count
            elif hasattr(collection_info.config, 'params') and hasattr(collection_info.config.params, 'vectors'):
                # Newer Qdrant API
                vectors_count = collection_info.points_count
            
            print(f"📊 Stats Debug:")
            print(f"   - Qdrant points: {collection_info.points_count}")
            print(f"   - Documents in memory: {docs_in_mem}")
            print(f"   - Chunks in memory: {chunks_in_mem}")
            
            return {
                "vectors_count": vectors_count,
                "points_count": collection_info.points_count,
                "documents_in_memory": docs_in_mem,
                "chunks_in_memory": chunks_in_mem
            }
        except Exception as e:
            print(f"❌ Error getting stats: {e}")
            import traceback
            traceback.print_exc()
            
            # Fallback: use in-memory counts
            docs_in_mem = len(self._documents)
            chunks_in_mem = sum(len(docs) for docs in self._documents.values())
            return {
                "vectors_count": chunks_in_mem,  # Use in-memory as fallback
                "points_count": chunks_in_mem,
                "documents_in_memory": docs_in_mem,
                "chunks_in_memory": chunks_in_mem
            }
    
    def clear(self):
        """Clear all documents (use with caution!)"""
        try:
            self.qdrant_client.delete_collection(settings.COLLECTION_NAME)
            self._ensure_collection()
            self.vector_store = QdrantVectorStore(
                client=self.qdrant_client,
                collection_name=settings.COLLECTION_NAME,
            )
            self._documents = {}
            print(f"Cleared collection: {settings.COLLECTION_NAME}")
        except Exception as e:
            print(f"Error clearing collection: {e}")
