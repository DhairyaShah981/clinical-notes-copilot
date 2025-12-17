"""
FastAPI Backend for Clinical Notes Search

Production-grade features:
- Document persistence in MongoDB
- Session management with chat history
- Multi-agent search with tool calling
- Hybrid search (Vector + BM25)
"""
from fastapi import FastAPI, UploadFile, File, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import shutil
import os
from pathlib import Path
from contextlib import asynccontextmanager

from llama_index.core import VectorStoreIndex

from config import settings
from document_processor import DocumentProcessor
from vector_store import VectorStoreManager
from hybrid_search import HybridSearchEngine
from agents import ClinicalSearchAgent, SimpleRAGEngine
from database import get_database, Database


# ============ Lifespan for startup/shutdown ============

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize services on startup, cleanup on shutdown"""
    print("🚀 Starting Clinical Notes Search API...")
    
    # Initialize services
    app.state.db = get_database()
    app.state.vector_store = VectorStoreManager()
    app.state.hybrid_engine = HybridSearchEngine()
    app.state.doc_processor = DocumentProcessor()
    
    # Load existing documents from Qdrant into memory for BM25
    print("📥 Loading existing documents from Qdrant...")
    app.state.vector_store.load_documents_from_qdrant()
    
    # Rebuild BM25 index if documents exist
    all_docs = app.state.vector_store.get_all_documents()
    if all_docs:
        app.state.hybrid_engine.index_documents(all_docs)
        print(f"✅ Restored {len(all_docs)} chunks for hybrid search")
    
    print("✅ API ready!")
    
    yield  # Server runs here
    
    # Cleanup
    print("👋 Shutting down...")
    app.state.db.close()


# ============ FastAPI App ============

app = FastAPI(
    title="Clinical Notes Search API",
    description="AI-powered clinical document search with hybrid RAG and multi-agent system",
    version="2.0.0",
    lifespan=lifespan
)

# Get allowed origins from environment variable
allowed_origins_str = os.getenv(
    "ALLOWED_ORIGINS",
    "http://localhost:3000,http://localhost:5173"
)
# Split by comma and strip whitespace from each URL
allowed_origins = [origin.strip() for origin in allowed_origins_str.split(",")]

print("🔒 CORS Configuration:")
print(f"   Allowed Origins: {allowed_origins}")

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

DATA_DIR = Path("data")
DATA_DIR.mkdir(exist_ok=True)


# ============ Request/Response Models ============

class QueryRequest(BaseModel):
    question: str
    session_id: Optional[str] = None  # For chat memory
    document_id: Optional[str] = None  # Query specific document
    use_agent: bool = True  # Use multi-agent (recommended)
    use_hybrid: bool = True


class QueryResponse(BaseModel):
    answer: str
    sources: List[Dict[str, Any]]
    search_type: str
    tools_used: List[str] = []
    session_id: Optional[str] = None


class DocumentResponse(BaseModel):
    id: str
    filename: str
    original_name: str
    upload_date: str
    chunk_count: int
    status: str
    ocr_method: str = "direct"


class SessionResponse(BaseModel):
    id: str
    document_id: str
    created_at: str
    message_count: int


# ============ Health & Stats ============

@app.get("/")
async def root():
    return {
        "message": "Clinical Notes Search API",
        "version": "2.0.0",
        "features": ["hybrid_search", "multi_agent", "session_memory", "document_persistence"]
    }


@app.get("/health")
async def health():
    db: Database = app.state.db
    vector_store: VectorStoreManager = app.state.vector_store
    
    return {
        "status": "healthy",
        "model": settings.LLM_MODEL,
        "embedding_model": settings.EMBEDDING_MODEL,
        "vector_store": "Qdrant Cloud",
        "database": "MongoDB",
        "stats": {
            **vector_store.get_stats(),
            **db.get_stats()
        }
    }


# ============ Document Management ============

@app.get("/documents", response_model=List[DocumentResponse])
async def list_documents():
    """List all uploaded documents"""
    db: Database = app.state.db
    docs = db.get_all_documents()
    
    return [
        DocumentResponse(
            id=doc["_id"],
            filename=doc.get("filename", ""),
            original_name=doc.get("original_name", ""),
            upload_date=doc.get("upload_date", "").isoformat() if doc.get("upload_date") else "",
            chunk_count=doc.get("chunk_count", 0),
            status=doc.get("status", "unknown"),
            ocr_method=doc.get("ocr_method", "direct")
        )
        for doc in docs
    ]


@app.post("/upload")
async def upload_document(files: List[UploadFile] = File(...)):
    """Upload and index documents (PDF, DOCX, TXT)"""
    db: Database = app.state.db
    vector_store: VectorStoreManager = app.state.vector_store
    hybrid_engine: HybridSearchEngine = app.state.hybrid_engine
    doc_processor: DocumentProcessor = app.state.doc_processor
    
    # Supported formats
    SUPPORTED_FORMATS = {'.pdf', '.docx', '.txt'}
    
    results = []
    
    for file in files:
        # Check file extension
        file_ext = Path(file.filename).suffix.lower()
        if file_ext not in SUPPORTED_FORMATS:
            raise HTTPException(400, f"Unsupported format: {file.filename}. Supported: PDF, DOCX, TXT")
        
        # Check if document already exists
        existing = db.document_exists(file.filename)
        if existing:
            results.append({
                "filename": file.filename,
                "status": "exists",
                "document_id": existing["_id"],
                "message": "Document already indexed"
            })
            continue
        
        # Save file
        safe_filename = file.filename.replace(" ", "_")
        file_path = DATA_DIR / safe_filename
        with open(file_path, "wb") as f:
            shutil.copyfileobj(file.file, f)
        
        # Process document (auto-detects format)
        docs = doc_processor.create_documents(str(file_path))
        
        if not docs:
            raise HTTPException(400, f"Could not extract text from {file.filename}")
        
        # Get OCR method from document metadata
        ocr_method = docs[0].metadata.get("ocr_method", "direct") if docs else "direct"
        ocr_quality = docs[0].metadata.get("ocr_quality", 0.0) if docs else 0.0
        
        # Create MongoDB document record first (to get ID)
        document_id = db.create_document(
            filename=safe_filename,
            original_name=file.filename,
            file_path=str(file_path),
            chunk_count=len(docs),
            qdrant_point_ids=[],  # Will update after indexing
            ocr_method=ocr_method
        )
        
        # Index to Qdrant with document_id tracking
        index, point_ids = vector_store.index_documents(docs, document_id=document_id)
        
        # Update MongoDB with point IDs
        db.documents.update_one(
            {"_id": db.documents.find_one({"_id": __import__("bson").ObjectId(document_id)})["_id"]},
            {"$set": {"qdrant_point_ids": point_ids}}
        )
        
        # Update BM25 index with ALL documents
        all_docs = vector_store.get_all_documents()
        if all_docs:
            hybrid_engine.index_documents(all_docs)
            print(f"🔍 BM25 index updated with {len(all_docs)} total chunks")
        
        # Determine display status based on OCR method
        display_status = "indexed"
        if ocr_method == "nanonets":
            display_status = "ocr_complete"
        elif ocr_method == "direct_fallback":
            display_status = "ocr_fallback"
        
        results.append({
            "filename": file.filename,
            "status": display_status,
            "document_id": document_id,
            "chunks": len(docs),
            "ocr_method": ocr_method,
            "ocr_quality": round(ocr_quality, 2)
        })
        
        print(f"✅ Indexed {file.filename}: {len(docs)} chunks (doc_id: {document_id})")
        print(f"📊 Total in memory: {len(vector_store.get_all_documents())} chunks")
    
    return {
        "message": f"Processed {len(files)} file(s)",
        "files": results,
        "stats": vector_store.get_stats()
    }


@app.get("/documents/{document_id}")
async def get_document(document_id: str):
    """Get document details"""
    db: Database = app.state.db
    doc = db.get_document(document_id)
    
    if not doc:
        raise HTTPException(404, "Document not found")
    
    return doc


@app.delete("/documents/{document_id}")
async def delete_document(document_id: str):
    """Delete a document and its vectors"""
    db: Database = app.state.db
    vector_store: VectorStoreManager = app.state.vector_store
    hybrid_engine: HybridSearchEngine = app.state.hybrid_engine
    
    doc = db.get_document(document_id)
    if not doc:
        raise HTTPException(404, "Document not found")
    
    print(f"\n🗑️  Deleting document: {document_id}")
    print(f"   📄 Filename: {doc.get('filename')}")
    print(f"   📍 Point IDs: {len(doc.get('qdrant_point_ids', []))} vectors")
    
    # Delete from Qdrant using point IDs (reliable method)
    point_ids = doc.get("qdrant_point_ids", [])
    vector_store.delete_document_vectors(document_id, point_ids=point_ids)
    
    # Delete file
    file_path = Path(doc.get("file_path", ""))
    if file_path.exists():
        file_path.unlink()
        print(f"   🗑️  Deleted file: {file_path}")
    
    # Delete from MongoDB (do this AFTER Qdrant/file deletion)
    db.delete_document(document_id)
    print(f"   🗑️  Deleted from MongoDB")
    
    # Rebuild BM25 index with remaining documents
    all_docs = vector_store.get_all_documents()
    if all_docs:
        hybrid_engine.index_documents(all_docs)
        print(f"   🔍 BM25 index rebuilt with {len(all_docs)} chunks")
    else:
        hybrid_engine.documents = []
        hybrid_engine.bm25 = None
        print(f"   📭 BM25 index cleared (no documents remaining)")
    
    print(f"✅ Document {document_id} deleted successfully\n")
    
    return {"message": f"Deleted document {document_id}"}


@app.delete("/documents")
async def clear_all_documents():
    """Clear all documents (use with caution!)"""
    db: Database = app.state.db
    vector_store: VectorStoreManager = app.state.vector_store
    hybrid_engine: HybridSearchEngine = app.state.hybrid_engine
    
    # Clear Qdrant
    vector_store.clear()
    
    # Clear MongoDB
    db.documents.delete_many({})
    db.sessions.delete_many({})
    
    # Clear BM25
    hybrid_engine.documents = []
    hybrid_engine.bm25 = None
    
    # Clear data directory (all supported formats)
    for pattern in ["*.pdf", "*.docx", "*.txt"]:
        for f in DATA_DIR.glob(pattern):
            f.unlink()
    
    return {"message": "All documents cleared"}


# ============ Session Management ============

@app.post("/sessions")
async def create_session(document_id: str):
    """Create a new chat session for a document"""
    db: Database = app.state.db
    
    # Verify document exists
    doc = db.get_document(document_id)
    if not doc:
        raise HTTPException(404, "Document not found")
    
    session_id = db.create_session(document_id)
    
    return {
        "session_id": session_id,
        "document_id": document_id,
        "message": "Session created"
    }


@app.get("/sessions/{session_id}")
async def get_session(session_id: str):
    """Get session details including chat history"""
    db: Database = app.state.db
    session = db.get_session(session_id)
    
    if not session:
        raise HTTPException(404, "Session not found")
    
    return session


@app.get("/documents/{document_id}/sessions")
async def get_document_sessions(document_id: str):
    """Get all sessions for a document"""
    db: Database = app.state.db
    sessions = db.get_sessions_for_document(document_id)
    
    return [
        SessionResponse(
            id=s["_id"],
            document_id=s["document_id"],
            created_at=s.get("created_at", "").isoformat() if s.get("created_at") else "",
            message_count=len(s.get("messages", []))
        )
        for s in sessions
    ]


@app.delete("/sessions/{session_id}")
async def delete_session(session_id: str):
    """Delete a chat session"""
    db: Database = app.state.db
    
    if not db.delete_session(session_id):
        raise HTTPException(404, "Session not found")
    
    return {"message": "Session deleted"}


# ============ Query / Search ============

@app.post("/query", response_model=QueryResponse)
async def query(request: QueryRequest):
    """
    Query the clinical notes
    
    Features:
    - Multi-agent system with automatic tool selection
    - Hybrid search (Vector + BM25)
    - Session memory for context
    """
    db: Database = app.state.db
    vector_store: VectorStoreManager = app.state.vector_store
    hybrid_engine: HybridSearchEngine = app.state.hybrid_engine
    
    # ✅ CRITICAL FIX: Filter by document_id if provided
    if request.document_id:
        print(f"\n{'='*60}")
        print(f"🔍 DOCUMENT-SPECIFIC QUERY")
        print(f"   📄 Document ID: {request.document_id}")
        print(f"{'='*60}\n")
        
        # Get document metadata from MongoDB
        doc_metadata = db.get_document(request.document_id)
        
        if not doc_metadata:
            raise HTTPException(404, f"Document {request.document_id} not found in database")
        
        # Try to get chunks by document_id first
        filtered_docs = vector_store.get_documents_for_document_id(request.document_id)
        
        # Fallback: If no chunks found by ID, try by source filename
        if not filtered_docs:
            source_filename = doc_metadata.get("filename")
            print(f"   ⚠️  No chunks found by document_id, trying by filename: {source_filename}")
            filtered_docs = vector_store.get_documents_by_source(source_filename)
        
        if not filtered_docs:
            raise HTTPException(404, f"No chunks found for document {request.document_id} (filename: {doc_metadata.get('filename')})")
        
        print(f"   ✅ Found {len(filtered_docs)} chunks for this document")
        print(f"   🔍 Searching ONLY within this document\n")
        
        # ✅ Use global index (retrieve from Qdrant, filter post-retrieval in Python)
        index = vector_store.get_index()
        
        # Determine filter parameters for post-retrieval filtering
        filter_document_id = None
        filter_source = None
        
        # Try document_id filter first
        if vector_store.get_documents_for_document_id(request.document_id):
            filter_document_id = request.document_id
            print(f"   🔍 Will filter results by document_id: {request.document_id}")
        else:
            # Fallback to source filename filter
            filter_source = doc_metadata.get("filename")
            print(f"   🔍 Will filter results by source filename: {filter_source}")
        
        # Build document-specific BM25 index (in-memory, fast)
        # Use silent=True to avoid log spam on every query
        doc_hybrid_engine = HybridSearchEngine()
        doc_hybrid_engine.index_documents(filtered_docs, silent=True)
        
        # DEBUG: Print chunk contents to diagnose search issues
        print(f"   📋 DEBUG: Chunk contents for this document:")
        for i, doc in enumerate(filtered_docs[:3]):  # Show first 3 chunks
            text_preview = doc.text[:200].replace('\n', ' ')
            print(f"      Chunk {i+1}: {text_preview}...")
        
        doc_chunk_count = len(filtered_docs)
        
    else:
        # Search ALL documents
        print(f"\n{'='*60}")
        print(f"🔍 MULTI-DOCUMENT QUERY")
        print(f"{'='*60}\n")
        
        all_docs = vector_store.get_all_documents()
        chunks_available = len(all_docs)
        
        print(f"   📦 Total chunks available: {chunks_available}")
        print(f"   🔍 Searching across ALL documents\n")
        
        if chunks_available == 0:
            raise HTTPException(400, "No documents indexed. Please upload PDF files first.")
        
        index = vector_store.get_index()
        doc_hybrid_engine = hybrid_engine
        doc_chunk_count = None
        filter_document_id = None  # No filtering for multi-document queries
        filter_source = None
    
    # Get chat history for context
    chat_history = []
    if request.session_id:
        chat_history = db.get_session_messages(request.session_id, limit=10)
    
    # Use agent or simple RAG
    if request.use_agent:
        agent = ClinicalSearchAgent(
            index, 
            doc_hybrid_engine, 
            document_chunk_count=doc_chunk_count,
            filter_document_id=filter_document_id,  # Pass filter params for post-retrieval filtering
            filter_source=filter_source
        )
        result = agent.query(
            request.question,
            chat_history=chat_history,
            max_iterations=3
        )
    else:
        engine = SimpleRAGEngine(index, doc_hybrid_engine)
        result = engine.query(
            request.question,
            use_hybrid=request.use_hybrid,
            chat_history=chat_history
        )
    
    # Save to session if provided
    session_id = request.session_id
    if session_id:
        # Save user message
        db.add_message_to_session(session_id, "user", request.question)
        # Save assistant response
        db.add_message_to_session(
            session_id, 
            "assistant", 
            result["answer"],
            sources=result.get("sources", []),
            tools_used=result.get("tools_used", [])
        )
    
    return QueryResponse(
        answer=result["answer"],
        sources=result.get("sources", []),
        search_type=result.get("search_type", "unknown"),
        tools_used=result.get("tools_used", []),
        session_id=session_id
    )


# ============ Main ============

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
