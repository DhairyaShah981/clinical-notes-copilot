"""MongoDB Database Layer for Document & Session Persistence"""
from datetime import datetime
from typing import List, Dict, Any, Optional
from pymongo import MongoClient
from bson import ObjectId
import os


class Database:
    """MongoDB connection and operations"""
    
    def __init__(self, connection_string: str = None):
        # Use local MongoDB or MongoDB Atlas
        self.connection_string = connection_string or os.getenv(
            "MONGODB_URI", 
            "mongodb://localhost:27017"
        )
        self.client = MongoClient(self.connection_string)
        self.db = self.client["clinical_notes_db"]
        
        # Collections
        self.documents = self.db["documents"]
        self.sessions = self.db["sessions"]
        
        # Create indexes for faster queries
        self._create_indexes()
    
    def _create_indexes(self):
        """Create indexes for common queries"""
        self.documents.create_index("filename")
        self.documents.create_index("upload_date")
        self.documents.create_index("status")
        self.sessions.create_index("document_id")
        self.sessions.create_index("last_accessed")
    
    # ============ Document Operations ============
    
    def create_document(
        self, 
        filename: str,
        original_name: str,
        file_path: str,
        chunk_count: int,
        qdrant_point_ids: List[str],
        patient_id: str = None,
        ocr_method: str = "pymupdf"
    ) -> str:
        """Create a new document record"""
        doc = {
            "filename": filename,
            "original_name": original_name,
            "file_path": file_path,
            "patient_id": patient_id,
            "upload_date": datetime.utcnow(),
            "chunk_count": chunk_count,
            "qdrant_point_ids": qdrant_point_ids,
            "status": "indexed",
            "ocr_method": ocr_method
        }
        result = self.documents.insert_one(doc)
        return str(result.inserted_id)
    
    def get_document(self, document_id: str) -> Optional[Dict]:
        """Get document by ID"""
        try:
            doc = self.documents.find_one({"_id": ObjectId(document_id)})
            if doc:
                doc["_id"] = str(doc["_id"])
            return doc
        except Exception:
            return None
    
    def get_all_documents(self) -> List[Dict]:
        """Get all documents, sorted by upload date"""
        docs = list(self.documents.find().sort("upload_date", -1))
        for doc in docs:
            doc["_id"] = str(doc["_id"])
        return docs
    
    def delete_document(self, document_id: str) -> bool:
        """Delete a document and its sessions"""
        try:
            oid = ObjectId(document_id)
            # Delete associated sessions
            self.sessions.delete_many({"document_id": document_id})
            # Delete document
            result = self.documents.delete_one({"_id": oid})
            return result.deleted_count > 0
        except Exception:
            return False
    
    def document_exists(self, filename: str) -> Optional[Dict]:
        """Check if document already exists by filename"""
        doc = self.documents.find_one({"original_name": filename})
        if doc:
            doc["_id"] = str(doc["_id"])
        return doc
    
    # ============ Session Operations ============
    
    def create_session(self, document_id: str) -> str:
        """Create a new chat session for a document"""
        session = {
            "document_id": document_id,
            "created_at": datetime.utcnow(),
            "last_accessed": datetime.utcnow(),
            "messages": []
        }
        result = self.sessions.insert_one(session)
        return str(result.inserted_id)
    
    def get_session(self, session_id: str) -> Optional[Dict]:
        """Get session by ID"""
        try:
            session = self.sessions.find_one({"_id": ObjectId(session_id)})
            if session:
                session["_id"] = str(session["_id"])
            return session
        except Exception:
            return None
    
    def get_sessions_for_document(self, document_id: str) -> List[Dict]:
        """Get all sessions for a document"""
        sessions = list(
            self.sessions.find({"document_id": document_id})
            .sort("last_accessed", -1)
        )
        for s in sessions:
            s["_id"] = str(s["_id"])
        return sessions
    
    def add_message_to_session(
        self, 
        session_id: str, 
        role: str,
        content: str,
        sources: List[Dict] = None,
        tools_used: List[str] = None
    ):
        """Add a message to session history"""
        message = {
            "role": role,
            "content": content,
            "timestamp": datetime.utcnow(),
            "sources": sources or [],
            "tools_used": tools_used or []
        }
        
        self.sessions.update_one(
            {"_id": ObjectId(session_id)},
            {
                "$push": {"messages": message},
                "$set": {"last_accessed": datetime.utcnow()}
            }
        )
    
    def get_session_messages(self, session_id: str, limit: int = 20) -> List[Dict]:
        """Get recent messages from session for context"""
        session = self.get_session(session_id)
        if not session:
            return []
        
        messages = session.get("messages", [])
        # Return last N messages for context
        return messages[-limit:] if len(messages) > limit else messages
    
    def delete_session(self, session_id: str) -> bool:
        """Delete a session"""
        try:
            result = self.sessions.delete_one({"_id": ObjectId(session_id)})
            return result.deleted_count > 0
        except Exception:
            return False
    
    # ============ Stats ============
    
    def get_stats(self) -> Dict:
        """Get database statistics"""
        return {
            "total_documents": self.documents.count_documents({}),
            "total_sessions": self.sessions.count_documents({}),
            "indexed_documents": self.documents.count_documents({"status": "indexed"})
        }
    
    def close(self):
        """Close MongoDB connection"""
        self.client.close()


# Singleton instance
_db_instance = None

def get_database() -> Database:
    """Get database singleton"""
    global _db_instance
    if _db_instance is None:
        _db_instance = Database()
    return _db_instance
