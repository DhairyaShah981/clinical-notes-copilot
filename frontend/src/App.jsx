import React, { useState, useRef, useEffect } from 'react';
import { 
  Search, 
  Upload, 
  FileText, 
  Loader2, 
  Trash2, 
  Stethoscope, 
  MessageSquare,
  Activity,
  FileUp,
  ChevronRight,
  Database,
  Zap,
  FolderOpen,
  Clock,
  Plus,
  ArrowLeft,
  Bot,
  Wrench
} from 'lucide-react';
import axios from 'axios';

const API = import.meta.env.VITE_API_URL || 'http://localhost:8000';

// Helper function to clean markdown symbols from text
const cleanText = (text) => {
  if (!text) return '';
  return text
    .replace(/#{1,6}\s/g, '')
    .replace(/\*\*/g, '')
    .replace(/\*/g, '')
    .replace(/\[([^\]]+)\]\([^\)]+\)/g, '$1')
    .replace(/`{1,3}/g, '')
    .replace(/^\s*[-*+]\s/gm, '')
    .replace(/\n{3,}/g, '\n\n')
    .trim();
};

export default function App() {
  // View state
  const [view, setView] = useState('library'); // 'library' | 'chat'
  
  // Document library state
  const [documents, setDocuments] = useState([]);
  const [selectedDocument, setSelectedDocument] = useState(null);
  const [sessions, setSessions] = useState([]);
  const [currentSession, setCurrentSession] = useState(null);
  
  // Chat state
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState('');
  const [loading, setLoading] = useState(false);
  const [uploading, setUploading] = useState(false);
  const [apiStatus, setApiStatus] = useState('checking');
  const [stats, setStats] = useState(null);
  
  const fileRef = useRef(null);
  const endRef = useRef(null);

  useEffect(() => {
    endRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  useEffect(() => {
    checkHealth();
    loadDocuments();
  }, []);

  const checkHealth = async () => {
    try {
      const r = await axios.get(`${API}/health`);
      setStats(r.data.stats);
      setApiStatus('connected');
    } catch {
      setApiStatus('disconnected');
    }
  };

  const loadDocuments = async () => {
    try {
      const r = await axios.get(`${API}/documents`);
      setDocuments(r.data);
    } catch (e) {
      console.error('Failed to load documents:', e);
    }
  };

  const loadSessions = async (documentId) => {
    try {
      const r = await axios.get(`${API}/documents/${documentId}/sessions`);
      setSessions(r.data);
    } catch (e) {
      console.error('Failed to load sessions:', e);
      setSessions([]);
    }
  };

  const selectDocument = async (doc) => {
    setSelectedDocument(doc);
    await loadSessions(doc.id);
  };

  const createNewSession = async () => {
    if (!selectedDocument) return;
    
    try {
      const r = await axios.post(`${API}/sessions?document_id=${selectedDocument.id}`);
      setCurrentSession({ id: r.data.session_id, messages: [] });
      setMessages([]);
      setView('chat');
    } catch (e) {
      console.error('Failed to create session:', e);
    }
  };

  const continueSession = async (session) => {
    try {
      const r = await axios.get(`${API}/sessions/${session.id}`);
      setCurrentSession(r.data);
      
      // Convert session messages to chat format
      const chatMessages = (r.data.messages || []).map(m => ({
        type: m.role === 'user' ? 'user' : 'assistant',
        content: m.content,
        sources: m.sources || [],
        tools_used: m.tools_used || []
      }));
      setMessages(chatMessages);
      setView('chat');
    } catch (e) {
      console.error('Failed to load session:', e);
    }
  };

  const upload = async (e) => {
    const files = e.target.files;
    if (!files.length) return;
    
    setUploading(true);
    const form = new FormData();
    [...files].forEach(f => form.append('files', f));
    
    try {
      const r = await axios.post(`${API}/upload`, form);
      await loadDocuments();
      await checkHealth();
      
      // Show upload result
      const results = r.data.files || [];
      const newDocs = results.filter(f => ['indexed', 'ocr_complete', 'ocr_fallback'].includes(f.status));
      const existingDocs = results.filter(f => f.status === 'exists');
      const ocrDocs = results.filter(f => f.ocr_method === 'nanonets');
      
      let message = '';
      if (newDocs.length > 0) {
        message += `✓ Indexed ${newDocs.length} document(s). `;
        if (ocrDocs.length > 0) {
          message += `\n🔍 ${ocrDocs.length} used OCR processing. `;
        }
      }
      if (existingDocs.length > 0) {
        message += `\n${existingDocs.length} document(s) already existed.`;
      }
      
      alert(message || 'Upload complete!');
    } catch (e) {
      alert(`Upload failed: ${e.response?.data?.detail || e.message}`);
    }
    
    setUploading(false);
    fileRef.current.value = '';
  };

  const query = async (e) => {
    e.preventDefault();
    if (!input.trim() || loading) return;
    
    const q = input.trim();
    setInput('');
    setMessages(prev => [...prev, { type: 'user', content: q }]);
    setLoading(true);
    
    try {
      const r = await axios.post(`${API}/query`, { 
        question: q, 
        session_id: currentSession?.id,
        document_id: selectedDocument?.id,
        use_agent: true,
        use_hybrid: true
      });
      
      setMessages(prev => [...prev, { 
        type: 'assistant', 
        content: r.data.answer, 
        sources: r.data.sources,
        searchType: r.data.search_type,
        tools_used: r.data.tools_used || []
      }]);
    } catch (e) {
      setMessages(prev => [...prev, { 
        type: 'error', 
        content: `Error: ${e.response?.data?.detail || e.message}` 
      }]);
    }
    
    setLoading(false);
  };

  const deleteDocument = async (docId) => {
    if (!confirm('Delete this document and all its sessions?')) return;
    
    try {
      await axios.delete(`${API}/documents/${docId}`);
      await loadDocuments();
      await checkHealth();
      setSelectedDocument(null);
      setSessions([]);
    } catch (e) {
      alert(`Delete failed: ${e.message}`);
    }
  };

  const backToLibrary = () => {
    setView('library');
    setMessages([]);
    setCurrentSession(null);
  };

  const samples = [
    "What medications is the patient currently taking?",
    "What is the patient's HbA1c level?",
    "Summarize the diagnosis and findings",
    "Are there any documented drug allergies?",
    "What is the primary diagnosis and ICD code?"
  ];

  // ============ LIBRARY VIEW ============
  if (view === 'library') {
    return (
      <div className="min-h-screen bg-slate-50">
        {/* Header */}
        <header className="bg-gradient-to-r from-slate-800 via-slate-900 to-slate-800 text-white shadow-xl border-b border-slate-700">
          <div className="max-w-7xl mx-auto px-6 py-5">
            <div className="flex items-center justify-between">
              <div className="flex items-center gap-4">
                <div className="p-2.5 bg-medical-accent/20 rounded-xl border border-medical-accent/30">
                  <Stethoscope className="h-7 w-7 text-medical-accent" />
                </div>
                <div>
                  <h1 className="text-2xl font-bold tracking-tight text-white">
                    Clinical Notes Search
                  </h1>
                  <p className="text-slate-300 text-sm font-medium">
                    AI-Powered Document Analysis • Multi-Agent RAG
                  </p>
                </div>
              </div>
              
              <div className="flex items-center gap-4">
                <div className={`flex items-center gap-2 px-4 py-2 rounded-full text-sm font-medium ${
                  apiStatus === 'connected' 
                    ? 'bg-emerald-500/30 text-emerald-100 border border-emerald-400/30' 
                    : 'bg-red-500/30 text-red-100 border border-red-400/30'
                }`}>
                  <span className={`w-2 h-2 rounded-full ${
                    apiStatus === 'connected' ? 'bg-emerald-300 animate-pulse' : 'bg-red-300'
                  }`} />
                  {apiStatus === 'connected' ? 'Connected' : 'Disconnected'}
                </div>
                
                {stats && (
                  <div className="flex items-center gap-2 bg-white/15 px-4 py-2 rounded-full border border-white/20">
                    <Database className="h-4 w-4 text-slate-300" />
                    <span className="text-sm font-medium text-white">
                      {stats.vectors_count || 0} vectors
                    </span>
                  </div>
                )}
              </div>
            </div>
          </div>
        </header>

        {/* Main Content */}
        <div className="max-w-7xl mx-auto px-6 py-8">
          <div className="grid grid-cols-12 gap-8">
            
            {/* Left Panel - Upload & Info */}
            <div className="col-span-4 space-y-6">
              {/* Upload Card */}
              <div className="bg-white rounded-2xl shadow-sm border border-slate-200 p-6">
                <h3 className="font-semibold text-slate-800 flex items-center gap-2 mb-4">
                  <FileUp className="h-5 w-5 text-medical-accent" />
                  Upload New Document
                </h3>
                <input type="file" ref={fileRef} onChange={upload} multiple accept=".pdf" className="hidden" />
                <button 
                  onClick={() => fileRef.current.click()} 
                  disabled={uploading || apiStatus !== 'connected'}
                  className="w-full py-4 bg-gradient-to-br from-medical-accent to-blue-600 text-white 
                    rounded-xl hover:from-blue-600 hover:to-blue-700 transition-all duration-200
                    flex items-center justify-center gap-3 disabled:opacity-50 disabled:cursor-not-allowed
                    font-medium shadow-lg shadow-blue-500/20"
                >
                  {uploading ? (
                    <><Loader2 className="h-5 w-5 animate-spin" /> Processing...</>
                  ) : (
                    <><Plus className="h-5 w-5" /> Upload PDF Files</>
                  )}
                </button>
                <p className="text-xs text-slate-500 mt-3 text-center">
                  Documents are stored persistently in MongoDB + Qdrant Cloud
                </p>
              </div>

              {/* Info Card */}
              <div className="bg-gradient-to-br from-slate-800 to-slate-900 rounded-2xl p-6 text-white">
                <div className="flex items-center gap-2 mb-3">
                  <Bot className="h-5 w-5 text-amber-400" />
                  <span className="font-semibold">Multi-Agent System</span>
                </div>
                <p className="text-sm text-slate-300 leading-relaxed mb-4">
                  The AI automatically selects the best search strategy:
                </p>
                <ul className="text-sm text-slate-400 space-y-2">
                  <li className="flex items-center gap-2">
                    <Wrench className="h-4 w-4 text-blue-400" />
                    <span><strong className="text-slate-200">Semantic</strong> - concepts, symptoms</span>
                  </li>
                  <li className="flex items-center gap-2">
                    <Wrench className="h-4 w-4 text-green-400" />
                    <span><strong className="text-slate-200">Keyword</strong> - lab values, drug names</span>
                  </li>
                  <li className="flex items-center gap-2">
                    <Wrench className="h-4 w-4 text-purple-400" />
                    <span><strong className="text-slate-200">Hybrid</strong> - complex queries</span>
                  </li>
                </ul>
                <div className="mt-4 pt-4 border-t border-white/10 text-xs text-slate-500">
                  Powered by OpenAI + Qdrant + MongoDB
                </div>
              </div>
            </div>

            {/* Right Panel - Document Library */}
            <div className="col-span-8">
              <div className="bg-white rounded-2xl shadow-sm border border-slate-200 overflow-hidden">
                <div className="px-6 py-4 border-b border-slate-100 bg-slate-50">
                  <h3 className="font-semibold text-slate-800 flex items-center gap-2">
                    <FolderOpen className="h-5 w-5 text-medical-accent" />
                    Document Library
                    <span className="ml-2 px-2 py-0.5 bg-medical-accent/10 text-medical-accent text-xs rounded-full">
                      {documents.length} document{documents.length !== 1 ? 's' : ''}
                    </span>
                  </h3>
                </div>
                
                <div className="divide-y divide-slate-100">
                  {documents.length === 0 ? (
                    <div className="text-center py-16 text-slate-400">
                      <FileText className="h-12 w-12 mx-auto mb-4 opacity-50" />
                      <p className="font-medium">No documents uploaded yet</p>
                      <p className="text-sm mt-1">Upload a PDF to get started</p>
                    </div>
                  ) : (
                    documents.map(doc => (
                      <div 
                        key={doc.id}
                        className={`p-4 hover:bg-slate-50 cursor-pointer transition-colors ${
                          selectedDocument?.id === doc.id ? 'bg-medical-light border-l-4 border-medical-accent' : ''
                        }`}
                        onClick={() => selectDocument(doc)}
                      >
                        <div className="flex items-center justify-between">
                          <div className="flex items-center gap-3">
                            <div className="p-2 bg-slate-100 rounded-lg">
                              <FileText className="h-5 w-5 text-slate-600" />
                            </div>
                            <div>
                              <p className="font-medium text-slate-800">{doc.original_name}</p>
                              <p className="text-xs text-slate-500">
                                {doc.chunk_count} chunks • Uploaded {new Date(doc.upload_date).toLocaleDateString()}
                              </p>
                            </div>
                          </div>
                          <div className="flex items-center gap-2">
                            {/* OCR Method Badge */}
                            {doc.ocr_method && doc.ocr_method !== 'direct' && (
                              <span className={`px-2 py-1 rounded text-xs font-medium ${
                                doc.ocr_method === 'nanonets' 
                                  ? 'bg-purple-100 text-purple-700' 
                                  : 'bg-orange-100 text-orange-700'
                              }`}>
                                {doc.ocr_method === 'nanonets' ? '🔍 OCR' : '⚠️ OCR'}
                              </span>
                            )}
                            {/* Status Badge */}
                            <span className={`px-2 py-1 rounded text-xs font-medium ${
                              doc.status === 'indexed' || doc.status === 'ocr_complete'
                                ? 'bg-emerald-100 text-emerald-700' 
                                : doc.status === 'processing'
                                ? 'bg-blue-100 text-blue-700 animate-pulse'
                                : 'bg-amber-100 text-amber-700'
                            }`}>
                              {doc.status === 'ocr_complete' ? 'indexed' : doc.status}
                            </span>
                            <button 
                              onClick={(e) => { e.stopPropagation(); deleteDocument(doc.id); }}
                              className="p-1.5 text-slate-400 hover:text-red-500 hover:bg-red-50 rounded"
                            >
                              <Trash2 className="h-4 w-4" />
                            </button>
                          </div>
                        </div>
                      </div>
                    ))
                  )}
                </div>
              </div>

              {/* Sessions Panel */}
              {selectedDocument && (
                <div className="mt-6 bg-white rounded-2xl shadow-sm border border-slate-200 overflow-hidden">
                  <div className="px-6 py-4 border-b border-slate-100 bg-slate-50 flex items-center justify-between">
                    <h3 className="font-semibold text-slate-800 flex items-center gap-2">
                      <Clock className="h-5 w-5 text-medical-accent" />
                      Sessions for "{selectedDocument.original_name}"
                    </h3>
                    <button
                      onClick={createNewSession}
                      className="px-4 py-2 bg-medical-accent text-white rounded-lg hover:bg-blue-600 
                        flex items-center gap-2 text-sm font-medium transition-colors"
                    >
                      <Plus className="h-4 w-4" />
                      New Session
                    </button>
                  </div>
                  
                  <div className="divide-y divide-slate-100 max-h-64 overflow-y-auto">
                    {sessions.length === 0 ? (
                      <div className="text-center py-8 text-slate-400">
                        <MessageSquare className="h-8 w-8 mx-auto mb-2 opacity-50" />
                        <p className="text-sm">No sessions yet. Start a new one!</p>
                      </div>
                    ) : (
                      sessions.map(session => (
                        <div 
                          key={session.id}
                          className="p-4 hover:bg-slate-50 cursor-pointer transition-colors flex items-center justify-between"
                          onClick={() => continueSession(session)}
                        >
                          <div className="flex items-center gap-3">
                            <MessageSquare className="h-5 w-5 text-slate-400" />
                            <div>
                              <p className="font-medium text-slate-700">
                                Session from {new Date(session.created_at).toLocaleString()}
                              </p>
                              <p className="text-xs text-slate-500">{session.message_count} messages</p>
                            </div>
                          </div>
                          <ChevronRight className="h-5 w-5 text-slate-400" />
                        </div>
                      ))
                    )}
                  </div>
                </div>
              )}
            </div>
          </div>
        </div>
      </div>
    );
  }

  // ============ CHAT VIEW ============
  return (
    <div className="min-h-screen bg-slate-50">
      {/* Header */}
      <header className="bg-gradient-to-r from-slate-800 via-slate-900 to-slate-800 text-white shadow-xl border-b border-slate-700">
        <div className="max-w-7xl mx-auto px-6 py-4">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-4">
              <button
                onClick={backToLibrary}
                className="p-2 hover:bg-white/10 rounded-lg transition-colors"
              >
                <ArrowLeft className="h-5 w-5" />
              </button>
              <div className="p-2 bg-medical-accent/20 rounded-xl border border-medical-accent/30">
                <Stethoscope className="h-6 w-6 text-medical-accent" />
              </div>
              <div>
                <h1 className="text-lg font-bold text-white">
                  {selectedDocument?.original_name || 'Clinical Notes Search'}
                </h1>
                <p className="text-slate-400 text-xs">
                  Session ID: {currentSession?.id?.slice(0, 8)}...
                </p>
              </div>
            </div>
            
            <div className="flex items-center gap-3">
              <div className="flex items-center gap-2 bg-white/10 px-3 py-1.5 rounded-full text-sm">
                <Bot className="h-4 w-4 text-amber-400" />
                <span className="text-slate-300">Multi-Agent Active</span>
              </div>
            </div>
          </div>
        </div>
      </header>

      {/* Main Chat */}
      <div className="max-w-5xl mx-auto px-6 py-6">
        <div className="grid grid-cols-12 gap-6">
          
          {/* Sidebar - Sample Queries */}
          <div className="col-span-3">
            <div className="bg-white rounded-2xl shadow-sm border border-slate-200 p-4">
              <h3 className="font-semibold text-slate-700 mb-3 text-sm">Quick Questions</h3>
              <div className="space-y-2">
                {samples.map((s, i) => (
                  <button 
                    key={i} 
                    onClick={() => setInput(s)}
                    className="w-full text-left text-xs p-2.5 bg-slate-50 hover:bg-medical-light 
                      rounded-lg text-slate-600 hover:text-medical-blue transition-colors"
                  >
                    {s}
                  </button>
                ))}
              </div>
            </div>
          </div>

          {/* Chat Area */}
          <div className="col-span-9">
            <div className="bg-white rounded-2xl shadow-sm border border-slate-200 
              flex flex-col h-[calc(100vh-180px)] overflow-hidden">
              
              {/* Messages */}
              <div className="flex-1 overflow-y-auto p-6 space-y-4">
                {messages.length === 0 && (
                  <div className="text-center py-16">
                    <MessageSquare className="h-12 w-12 mx-auto mb-4 text-slate-300" />
                    <h3 className="text-lg font-semibold text-slate-700 mb-2">
                      Start Your Search
                    </h3>
                    <p className="text-slate-500 text-sm max-w-md mx-auto">
                      Ask questions about the clinical notes. The AI will automatically 
                      choose the best search strategy.
                    </p>
                  </div>
                )}
                
                {messages.map((m, i) => (
                  <div 
                    key={i} 
                    className={`flex ${m.type === 'user' ? 'justify-end' : 'justify-start'}`}
                  >
                    <div className={`max-w-2xl rounded-2xl px-5 py-4 ${
                      m.type === 'user' 
                        ? 'bg-medical-blue text-white rounded-br-md' 
                        : m.type === 'error' 
                        ? 'bg-red-50 text-red-700 border border-red-200' 
                        : 'bg-slate-100 text-slate-800 rounded-bl-md'
                    }`}>
                      {m.type === 'assistant' && m.tools_used?.length > 0 && (
                        <div className="flex items-center gap-2 mb-2 pb-2 border-b border-slate-200/50">
                          <Activity className="h-4 w-4 text-medical-accent" />
                          <span className="text-xs font-medium text-medical-accent">
                            Tools: {m.tools_used.join(', ')}
                          </span>
                        </div>
                      )}
                      
                      <p className="whitespace-pre-wrap leading-relaxed">{m.content}</p>
                      
                      {m.sources?.length > 0 && (
                        <div className="mt-4 pt-4 border-t border-slate-300">
                          <div className="flex items-center gap-2 mb-3">
                            <FileText className="h-4 w-4 text-medical-accent" />
                            <span className="text-sm font-semibold text-slate-700">
                              Clinical Evidence ({m.sources.length} sources)
                            </span>
                          </div>
                          <div className="space-y-2">
                            {m.sources.slice(0, 3).map((s, j) => (
                              <div 
                                key={j} 
                                className="bg-white rounded-lg p-3 border-l-4 border-medical-accent text-sm"
                              >
                                <div className="flex items-center gap-2 mb-1">
                                  <span className="inline-flex items-center justify-center w-5 h-5 
                                    bg-medical-accent text-white text-xs font-bold rounded-full">
                                    {j + 1}
                                  </span>
                                  <span className="font-medium text-slate-700">
                                    {s.source} • Page {s.page}
                                  </span>
                                </div>
                                <p className="text-slate-600 text-xs pl-7">
                                  {cleanText(s.text).substring(0, 150)}...
                                </p>
                              </div>
                            ))}
                          </div>
                        </div>
                      )}
                    </div>
                  </div>
                ))}
                
                {loading && (
                  <div className="flex justify-start">
                    <div className="bg-slate-100 rounded-2xl px-5 py-4 flex items-center gap-3">
                      <div className="flex gap-1">
                        <span className="w-2 h-2 bg-medical-accent rounded-full animate-bounce" style={{ animationDelay: '0ms' }} />
                        <span className="w-2 h-2 bg-medical-accent rounded-full animate-bounce" style={{ animationDelay: '150ms' }} />
                        <span className="w-2 h-2 bg-medical-accent rounded-full animate-bounce" style={{ animationDelay: '300ms' }} />
                      </div>
                      <span className="text-slate-500 text-sm">Agent searching...</span>
                    </div>
                  </div>
                )}
                
                <div ref={endRef} />
              </div>

              {/* Input */}
              <div className="border-t border-slate-200 p-4 bg-slate-50">
                <form onSubmit={query} className="flex gap-3">
                  <input 
                    type="text" 
                    value={input} 
                    onChange={e => setInput(e.target.value)}
                    placeholder="Ask about the clinical notes..."
                    className="flex-1 px-4 py-3 border border-slate-200 rounded-xl 
                      focus:outline-none focus:ring-2 focus:ring-medical-accent/30 focus:border-medical-accent
                      bg-white text-slate-800 placeholder:text-slate-400"
                  />
                  <button 
                    type="submit" 
                    disabled={loading || !input.trim()}
                    className="px-6 py-3 bg-medical-blue text-white rounded-xl 
                      hover:bg-slate-800 disabled:opacity-50 disabled:cursor-not-allowed
                      flex items-center gap-2 font-medium transition-colors
                      shadow-lg shadow-slate-900/20"
                  >
                    <Search className="h-5 w-5" />
                    Search
                  </button>
                </form>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
