# Clinical Notes Search System v2.0

Production-grade AI-powered clinical document search using Multi-Agent RAG with Hybrid Search.

## Architecture

```
┌──────────────────────────────────────────────────────────────────┐
│                         React Frontend                            │
│   - Document Library (upload once, search forever)               │
│   - Session Management (continue previous chats)                  │
│   - Chat with Memory                                              │
└──────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌──────────────────────────────────────────────────────────────────┐
│                        FastAPI Backend                            │
│  ┌────────────────────────────────────────────────────────────┐  │
│  │         Multi-Agent Orchestrator (OpenAI Function Calling)  │  │
│  │   ┌──────────────┐ ┌──────────────┐ ┌──────────────────┐   │  │
│  │   │  Semantic    │ │   Keyword    │ │     Hybrid       │   │  │
│  │   │   Search     │ │   Search     │ │     Search       │   │  │
│  │   │  (Vector)    │ │   (BM25)     │ │   (RRF Fusion)   │   │  │
│  │   └──────────────┘ └──────────────┘ └──────────────────┘   │  │
│  └────────────────────────────────────────────────────────────┘  │
└──────────────────────────────────────────────────────────────────┘
        │                    │                    │
        ▼                    ▼                    ▼
┌─────────────┐      ┌─────────────┐      ┌─────────────┐
│   MongoDB   │      │   Qdrant    │      │   OpenAI    │
│             │      │   Cloud     │      │             │
│ - Documents │◄────►│ - Vectors   │      │ - GPT-4o    │
│ - Sessions  │      │ - Search    │      │ - Embeddings│
│ - Chat Hist │      │             │      │             │
└─────────────┘      └─────────────┘      └─────────────┘
```

## Key Features

### ✅ Persistent Storage
- **Documents** stored in MongoDB - upload once, search forever
- **Vectors** stored in Qdrant Cloud - survives restarts
- **Sessions** preserved - continue previous conversations

### ✅ Multi-Agent System
The AI automatically selects the best search strategy:
- **Semantic Search** - for concepts (symptoms, conditions, treatment plans)
- **Keyword Search (BM25)** - for exact terms (HbA1c, metformin, ICD codes)
- **Hybrid Search (RRF)** - combines both for complex queries

### ✅ Chat Memory
Sessions remember previous questions, enabling follow-up queries like:
- "What medications?" → "Are there any interactions?"
- Context is preserved across the conversation

### ✅ Production-Ready
- Document deduplication (won't re-index same file)
- Proper error handling
- Clean source citations
- Session management

## Prerequisites

1. **MongoDB** - Local or Atlas
   ```bash
   # macOS
   brew tap mongodb/brew
   brew install mongodb-community
   brew services start mongodb-community
   
   # Or use MongoDB Atlas (free tier): https://cloud.mongodb.com
   ```

2. **OpenAI API Key** - https://platform.openai.com/api-keys

3. **Qdrant Cloud** - Already configured

## Setup

### 1. Backend Setup

```bash
cd backend

# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Configure Environment

Edit `backend/.env`:
```env
OPENAI_API_KEY=sk-your-key-here
LLM_MODEL=gpt-4o-mini
EMBEDDING_MODEL=text-embedding-3-small

# Qdrant Cloud (already configured)
QDRANT_URL=https://65e86e65-d536-4fb1-a52e-5cbc09d9f40d.us-east4-0.gcp.cloud.qdrant.io:6333
QDRANT_API_KEY=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJhY2Nlc3MiOiJtIn0.TERquqAx5wRwEYoWMM9_ofUpdauBqXRlOt8k3K51I30

# MongoDB (local default, or Atlas connection string)
MONGODB_URI=mongodb://localhost:27017
```

### 3. Frontend Setup

```bash
cd frontend
npm install
```

### 4. Run

**Terminal 1 - Backend:**
```bash
cd backend
source venv/bin/activate
python main.py
```

**Terminal 2 - Frontend:**
```bash
cd frontend
npm run dev
```

Open **http://localhost:3000**

## Usage

### Document Library
1. **Upload PDFs** - Documents are stored permanently
2. **Select a document** - View its sessions
3. **Start New Session** or **Continue Previous**

### Chat Interface
1. Ask questions about the clinical notes
2. AI selects appropriate search tool automatically
3. Sources are cited with page numbers
4. Chat history is preserved per session

## API Endpoints

### Documents
- `GET /documents` - List all documents
- `POST /upload` - Upload PDF files
- `GET /documents/{id}` - Get document details
- `DELETE /documents/{id}` - Delete document and vectors
- `DELETE /documents` - Clear all (caution!)

### Sessions
- `POST /sessions?document_id=xxx` - Create new session
- `GET /sessions/{id}` - Get session with chat history
- `GET /documents/{id}/sessions` - List sessions for document
- `DELETE /sessions/{id}` - Delete session

### Query
- `POST /query` - Search with optional session context
  ```json
  {
    "question": "What medications is the patient taking?",
    "session_id": "optional-for-memory",
    "document_id": "optional-to-filter",
    "use_agent": true,
    "use_hybrid": true
  }
  ```

### Health
- `GET /health` - System status and stats

## How Multi-Agent Works

1. **User asks question**
2. **Agent analyzes** and picks best tool(s):
   - Conceptual question → `semantic_search`
   - Specific values → `keyword_search`
   - Complex query → `hybrid_search`
3. **Agent may chain** multiple searches
4. **Agent synthesizes** final answer with citations

Example:
```
User: "What is the patient's HbA1c and what does it indicate?"

Agent thinking:
- "HbA1c" is a specific value → keyword_search
- "what does it indicate" needs context → semantic_search
- Using both tools...

→ Executes keyword_search("HbA1c")
→ Executes semantic_search("diabetes control indication")
→ Synthesizes answer from both results
```

## Cost Estimate

For 100 queries:
- **Embeddings**: ~$0.02
- **GPT-4o-mini**: ~$0.20
- **MongoDB**: Free (local) or ~$0/month (Atlas free tier)
- **Qdrant Cloud**: Free tier
- **Total**: < $1.00

## Future Enhancements

- [ ] Nanonets OCR for scanned documents
- [ ] MongoDB Atlas integration
- [ ] Multi-user support with authentication
- [ ] Document comparison agent
- [ ] Export chat sessions as PDF

## Troubleshooting

**MongoDB connection failed:**
```bash
# Check if MongoDB is running
brew services list
# Start if stopped
brew services start mongodb-community
```

**Documents not persisting:**
- Check MongoDB connection string in `.env`
- Verify Qdrant Cloud credentials

**Agent not using tools:**
- Increase `max_iterations` in agent config
- Check OpenAI API quota

## License

MIT
