# Deployment Guide - Clinical RAG System

**Status:** ✅ Production-Ready  
**Last Updated:** December 16, 2025

---

## Quick Start (Local Development)

### Prerequisites
- Python 3.11+
- Node.js 18+
- MongoDB (local or Atlas)
- API Keys (OpenAI, Qdrant, Nanonets)

### 1. Backend Setup

```bash
cd backend
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

Create `.env` file:
```bash
OPENAI_API_KEY=sk-...
LLM_MODEL=gpt-4o-mini
EMBEDDING_MODEL=text-embedding-3-large

QDRANT_URL=https://your-cluster.gcp.cloud.qdrant.io:6333
QDRANT_API_KEY=your-qdrant-key

MONGODB_URI=mongodb://localhost:27017  # Or Atlas URI
MONGODB_DB_NAME=clinical_notes_db

NANONETS_API_KEY=your-nanonets-key

CHUNK_SIZE=2048
CHUNK_OVERLAP=400
COLLECTION_NAME=clinical_notes
```

Start server:
```bash
python main.py
# Server runs on http://localhost:8000
```

### 2. Frontend Setup

```bash
cd frontend
npm install
npm run dev
# Frontend runs on http://localhost:5173
```

---

## Production Deployment

### Option 1: Railway (Recommended for MVP)

**Backend:**
1. Push code to GitHub
2. Connect Railway to repo
3. Add environment variables
4. Deploy automatically

**Frontend:**
1. Build: `npm run build`
2. Deploy `dist/` to Vercel/Netlify

**Cost:** ~$5/month (Railway starter)

---

### Option 2: AWS Lambda + S3

**Backend (Lambda):**
```bash
# Install serverless framework
npm install -g serverless

# Deploy
serverless deploy
```

**Frontend (S3 + CloudFront):**
```bash
npm run build
aws s3 sync dist/ s3://your-bucket/
```

**Cost:** Pay-as-you-go (~$10/month for 10K queries)

---

### Option 3: Docker Compose

```yaml
# docker-compose.yml
version: '3.8'

services:
  backend:
    build: ./backend
    ports:
      - "8000:8000"
    env_file:
      - ./backend/.env
    depends_on:
      - mongodb

  frontend:
    build: ./frontend
    ports:
      - "3000:3000"
    depends_on:
      - backend

  mongodb:
    image: mongo:7.0
    ports:
      - "27017:27017"
    volumes:
      - mongodb_data:/data/db

volumes:
  mongodb_data:
```

Deploy:
```bash
docker-compose up -d
```

---

## MongoDB Setup

### Option A: Local MongoDB

```bash
# macOS
brew install mongodb-community
brew services start mongodb-community

# Linux
sudo apt install mongodb
sudo systemctl start mongodb
```

### Option B: MongoDB Atlas (Recommended)

1. Create free cluster at [mongodb.com/atlas](https://mongodb.com/atlas)
2. Get connection URI
3. Update `.env`:
   ```
   MONGODB_URI=mongodb+srv://user:pass@cluster.mongodb.net/clinical_notes_db
   ```

---

## Qdrant Cloud Setup

1. Sign up at [cloud.qdrant.io](https://cloud.qdrant.io)
2. Create free cluster (1GB)
3. Get URL and API key
4. Update `.env`

**Note:** Free tier is sufficient for <10K documents

---

## Environment Variables Checklist

| Variable | Required | Example | Notes |
|----------|----------|---------|-------|
| `OPENAI_API_KEY` | ✅ Yes | `sk-...` | From platform.openai.com |
| `QDRANT_URL` | ✅ Yes | `https://...` | From Qdrant Cloud |
| `QDRANT_API_KEY` | ✅ Yes | `...` | From Qdrant Cloud |
| `MONGODB_URI` | ✅ Yes | `mongodb://...` | Local or Atlas |
| `NANONETS_API_KEY` | ⚠️ Optional | `...` | Only for scanned PDFs |
| `EMBEDDING_MODEL` | ✅ Yes | `text-embedding-3-large` | Or `text-embedding-3-small` |
| `LLM_MODEL` | ✅ Yes | `gpt-4o-mini` | Or `gpt-4` |
| `CHUNK_SIZE` | No | `2048` | Default works well |
| `CHUNK_OVERLAP` | No | `400` | Default works well |

---

## Health Check

After deployment, verify:

```bash
# Backend health
curl http://localhost:8000/health

# Expected response:
{
  "status": "healthy",
  "model": "gpt-4o-mini",
  "embedding_model": "text-embedding-3-large",
  "stats": {
    "vectors_count": 0,
    "documents_in_memory": 0
  }
}
```

---

## Monitoring (Production)

### Add Prometheus + Grafana

1. Install Prometheus:
```yaml
# prometheus.yml
scrape_configs:
  - job_name: 'clinical-rag'
    static_configs:
      - targets: ['localhost:8000']
```

2. Add metrics endpoint to FastAPI:
```python
from prometheus_client import Counter, Histogram, generate_latest

query_counter = Counter('queries_total', 'Total queries')
query_latency = Histogram('query_latency_seconds', 'Query latency')

@app.get("/metrics")
def metrics():
    return Response(generate_latest(), media_type="text/plain")
```

---

## Backup & Recovery

### MongoDB Backup
```bash
# Backup
mongodump --uri="mongodb://..." --out=./backup

# Restore
mongorestore --uri="mongodb://..." ./backup
```

### Qdrant Backup
```python
# Snapshot via API
qdrant_client.create_snapshot(collection_name="clinical_notes")
```

---

## Scaling Considerations

| Load Level | Configuration | Monthly Cost |
|------------|--------------|--------------|
| **Dev/Testing** | Local MongoDB, Qdrant Free | $0 |
| **MVP (<1K docs)** | MongoDB Atlas Free, Qdrant Free | $0 |
| **Small Clinic** | MongoDB Shared, Qdrant 1GB | $57 + $0 |
| **Medium Clinic** | MongoDB Dedicated, Qdrant 2GB | $57 + $95 |
| **Large Hospital** | Sharded MongoDB, Qdrant 8GB | $250 + $380 |

---

## Troubleshooting

### Error: "OPENAI_API_KEY not set"
**Fix:** Check `.env` file exists and has correct key

### Error: "Qdrant connection failed"
**Fix:** Verify URL and API key, check firewall

### Error: "MongoDB connection timeout"
**Fix:** Check MongoDB is running, verify URI format

### Error: "Nanonets OCR failed"
**Fix:** Check API key, or set `NANONETS_API_KEY=""` to skip OCR

### Slow queries (>5s)
**Fix:** 
- Check Qdrant latency
- Reduce `CHUNK_SIZE` to 1024
- Add Qdrant payload indexes

---

## Security Checklist

- [ ] API keys in `.env` (never commit)
- [ ] MongoDB authentication enabled
- [ ] Qdrant API key rotation (monthly)
- [ ] HTTPS/TLS for all endpoints
- [ ] Rate limiting (nginx/CloudFlare)
- [ ] Input validation (Pydantic models) ✅
- [ ] CORS restricted to frontend domain
- [ ] JWT authentication (future)
- [ ] Audit logging (future)

---

## Performance Tuning

### For faster queries:
```python
# config.py
CHUNK_SIZE = 1024  # Smaller chunks
CHUNK_OVERLAP = 200
```

### For better accuracy:
```python
# config.py
CHUNK_SIZE = 2048  # Current (balanced)
CHUNK_OVERLAP = 400
```

### For cost savings:
```python
EMBEDDING_MODEL = "text-embedding-3-small"  # 1536-dim, cheaper
```

---

## Support & Resources

- **Documentation:** See `ARCHITECTURE.md`
- **Issues:** Check logs in `backend/logs/`
- **API Docs:** http://localhost:8000/docs
- **Community:** [Your Discord/Slack]

---

## License

MIT License - See LICENSE file

---

**Deployment Status:** ✅ Ready for Production  
**Tested On:** macOS 14, Ubuntu 22.04, Windows 11  
**Last Verified:** December 16, 2025

