# SHL Assessment Recommender System

> **Intelligent Assessment Recommendation System** - Production-ready application that matches job descriptions to relevant SHL assessments using advanced workflow orchestration, hybrid retrieval, and intelligent reranking.

[![FastAPI](https://img.shields.io/badge/FastAPI-0.104.1-009688?logo=fastapi)](https://fastapi.tiangolo.com/)
[![LangGraph](https://img.shields.io/badge/LangGraph-0.2.52-4F46E5)](https://langchain-ai.github.io/langgraph/)
[![Gemini](https://img.shields.io/badge/Gemini-2.0%20Flash-4285F4?logo=google)](https://ai.google.dev/)
[![Python](https://img.shields.io/badge/Python-3.12-3776AB?logo=python)](https://www.python.org/)

---

## Quick Start

### 1. Setup Environment

```bash
# Clone the repository
git clone <repository-url>
cd shl-recommender

# Create virtual environment (recommended)
python -m venv venv
.\venv\Scripts\Activate.ps1  # Windows PowerShell
# or
source venv/bin/activate      # Linux/Mac

# Install dependencies
pip install -r requirements.txt
```

### 2. Configure API Key

```bash
# Copy environment template
cp .env.example .env

# Edit .env and add your Gemini API key
# GEMINI_API_KEY=your_api_key_here
```

Get your API key from: https://makersuite.google.com/app/apikey

### 3. Run the Application

**Option A: Streamlit Web UI (Recommended)**
```bash
python run_ui.py
```
Opens at: http://localhost:8501

**Option B: FastAPI REST API**
```bash
python run_api.py
```
Opens at: http://localhost:8000
- API Docs: http://localhost:8000/docs
- Health Check: http://localhost:8000/health

---

## System Architecture

```
Query → Extract → RAG → Filter → Results
         ↓         ↓      ↓
    (enhance) (retrieve) (rerank + balance)
```

### Pipeline Stages

1. **Query Enhancement** - Extracts role, skills, seniority using Gemini
2. **Hybrid Retrieval** - Combines semantic (30%), BM25 (20%), specificity (40%), quality (10%)
3. **Test Type Balancing** - Auto-detects K/P ratio (Knowledge vs Personality)
4. **Intelligent Reranking** - Gemini 3.8 Flash (temperature 0)
5. **Post-Processing** - Deduplication, soft duration scoring, domain validation

---

## Key Features

### LangGraph Workflow Orchestration
- State machine with confidence checks
- Automatic retry with broader retrieval on low confidence
- Graceful error handling and fallbacks

### Hybrid Retrieval System
- **FAISS Vector Search** (30%) - Semantic similarity using Gemini embeddings
- **BM25 Keyword Matching** (20%) - Exact term matching
- **Specificity Scoring** (40%) - Domain-specific boosts for finance, analyst, consultant roles
- **Quality Filtering** (10%) - Assessment quality indicators

### Intelligent Reranking
- Gemini 3.8 Flash at temperature 0
- Multi-criteria evaluation: role alignment, skill depth, assessment type
- Functional skill prioritization for manager roles
- Domain-specific matching patterns

### Test Type Balancing
- Auto K/P ratio detection from query intent
- 50/50 balance for mixed queries (technical + collaboration)
- 80/20 for pure technical roles
- 30/70 for behavioral roles
- Custom ratios via API parameter

### Soft Duration Scoring
- Duration preferences (not hard filters)
- All results kept, just reordered by preference
- 5% weight for duration matching
- No more empty result sets

---

## Project Structure

```
shl-recommender/
├── src/                        # Core application modules
│   ├── __init__.py
│   ├── api.py                 # FastAPI REST endpoints
│   ├── ui_app.py              # Streamlit web interface
│   ├── query_enhancer.py      # Query processing & skill expansion
│   ├── retreiver.py           # Hybrid retrieval (FAISS + BM25)
│   ├── gemini_reranker.py     # Intelligent reranking
│   ├── test_type_balancer.py  # K/P ratio balancing
│   ├── improvements.py        # Post-processing pipeline
│   └── logging_config.py      # Logging configuration
│
├── outputs/                    # Pre-computed indices (REQUIRED)
│   ├── embeddings_gemini_001.npy    # 768-dim embeddings
│   ├── faiss_gemini_001.index       # FAISS vector index
│   ├── bm25_index.pkl               # BM25 keyword index
│   └── assessments_processed.csv    # 506 assessments
│
├── data/                       # Metadata and training data
│   ├── train.csv              # Training queries
│   ├── test.csv               # Test queries
│   ├── assessments_metadata.json
│   ├── skill_index.json
│   └── role_templates.json
│
├── configs/                    # Configuration files
│   └── recommender_config.json
│
├── .env                        # API keys (REQUIRED)
├── .env.example               # Environment template
├── requirements.txt           # Python dependencies
├── run_api.py                 # API launcher
├── run_ui.py                  # UI launcher
├── workflow_graph.py          # LangGraph orchestration
└── README.md                  # This file
```

---

## API Documentation

### Base URL
```
http://localhost:8000
```

### Endpoints

#### 1. Health Check
```bash
GET /health
```

**Response:**
```json
{
  "status": "healthy",
  "timestamp": "2025-11-11T19:00:00.000000",
  "version": "1.0.0",
  "uptime_seconds": 123.45
}
```

#### 2. Get Recommendations
```bash
POST /recommend
```

**Request:**
```json
{
  "query": "Java developer with strong communication skills",
  "top_k": 10,
  "test_type_ratio": {"K": 0.6, "P": 0.4}  // optional
}
```

**Response:**
```json
{
  "recommended_assessments": [
    {
      "url": "https://www.shl.com/solutions/products/...",
      "name": "Core Java (Advanced Level)",
      "adaptive_support": "No",
      "description": "Assesses advanced Java programming...",
      "duration": 30,
      "remote_support": "Yes",
      "test_type": ["Knowledge"]
    }
  ]
}
```

#### 3. System Statistics
```bash
GET /stats
```

**Response:**
```json
{
  "total_requests": 42,
  "total_assessments": 506,
  "avg_processing_time_ms": 25340.5,
  "uptime_seconds": 3600.0
}
```

### Example cURL Commands

```bash
# Health check
curl http://localhost:8000/health

# Get recommendations
curl -X POST "http://localhost:8000/recommend" \
  -H "Content-Type: application/json" \
  -d '{"query": "Python developer with SQL skills", "top_k": 10}'

# Custom test type ratio
curl -X POST "http://localhost:8000/recommend" \
  -H "Content-Type: application/json" \
  -d '{"query": "Product Manager", "top_k": 10, "test_type_ratio": {"K": 0.3, "P": 0.7}}'
```

---

## Streamlit Web UI Features

### Main Interface
- **Job Description Input** - Paste job postings or write queries
- **Quick Templates** - Pre-filled examples (Software Engineer, Data Scientist, etc.)
- **Real-time Results** - Get recommendations instantly

### Results Display
- Assessment name and description
- Duration and test type
- Direct links to SHL catalog
- Relevance scoring

### Advanced Features
- Adjustable number of results (5-20)
- Custom K/P ratio configuration
- Export results to CSV
- Debug mode for troubleshooting

---

## Configuration

### Environment Variables (.env)

```env
# Required: Google Gemini API Key
GEMINI_API_KEY=your_api_key_here

# Optional: model overrides
GEMINI_MODEL=gemini-3.8-flash
GEMINI_EMBEDDING_MODEL=gemini-embedding-001
```

### Customization

Edit `src/retreiver.py` to adjust:
- **Hybrid weights**: Modify semantic/BM25/specificity ratios
- **Domain boosts**: Adjust finance/analyst/consultant boost values
- **Top-K results**: Change default retrieval count

Edit `src/query_enhancer.py` to:
- Add new skill mappings
- Customize role templates
- Adjust skill expansion logic

---

## Evaluation

`scripts/evaluate.py` measures Mean Recall@10 and MAP@10 on the 10 labelled queries in `data/train.csv` (65 labelled URLs), stage by stage:

```bash
python scripts/evaluate.py                       # bm25, hybrid and full workflow
python scripts/evaluate.py --stages bm25 hybrid  # no LLM calls
```

| Configuration | Stage | Recall@10 | MAP@10 |
|---|---|---|---|
| Before fixes (retired Gemini models, all LLM/embedding calls failing) | full workflow | 0.303 | 0.137 |
| Current models, before FAISS fix | hybrid retriever | 0.228 | 0.164 |
| Current models, before FAISS fix | full workflow | 0.478 | 0.297 |
| **After fixes** | BM25 only | 0.154 | 0.108 |
| **After fixes** | hybrid retriever | **0.283** | **0.181** |

Full-workflow numbers after the fixes are still to be measured (the free-tier key allows 20 LLM requests per day).
These are training queries that some retrieval rules were tuned on, so they overstate held-out quality.

### Rebuilding the vector index

The FAISS index must be built with the same embedding model used for queries. After changing `GEMINI_EMBEDDING_MODEL`:

```bash
python scripts/build_index.py
```

This writes `outputs/faiss_gemini_001.index` and records the model in `outputs/metadata.json`.

### Tests

```bash
pytest tests
```

---

## Troubleshooting

### Module not found errors

```bash
# Ensure you're in project root
cd shl-recommender

# Activate virtual environment
.\venv\Scripts\Activate.ps1  # Windows
source venv/bin/activate      # Linux/Mac

# Reinstall dependencies
pip install -r requirements.txt
```

### API connection errors in UI

```bash
# Ensure API is running first (Terminal 1)
python run_api.py

# Then start UI (Terminal 2)
python run_ui.py
```

### Gemini API errors

```bash
# Check your .env file has valid API key
cat .env  # Linux/Mac
type .env  # Windows

# Test API key
python -c "import os; from dotenv import load_dotenv; load_dotenv(); print(os.getenv('GEMINI_API_KEY'))"
```

### Missing data files

If `outputs/` folder is missing required files:
- Ensure you have the complete repository
- Check that embeddings and indices are present
- Contact maintainer for pre-computed files

---

## Tech Stack

### Core Technologies
- **ML/Search**: Google Gemini embeddings, FAISS vector search, BM25 ranking
- **Backend**: FastAPI, LangGraph, Pydantic
- **Frontend**: Streamlit
- **Data**: Pandas, NumPy

### Key Dependencies
```
google-generativeai>=0.3.0  # Gemini API
faiss-cpu>=1.7.4            # Vector search
rank-bm25>=0.2.2            # Keyword matching
fastapi>=0.100.0            # REST API
streamlit>=1.28.0           # Web UI
langgraph>=0.2.0            # Workflow orchestration
```

---

## Key Improvements

### Domain-Specific Boosts
- **Finance**: +15 for financial/accounting, -10 for customer service
- **Analyst**: +8 for cognitive/analytical, -5 for generic solutions
- **Consultant**: +10 for verify/cognitive, +8 for analytical
- **Executive**: +12 for personality/OPQ, +10 for situational

### Soft Duration Scoring
- Duration is now a **preference** (5% weight), not a hard filter
- All results kept and reordered by duration proximity
- ±20 minute tolerance for exact requests
- No more empty result sets

### Test Type Balancing
- Auto-detects technical vs behavioral emphasis
- Ensures balanced recommendations for mixed queries
- Customizable via API parameter

---

## Deployment

### Streamlit Cloud
```bash
# Ensure all files in outputs/ are committed
# Add .env secrets in Streamlit Cloud dashboard
# Deploy from GitHub repository
```

### Docker (Optional)
```bash
# Build image
docker build -t shl-recommender .

# Run container
docker run -p 8501:8501 --env-file .env shl-recommender
```

---

## Support

For questions or issues:
- Check this README for common solutions
- Review `detailed_fixes.md` for implementation details
- Check API documentation at `/docs` endpoint

---

## License

This project is for educational and demonstration purposes.

---

## Author

**Anshaj Shukla**
- GitHub: [@anshajshukla](https://github.com/anshajshukla)

---

**Last Updated**: November 11, 2025  
**Version**: 1.0 (Production Ready)
