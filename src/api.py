"""
FastAPI Application for SHL Recommender System

Provides REST API endpoints for assessment recommendations.

Endpoints:
    GET  /health - Health check endpoint
    POST /recommend - Get assessment recommendations for a query
    GET  /stats - System statistics (bonus)
    GET  / - Root endpoint with API information
"""

from collections import defaultdict, deque
from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, Header, HTTPException, Request, Response, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from typing import List, Dict, Optional
import os
import time
from datetime import datetime

from src.logging_config import get_logger
from workflow_graph import get_orchestrator

logger = get_logger(__name__)

# ============================================================================
# Pydantic Models for Request/Response Validation
# ============================================================================

class RecommendationRequest(BaseModel):
    """Request model for recommendation endpoint."""
    query: str = Field(
        ...,
        description="Job description or hiring query",
        min_length=5,
        max_length=5000,
        example="I need a Java developer with strong communication skills"
    )
    top_k: Optional[int] = Field(
        default=10,
        description="Number of recommendations to return",
        ge=1,
        le=50
    )
    test_type_ratio: Optional[Dict[str, float]] = Field(
        default=None,
        description="Custom test type ratio (e.g., {'K': 0.6, 'P': 0.4} for 60% Knowledge, 40% Personality). If not provided, ratio is auto-detected from query.",
        example={"K": 0.5, "P": 0.5}
    )


class Assessment(BaseModel):
    """Individual assessment recommendation - matches assignment format."""
    url: str = Field(..., description="Valid URL to the assessment resource")
    name: str = Field(..., description="Name of the assessment")
    adaptive_support: str = Field(default="No", description='Either "Yes" or "No" indicating if the assessment supports adaptive testing')
    description: str = Field(..., description="Detailed description of the assessment")
    duration: int = Field(..., description="Duration of the assessment in minutes (0 when the catalogue does not publish one)")
    remote_support: str = Field(default="Yes", description='Either "Yes" or "No" indicating if the assessment can be taken remotely')
    test_type: List[str] = Field(..., description="Categories or types of the assessment")


class RecommendationResponse(BaseModel):
    """Response model for recommendation endpoint - matches assignment format."""
    recommended_assessments: List[Assessment] = Field(..., description="List of recommended assessments")


class HealthResponse(BaseModel):
    """Response model for health check endpoint."""
    status: str = Field(..., description="Service health status")
    timestamp: str = Field(..., description="Current timestamp")
    version: str = Field(..., description="API version")
    uptime_seconds: float = Field(..., description="Service uptime in seconds")


class StatsResponse(BaseModel):
    """Response model for statistics endpoint."""
    total_requests: int = Field(..., description="Total requests processed")
    total_assessments: int = Field(..., description="Total assessments in database")
    avg_processing_time_ms: float = Field(..., description="Average processing time")
    uptime_seconds: float = Field(..., description="Service uptime")
    timestamp: str = Field(..., description="Current timestamp")


# ============================================================================
# FastAPI Application Setup
# ============================================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load the workflow orchestrator on startup."""
    logger.info("Starting SHL Recommender API")
    try:
        app_state.orchestrator = get_orchestrator()
        logger.info("Loaded %s assessments; API ready", len(app_state.orchestrator.retriever.df))
    except Exception as e:
        logger.error("Failed to initialize system: %s", e, exc_info=True)
        raise
    yield
    logger.info("Shutting down SHL Recommender API")


app = FastAPI(
    lifespan=lifespan,
    title="SHL Recommender API",
    description="Assessment recommendation system using advanced workflow orchestration",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

_cors_origins = os.getenv("API_ALLOW_ORIGINS", "*")
if _cors_origins.strip() == "*":
    _allowed_origins = ["*"]
else:
    _allowed_origins = [origin.strip() for origin in _cors_origins.split(",") if origin.strip()]

# CORS middleware for cross-origin requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=_allowed_origins,
    # Browsers reject credentials with a wildcard origin, so only allow them for explicit origins
    allow_credentials=_allowed_origins != ["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================================================================
# Application State & Statistics
# ============================================================================

class AppState:
    """Global application state for statistics tracking."""
    def __init__(self):
        self.start_time = time.time()
        self.total_requests = 0
        self.processing_times = []
        self.orchestrator = None
    
    def record_request(self, processing_time_ms: float):
        """Record a completed request."""
        self.total_requests += 1
        self.processing_times.append(processing_time_ms)
        # Keep only last 100 processing times
        if len(self.processing_times) > 100:
            self.processing_times = self.processing_times[-100:]
    
    @property
    def avg_processing_time(self) -> float:
        """Calculate average processing time."""
        if not self.processing_times:
            return 0.0
        return sum(self.processing_times) / len(self.processing_times)
    
    @property
    def uptime(self) -> float:
        """Calculate service uptime in seconds."""
        return time.time() - self.start_time


# Global app state
app_state = AppState()


# ============================================================================
# Access Control
# ============================================================================

API_KEY = os.getenv("API_KEY")  # when set, /recommend requires a matching X-API-Key header
RATE_LIMIT_PER_MINUTE = int(os.getenv("RATE_LIMIT_PER_MINUTE", "30"))
_request_log: Dict[str, deque] = defaultdict(deque)


def require_api_key(x_api_key: Optional[str] = Header(default=None)):
    if API_KEY and x_api_key != API_KEY:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid or missing API key")


def rate_limit(request: Request):
    """Sliding one-minute window per client IP (in-memory, per process)."""
    if RATE_LIMIT_PER_MINUTE <= 0:
        return
    client = request.client.host if request.client else "unknown"
    now = time.time()
    window = _request_log[client]
    while window and now - window[0] > 60:
        window.popleft()
    if len(window) >= RATE_LIMIT_PER_MINUTE:
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail="Rate limit exceeded",
            headers={"Retry-After": str(int(60 - (now - window[0])) + 1)},
        )
    window.append(now)


# ============================================================================
# API Endpoints
# ============================================================================

@app.get("/", tags=["Info"])
async def root():
    """
    Root endpoint with API information.
    
    Returns:
        Basic API information and available endpoints
    """
    return {
        "name": "SHL Recommender API",
        "version": "1.0.0",
        "description": "Assessment recommendation system with intelligent matching",
        "endpoints": {
            "health": "/health",
            "recommend": "/recommend (POST)",
            "stats": "/stats",
            "docs": "/docs"
        },
        "status": "operational",
        "uptime_seconds": app_state.uptime
    }


@app.get(
    "/health",
    response_model=HealthResponse,
    tags=["Health"],
    summary="Health check endpoint",
    description="Check if the API service is healthy and operational"
)
async def health_check():
    """
    Health check endpoint - REQUIRED FOR SUBMISSION.
    
    Returns:
        HealthResponse with status "healthy" if service is operational
    """
    return HealthResponse(
        status="healthy",
        timestamp=datetime.now().isoformat(),
        version="1.0.0",
        uptime_seconds=app_state.uptime
    )


@app.post(
    "/recommend",
    response_model=RecommendationResponse,
    tags=["Recommendations"],
    summary="Get assessment recommendations",
    description="Provide a job description or query to get personalized assessment recommendations",
    dependencies=[Depends(require_api_key), Depends(rate_limit)],
)
async def recommend(request: RecommendationRequest, response: Response):
    """
    Assessment recommendation endpoint - REQUIRED FOR SUBMISSION.
    
    This endpoint processes a job query through the LangGraph workflow:
    1. Query enhancement (extract role, skills, preferences)
    2. Hybrid retrieval (FAISS + BM25 + specificity scoring)
    3. LLM reranking (Gemini with thinking mode)
    4. Test type balancing
    
    Args:
        request: RecommendationRequest with query and optional top_k
        
    Returns:
        RecommendationResponse with recommended assessments
        
    Raises:
        HTTPException: If processing fails
    """
    start_time = time.time()
    
    logger.info("=" * 70)
    logger.info("Received recommendation request")
    logger.info("Query: %s...", request.query[:100])
    logger.info("Requested top_k: %s", request.top_k)
    
    if app_state.orchestrator is None:
        logger.error("Orchestrator not initialized")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Service not fully initialized. Please try again."
        )
    
    try:
        # Run workflow with optional custom test type ratio
        result = await app_state.orchestrator.run(
            query=request.query,
            custom_test_type_ratio=request.test_type_ratio
        )
        
        if not result.get("results"):
            logger.error("Workflow returned no results: %s", result.get("error"))
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Recommendations are temporarily unavailable. Please try again."
            )
        
        # Fallbacks taken (embedding, enhancer, reranker) are reported in headers so the
        # response body keeps the required format
        degraded_reasons = result.get("degraded_reasons", [])
        response.headers["X-Degraded"] = "true" if degraded_reasons else "false"
        if degraded_reasons:
            response.headers["X-Degraded-Reasons"] = ",".join(degraded_reasons)
            logger.warning("Degraded response: %s", degraded_reasons)
        
        # Extract results and confidence
        recommendations = result.get("results", [])[:request.top_k]
        confidence = result.get("confidence", {"retrieval": 0.0, "reranking": 0.0})
        
        # Convert to Assessment objects matching assignment format
        assessment_list = []
        for rec in recommendations:
            # Get test_types and ensure it's a proper list (fallback to test_type if needed)
            test_types = rec.get("test_types", rec.get("test_type", []))
            
            # Handle various formats
            if isinstance(test_types, str):
                # If it's a string representation of a list, parse it
                import ast
                try:
                    test_types = ast.literal_eval(test_types) if test_types.startswith('[') else [test_types]
                except (ValueError, SyntaxError):
                    test_types = [test_types]
            elif not isinstance(test_types, list):
                test_types = [str(test_types)]
            
            # Get adaptive_support and remote_support as Yes/No strings
            adaptive_support = "Yes" if rec.get("adaptive_irt", False) else "No"
            remote_support = "Yes" if rec.get("remote_testing", True) else "No"
            
            assessment_list.append(Assessment(
                url=rec.get("url", ""),
                name=rec.get("name", "Unknown"),
                adaptive_support=adaptive_support,
                description=rec.get("description", ""),
                duration=rec.get("duration") or 0,  # 0 = duration not published in the catalogue
                remote_support=remote_support,
                test_type=test_types
            ))
        
        # Calculate processing time
        processing_time_ms = (time.time() - start_time) * 1000
        
        # Record statistics
        app_state.record_request(processing_time_ms)
        
        logger.info("Returned %s recommendations", len(assessment_list))
        logger.info("Processing time: %.2fms", processing_time_ms)
        logger.info(
            "Confidence: retrieval=%.2f, reranking=%.2f",
            confidence.get("retrieval", 0.0),
            confidence.get("reranking", 0.0),
        )
        logger.info("=" * 70)
        
        return RecommendationResponse(
            recommended_assessments=assessment_list
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Recommendation failed: %s", e, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to generate recommendations"
        )


@app.get(
    "/stats",
    response_model=StatsResponse,
    tags=["Statistics"],
    summary="Get system statistics",
    description="Get statistics about API usage and performance"
)
async def get_stats():
    """
    System statistics endpoint (bonus feature).
    
    Returns:
        StatsResponse with usage statistics
    """
    try:
        # Get total assessments count
        total_assessments = 0
        if app_state.orchestrator and app_state.orchestrator.retriever:
            total_assessments = len(app_state.orchestrator.retriever.df)
        
        return StatsResponse(
            total_requests=app_state.total_requests,
            total_assessments=total_assessments,
            avg_processing_time_ms=app_state.avg_processing_time,
            uptime_seconds=app_state.uptime,
            timestamp=datetime.now().isoformat()
        )
    except Exception as e:
        logger.error(f"Failed to get statistics: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve statistics"
        )


# ============================================================================
# Error Handlers
# ============================================================================

@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """Global exception handler for unhandled errors."""
    logger.error(f"Unhandled exception: {exc}", exc_info=True)
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={"detail": "An unexpected error occurred"}
    )


if __name__ == "__main__":
    # This won't be called when using uvicorn, but useful for debugging
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
