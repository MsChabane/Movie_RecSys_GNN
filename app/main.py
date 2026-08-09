import sys
from pathlib import Path
from contextlib import asynccontextmanager
from typing import Dict, Any
from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware

# Ensure src modules are importable
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

# Import your existing RecommenderInference directly!
from src.inference import RecommenderInference
from app.schemas import RecommendRequest, RecommendationResponse


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Loads FAISS index and User Embeddings into memory ONCE during server startup."""
    artifacts_dir = PROJECT_ROOT / "artifacts"
    preprocessed_dir = PROJECT_ROOT / "data" / "preprocessed"

    try:
        print("Loading Recommender Engine into memory...")
        app.state.recommender = RecommenderInference(
            artifacts_dir=str(artifacts_dir),
            preprocessed_dir=str(preprocessed_dir),
        )
        print("Recommender Engine successfully loaded!")
    except Exception as e:
        print(f"Error loading recommendation artifacts: {e}")
        raise e

    yield

    # Cleanup on shutdown
    app.state.recommender = None
    print("Recommender Engine shut down.")


app = FastAPI(
    title="Movie Recommendation System API",
    description="FastAPI server for GraphSAGE + FAISS Movie Recommender",
    version="1.0.0",
    lifespan=lifespan,
)

# Enable CORS for web frontends
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def root():
    return {
        "service": "Movie Recommendation API",
        "status": "online",
        "documentation": "/docs",
    }


@app.get("/health")
def health(request: Request):
    recommender: RecommenderInference = request.app.state.recommender
    if not recommender:
        raise HTTPException(status_code=503, detail="Engine not ready")

    return {
        "status": "healthy",
        "total_users_loaded": len(recommender.user_embeddings),
        "total_movies_indexed": int(recommender.faiss_index.ntotal),
    }


@app.get("/recommend/{user_id}", response_model=RecommendationResponse)
def get_recommendations(
    request: Request,
    user_id: str,
    top_k: int = Query(default=5, ge=1, le=100),
    filter_seen: bool = Query(default=True),
):
    """GET Endpoint: Retrieve top-K movie recommendations for a given user_id string."""
    recommender: RecommenderInference = request.app.state.recommender

    try:
        is_known = user_id in recommender.user_embeddings
        results = recommender.recommend(
            user_str_id=user_id, top_k=top_k, filter_seen=filter_seen
        )

        return {
            "user_id": user_id,
            "is_known_user": is_known,
            "top_k": len(results),
            "recommendations": results,
        }
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Recommendation failed: {str(e)}"
        )


@app.post("/recommend", response_model=RecommendationResponse)
def post_recommendations(payload: RecommendRequest, request: Request):
    """POST Endpoint: Retrieve top-K movie recommendations via JSON payload."""
    recommender: RecommenderInference = request.app.state.recommender

    try:
        is_known = payload.user_id in recommender.user_embeddings
        results = recommender.recommend(
            user_str_id=payload.user_id,
            top_k=payload.top_k,
            filter_seen=payload.filter_seen,
        )

        return {
            "user_id": payload.user_id,
            "is_known_user": is_known,
            "top_k": len(results),
            "recommendations": results,
        }
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Recommendation failed: {str(e)}"
        )
