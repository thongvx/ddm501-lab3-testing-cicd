"""
FastAPI application for Movie Rating Prediction.
"""

import logging
from contextlib import asynccontextmanager
from typing import Any, AsyncIterator, Dict, Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from app.config import API_DESCRIPTION, API_TITLE, API_VERSION, MODEL_VERSION
from app.model import MovieRatingModel
from app.schemas import (
    BatchPredictionRequest,
    BatchPredictionResponse,
    HealthResponse,
    PredictionRequest,
    PredictionResponse,
)

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global model instance (set at startup; None if loading failed)
model: Optional[MovieRatingModel] = None


def load_model() -> None:
    """Load the model into the global slot; keep the API up (unhealthy) if it fails."""
    global model
    try:
        model = MovieRatingModel()
        logger.info("Model loaded successfully at startup")
    except Exception as e:
        model = None
        logger.error("Failed to load model: %s", e)


@asynccontextmanager
async def lifespan(_: FastAPI) -> AsyncIterator[None]:
    """Load the model when the application starts (replaces deprecated on_event)."""
    load_model()
    yield


# Initialize FastAPI app
app = FastAPI(
    title=API_TITLE,
    description=API_DESCRIPTION,
    version=API_VERSION,
    lifespan=lifespan,
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def _require_model() -> MovieRatingModel:
    """Return the loaded model or fail with 503 Service Unavailable."""
    if model is None or not model.is_loaded():
        raise HTTPException(status_code=503, detail="Model not loaded")
    return model


@app.get("/", tags=["Info"])
async def root() -> Dict[str, str]:
    """Root endpoint with API information."""
    return {
        "name": API_TITLE,
        "version": API_VERSION,
        "description": API_DESCRIPTION,
        "docs": "/docs",
        "health": "/health",
    }


@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check() -> HealthResponse:
    """
    Health check endpoint.

    Returns the health status of the API and whether the model is loaded.
    """
    loaded = model is not None and model.is_loaded()
    return HealthResponse(status="healthy" if loaded else "unhealthy", model_loaded=loaded)


@app.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
async def predict(request: PredictionRequest) -> PredictionResponse:
    """
    Predict movie rating for a user.

    Args:
        request: PredictionRequest with user_id and movie_id

    Returns:
        PredictionResponse with predicted rating
    """
    current = _require_model()
    try:
        rating = current.predict(request.user_id, request.movie_id)
    except Exception as e:
        logger.error("Prediction error: %s", e)
        raise HTTPException(status_code=500, detail="Prediction failed") from e
    return PredictionResponse(
        user_id=request.user_id,
        movie_id=request.movie_id,
        predicted_rating=rating,
        model_version=MODEL_VERSION,
    )


@app.post("/predict/batch", response_model=BatchPredictionResponse, tags=["Prediction"])
async def predict_batch(request: BatchPredictionRequest) -> BatchPredictionResponse:
    """
    Predict movie ratings for multiple user-movie pairs.

    Args:
        request: BatchPredictionRequest with list of predictions

    Returns:
        BatchPredictionResponse with all predicted ratings
    """
    current = _require_model()
    try:
        results = [
            PredictionResponse(
                user_id=item.user_id,
                movie_id=item.movie_id,
                predicted_rating=current.predict(item.user_id, item.movie_id),
                model_version=MODEL_VERSION,
            )
            for item in request.predictions
        ]
    except Exception as e:
        logger.error("Batch prediction error: %s", e)
        raise HTTPException(status_code=500, detail="Batch prediction failed") from e
    return BatchPredictionResponse(predictions=results, total_count=len(results))


@app.get("/model/info", tags=["Info"])
async def model_info() -> Dict[str, Any]:
    """Get information about the loaded model."""
    return {
        "model_version": MODEL_VERSION,
        "model_type": "SVD (Collaborative Filtering)",
        "is_loaded": model is not None and model.is_loaded(),
    }


if __name__ == "__main__":  # pragma: no cover
    import uvicorn

    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)  # nosec B104
