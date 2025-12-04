from fastapi import FastAPI, HTTPException
from contextlib import asynccontextmanager
import kagglehub

from .models import RecommendationRequest, RecommendationResponse, MovieLensRecommender

model = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Загрузка модели при старте, очистка при завершении."""
    global model
    dataset_path = kagglehub.dataset_download("shubhammehta21/movie-lens-small-latest-dataset")
    model = MovieLensRecommender(dataset_path)
    yield
    model = None


app = FastAPI(
    title="MovieLens Recommender API",
    description="Рекомендации фильмов на основе MovieLens Small dataset",
    version="1.0.0",
    lifespan=lifespan
)


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "ok"}


@app.post("/recommendations", response_model=RecommendationResponse)
async def get_recommendations(request: RecommendationRequest):
    """Получить персонализированные рекомендации фильмов."""
    if model is None:
        raise HTTPException(status_code=503, detail="Модель не загружена")

    try:
        result = model.predict(request)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ошибка предсказания: {str(e)}")
