import math
from pathlib import Path
from typing import Dict, List, Set
import pandas as pd
import numpy as np
from pydantic import BaseModel, Field, validator
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split


class RecommendationRequest(BaseModel):
    """Pydantic модель для валидации запроса рекомендаций."""
    user_id: int = Field(..., ge=1, description="ID пользователя")
    top_n: int = Field(10, ge=1, le=50, description="Количество рекомендаций")
    model_type: str = Field("content", description="Тип модели: 'baseline' или 'content'")

    @validator('model_type')
    def validate_model_type(cls, v):
        if v not in ['baseline', 'content']:
            raise ValueError("model_type должен быть 'baseline' или 'content'")
        return v


class RecommendationResponse(BaseModel):
    """Pydantic модель для ответа."""
    user_id: int
    model_type: str
    top_n: int
    recommendations: List[int]
    count: int


class MovieLensRecommender:
    """Рекомендатор фильмов на основе MovieLens Small."""

    def __init__(self, dataset_path: str):
        self.global_mean = 0.0
        self.user_bias: Dict[int, float] = {}
        self.item_bias: Dict[int, float] = {}
        self.user_profile = pd.DataFrame()
        self.movie_features = pd.DataFrame()
        self.genre_cols: List[str] = []
        self.train_items_by_user: Dict[int, Set[int]] = {}
        self.user_seen: Dict[int, Set[int]] = {}
        self._load_model(dataset_path)

    def _load_model(self, dataset_path: str):
        """Загрузка и обучение модели."""
        print("🔄 Загрузка MovieLens Small dataset...")

        ratings = pd.read_csv(Path(dataset_path) / "ratings.csv")[["userId", "movieId", "rating"]]
        movies = pd.read_csv(Path(dataset_path) / "movies.csv")

        # Train/test split
        train, _ = train_test_split(ratings, test_size=0.2, random_state=42, shuffle=True)

        # Baseline параметры
        self.global_mean = train["rating"].mean()
        self.item_bias = (train.groupby("movieId")["rating"].mean() - self.global_mean).to_dict()
        self.user_bias = (train.groupby("userId")["rating"].mean() - self.global_mean).to_dict()

        # Train items by user
        self.train_items_by_user = train.groupby("userId")["movieId"].apply(set).to_dict()

        # Content-based фичи
        movies_genres = movies.copy()
        genre_ohe = movies_genres["genres"].str.get_dummies(sep="|")
        if "(no genres listed)" in genre_ohe.columns:
            genre_ohe = genre_ohe.drop(columns="(no genres listed)")

        self.movie_features = pd.concat([movies_genres[["movieId"]], genre_ohe], axis=1)
        self.movie_features.set_index("movieId", inplace=True)
        self.genre_cols = list(genre_ohe.columns)

        # User profiles (rating >= 4)
        positive_ratings = ratings[ratings["rating"] >= 4.0]
        user_movie = positive_ratings.merge(
            self.movie_features[self.genre_cols],
            left_on="movieId",
            right_index=True,
            how="inner"
        )
        self.user_profile = user_movie.groupby("userId")[self.genre_cols].mean()

        # User seen movies
        self.user_seen = ratings.groupby("userId")["movieId"].apply(set).to_dict()

        print(f"✅ Модель загружена: {len(self.movie_features)} фильмов, {len(self.user_profile)} профилей")

    def recommend_baseline(self, user_id: int, top_n: int = 10) -> List[int]:
        """Baseline рекомендации µ + b_u + b_i."""
        bu = self.user_bias.get(user_id, 0.0)
        scores = self.global_mean + bu + pd.Series(
            [self.item_bias.get(mid, 0.0) for mid in self.movie_features.index],
            index=self.movie_features.index
        )

        seen = self.train_items_by_user.get(user_id, set())
        scores = scores.drop(labels=list(seen), errors="ignore")

        return scores.sort_values(ascending=False).head(top_n).index.tolist()

    def recommend_content_based(self, user_id: int, top_n: int = 10) -> List[int]:
        """Content-based рекомендации по жанрам."""
        if user_id not in self.user_profile.index:
            return []

        u_vec = self.user_profile.loc[user_id][self.genre_cols].values.reshape(1, -1)
        movie_vecs = self.movie_features[self.genre_cols].fillna(0).values

        sims = cosine_similarity(u_vec, movie_vecs)[0]
        scores = pd.Series(sims, index=self.movie_features.index)

        seen = self.user_seen.get(user_id, set())
        scores = scores.drop(labels=list(seen), errors="ignore")

        return scores.sort_values(ascending=False).head(top_n).index.tolist()

    def predict(self, request: RecommendationRequest) -> RecommendationResponse:
        """Основной метод предсказания."""
        try:
            if request.model_type == "baseline":
                recommendations = self.recommend_baseline(request.user_id, request.top_n)
            else:  # content
                recommendations = self.recommend_content_based(request.user_id, request.top_n)

            return RecommendationResponse(
                user_id=request.user_id,
                model_type=request.model_type,
                top_n=request.top_n,
                recommendations=recommendations,
                count=len(recommendations)
            )
        except Exception as e:
            raise RuntimeError(f"Ошибка предсказания: {str(e)}")
