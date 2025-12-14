from fastapi import FastAPI, HTTPException
import math
from pathlib import Path
import pandas as pd
import numpy as np
import kagglehub
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, pairwise_distances
from sklearn.metrics.pairwise import cosine_similarity
from contextlib import asynccontextmanager

# --- Глобкалка для хранения моделей и данных ---
models_data = {}

# --- Функции метрик  ---
def precision_at_k(pred_items, true_items, k):
    if k == 0: return 0.0
    pred_k = pred_items[:k]
    hit_count = sum(1 for item in pred_k if item in true_items)
    return hit_count / k

def recall_at_k(pred_items, true_items, k):
    if not true_items: return 0.0
    pred_k = pred_items[:k]
    hit_count = sum(1 for item in pred_k if item in true_items)
    return hit_count / len(true_items)

# --- Логика обучения ---
def prepare_models():
    print("Дергаю датафрейм с kagglehub...")
    path = kagglehub.dataset_download("shubhammehta21/movie-lens-small-latest-dataset")
    print(f"Датафрейм сохранен в: {path}")

    ratings = pd.read_csv(Path(path) / "ratings.csv")
    movies = pd.read_csv(Path(path) / "movies.csv")
    
    ratings = ratings[["userId", "movieId", "rating"]]
    train, test = train_test_split(ratings, test_size=0.2, random_state=42, shuffle=True)
    
    # --- Обучение базовой модели ---
    global_mean = train["rating"].mean()
    item_mean = train.groupby("movieId")["rating"].mean()
    item_bias = item_mean - global_mean
    user_mean = train.groupby("userId")["rating"].mean()
    user_bias = user_mean - global_mean
    
    train_items_by_user = train.groupby("userId")["movieId"].apply(set).to_dict()

    # --- Ебашилово обучения на данных ---
    movies_genres = movies.copy()
    genre_ohe = movies_genres["genres"].str.get_dummies(sep="|")
    if "(no genres listed)" in genre_ohe.columns:
        genre_ohe = genre_ohe.drop(columns="(no genres listed)")
    
    movie_features = pd.concat([movies_genres[["movieId", "title"]], genre_ohe], axis=1)
    movie_features.set_index("movieId", inplace=True)
    genre_cols = genre_ohe.columns
    
    positive_ratings = ratings[ratings["rating"] >= 4.0]
    user_movie = positive_ratings.merge(movie_features[genre_cols], left_on="movieId", right_index=True, how="inner")
    user_profile = user_movie.groupby("userId")[genre_cols].mean()
    user_seen = ratings.groupby("userId")["movieId"].apply(set).to_dict()

    # Все в глобальный словарь
    models_data['global_mean'] = global_mean
    models_data['user_bias'] = user_bias
    models_data['item_bias'] = item_bias
    models_data['movies'] = movies
    models_data['train_items_by_user'] = train_items_by_user
    models_data['user_profile'] = user_profile
    models_data['movie_features'] = movie_features
    models_data['genre_cols'] = genre_cols
    models_data['user_seen'] = user_seen
    
    print("Models prepared successfully!")

# --- ПУСК ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    # ЗАГРУЗКА И ОБУЧЕНИЕ
    prepare_models()
    yield
    # Тут можно очистить ресурсы

app = FastAPI(lifespan=lifespan)

# --- Рекомендационные функции ---
def get_baseline_recs(user_id: int, top_n: int = 10):
    user_bias = models_data['user_bias']
    item_bias = models_data['item_bias']
    global_mean = models_data['global_mean']
    movies = models_data['movies']
    train_items_by_user = models_data['train_items_by_user']

    bu = user_bias.get(user_id, 0.0)
    scores = global_mean + bu + item_bias.reindex(movies["movieId"]).fillna(0.0)
    scores.index = movies["movieId"]
    
    seen = train_items_by_user.get(user_id, set())
    scores = scores.drop(labels=list(seen), errors="ignore")
    
    top_ids = scores.sort_values(ascending=False).head(top_n).index.tolist()
    
    # Добавление названия фильмов шоб наглядно было
    results = []
    for mid in top_ids:
        title = movies[movies['movieId'] == mid]['title'].values[0] if not movies[movies['movieId'] == mid].empty else "Unknown"
        results.append({"movieId": int(mid), "title": title})
    return results

def get_content_recs(user_id: int, top_n: int = 10):
    user_profile = models_data['user_profile']
    movie_features = models_data['movie_features']
    genre_cols = models_data['genre_cols']
    user_seen = models_data['user_seen']
    movies = models_data['movies']

    if user_id not in user_profile.index:
        return []

    u_vec = user_profile.loc[user_id][genre_cols].values.reshape(1, -1)
    movie_vecs = movie_features[genre_cols].values

    sims = cosine_similarity(u_vec, movie_vecs)[0]
    scores = pd.Series(sims, index=movie_features.index)

    seen = user_seen.get(user_id, set())
    scores = scores.drop(labels=list(seen), errors="ignore")

    top_ids = scores.sort_values(ascending=False).head(top_n).index.tolist()
    
    results = []
    for mid in top_ids:
        title = movies[movies['movieId'] == mid]['title'].values[0] if not movies[movies['movieId'] == mid].empty else "Unknown"
        results.append({"movieId": int(mid), "title": title})
    return results

# --- Эндпоинты ---

@app.get("/")
def read_root():
    return {"message": "Movie Recommender API is Ready!", "status": "OK"}

@app.get("/recommend/baseline/{user_id}")
def recommend_baseline_endpoint(user_id: int):
    try:
        recs = get_baseline_recs(user_id)
        return {"user_id": user_id, "algorithm": "baseline", "recommendations": recs}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/recommend/content/{user_id}")
def recommend_content_endpoint(user_id: int):
    try:
        recs = get_content_recs(user_id)
        if not recs:
            return {"user_id": user_id, "message": "User not found or no profile available"}
        return {"user_id": user_id, "algorithm": "content-based", "recommendations": recs}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
def health_check():
    return {"status": "active", "models_loaded": bool(models_data)}


