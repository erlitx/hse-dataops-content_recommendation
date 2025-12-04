import pytest
from httpx import AsyncClient
import sys
import os

# Добавляем app в путь
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from app.main import app


@pytest.fixture
async def client():
    async with AsyncClient(app=app, base_url="http://test") as ac:
        # Ждем загрузки модели
        await ac.get("/health")
        yield ac


@pytest.mark.asyncio
async def test_health_check(client: AsyncClient):
    """Тест health check."""
    response = await client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


@pytest.mark.asyncio
async def test_happy_path(client: AsyncClient):
    """Тест успешного запроса рекомендаций."""
    payload = {
        "user_id": 1,
        "top_n": 10,
        "model_type": "baseline"
    }

    response = await client.post("/recommendations", json=payload)
    assert response.status_code == 200

    data = response.json()
    assert data["user_id"] == 1
    assert data["top_n"] == 10
    assert data["model_type"] == "baseline"
    assert isinstance(data["recommendations"], list)
    assert data["count"] == 10


@pytest.mark.asyncio
async def test_bad_input(client: AsyncClient):
    """Тест невалидного запроса."""
    payload = {
        "user_id": "invalid",  # не число
        "top_n": 10,
        "model_type": "baseline"
    }

    response = await client.post("/recommendations", json=payload)
    assert response.status_code == 422


@pytest.mark.asyncio
async def test_invalid_model_type(client: AsyncClient):
    """Тест неверного типа модели."""
    payload = {
        "user_id": 1,
        "top_n": 10,
        "model_type": "invalid"
    }

    response = await client.post("/recommendations", json=payload)
    assert response.status_code == 422


@pytest.mark.asyncio
async def test_content_based(client: AsyncClient):
    """Тест content-based модели."""
    payload = {
        "user_id": 1,
        "top_n": 5,
        "model_type": "content"
    }

    response = await client.post("/recommendations", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["model_type"] == "content"
    assert data["count"] >= 0  # может быть 0 если нет профиля
