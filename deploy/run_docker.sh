#!/usr/bin/env bash
set -euo pipefail

IMAGE_NAME="movielens-recommender"

echo "Сборка Docker-образа ${IMAGE_NAME}..."
docker build -t "${IMAGE_NAME}" .

echo "Запуск контейнера на порту 8000..."
docker run --rm -p 8000:8000 "${IMAGE_NAME}"
