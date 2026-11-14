FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# R5a: không còn lớp apt-get (trước đây chỉ để cài curl cho healthcheck và chạy
# `apt-get upgrade`). Healthcheck trong docker-compose.yml dùng python của image.
# Bản vá OS đến từ base image: rebuild định kỳ với `--pull` để lấy python:3.11-slim mới,
# vd. `docker compose build --pull rag-api && docker compose up -d rag-api`.

COPY requirements.txt .

RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8090

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8090"]
