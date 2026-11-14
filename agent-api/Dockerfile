FROM python:3.12-slim

WORKDIR /app

# R5a: không cài curl (apt-get) — healthcheck trong docker-compose.yml dùng python của image.
# Bản vá OS đến từ base image: rebuild định kỳ với `--pull` để lấy python:3.12-slim mới,
# vd. `docker compose build --pull agent-api && docker compose up -d agent-api`.

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8091

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8091"]
