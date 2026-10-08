FROM python:3.10-slim

WORKDIR /app

# 1. Install system dependencies + C compilation tools
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgl1 \
    libglib2.0-0 \
    libxcb1 \
    && rm -rf /var/lib/apt-get/lists/*

COPY requirements.txt .

# 2. Re-enable local BuildKit cache + add timeout protection
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --default-timeout=1000 -r requirements.txt

# 3. Application Code
COPY app.py .
COPY retrain.py .
COPY models/yolov8n.onnx ./models/yolov8n.onnx

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]