FROM python:3.10-slim

# System dependencies for OpenCV and image operations
RUN apt-get update && apt-get install -y \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy requirements and install Python packages
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application file and exported ONNX model
COPY app.py .
COPY models/yolov8n.onnx ./models/yolov8n.onnx

EXPOSE 8000

# Start the FastAPI application using Uvicorn
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]