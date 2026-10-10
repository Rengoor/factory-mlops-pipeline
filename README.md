# Industrial Edge MLOps Pipeline

An end-to-end MLOps pipeline for real-time defect detection on edge camera streams. Built with **FastAPI**, **MLflow**, **Prometheus**, **Grafana**, and **Docker Compose**, featuring automated model retraining loops and real-time operational metrics.

---

## System Architecture

```text
               +----------------------------------+
               |  Simulated Video Frame Stream   |
               +----------------------------------+
                                | (HTTP POST)
                                v
               +----------------------------------+
               |    FastAPI Inference API         |
               |  (OpenCV + YOLOv8 Edge Camera)   |
               +----------------------------------+
                  /                            \
                 / (Metrics Scraping)           \ (Model Tracking)
                v                                v
    +-----------------------+          +-----------------------+
    |   Prometheus Server   |          |    MLflow Server      |
    |  (Metrics Engine)     |          | (Experiment Tracking) |
    +-----------------------+          +-----------------------+
                |                                  |
                v                                  v
    +-----------------------+          +-----------------------+
    |   Grafana Dashboard   |          |  Retraining Trigger   |
    | (Real-time Latency/RPS)|         | (Champion vs Challenger)|
    +-----------------------+          +-----------------------+
```

---

## 💡 Design Choices & Core Philosophy

### 1. Full-Stack Observability (Prometheus + Grafana)
* **Why:** Experiment tracking with MLflow tells you how well a model trained offline, but it reveals nothing about live operational health. In an industrial factory, frame processing delays directly impact conveyor belt speed and camera stream ingestion.
* **Choice:** Instrumented FastAPI endpoints using `prometheus-fastapi-instrumentator` to export throughput and frame latency. Provisioned Grafana automatically via YAML configuration files so dashboards load pre-configured without manual setup.

### 2. Microservice Architecture (Docker Compose)
* **Why:** In an edge deployment, services must run isolated to prevent dependency conflicts between ML frameworks, tracking servers, and web UI tools.
* **Choice:** Decoupled the architecture into 4 dedicated containers attached to an isolated bridge network called `mlops_network`: `inference_api`, `mlflow_server`, `prometheus`, and `grafana`.

### 3. BuildKit Caching Optimization
* **Why:** Deep learning and computer vision dependencies (OpenCV, PyTorch/ONNX) cause long Docker build times during local testing.
* **Choice:** Leveraged Docker BuildKit pip cache mounts in the dockerfile (`/root/.cache/pip`) and strategic layer ordering, drastically reducing re-build times during iteration.

### 4. Automated Drift & Retraining Loop
* **Why:** Manual model updates lead to silent degradation in factory environments as lighting or defect patterns drift over time.
* **Choice:** Built a "Champion vs. Challenger" retraining trigger that compares incoming inference performance against MLflow logged baseline metrics.

---

## 🚀 Getting Started & Setup Instructions

### Prerequisites
* [Docker Desktop](https://www.docker.com/products/docker-desktop/) (v20.10+ recommended)
* [Python 3.10](https://www.python.org/) (for client-side frame simulation)
* [Git](https://git-scm.com/)

### Step 1: Clone the Repository
```bash
git clone https://github.com/Rengoor/factory-mlops-pipeline.git
cd factory-mlops-pipeline
```

### Step 2: Spin Up the Stack
Run the entire containerized pipeline with a single command:
```bash
docker compose up -d --build
```

### Step 3: Verify Running Services
Run `docker ps` to verify all 4 services are healthy:
* **FastAPI Inference Engine:** [http://localhost:8000/docs](http://localhost:8000/docs)
* **MLflow Tracking Server:** [http://localhost:5001](http://localhost:5001)
* **Prometheus Server:** [http://localhost:9090](http://localhost:9090)
* **Grafana Monitoring Dashboard:** [http://localhost:3000](http://localhost:3000) *(Credentials: `admin` / `admin`)*

### Step 4: Simulate Factory Camera Stream
In a local Python environment, run the frame generator script to stream simulated video frames to the API:
```bash
pip install requests
python simulate_video_frames.py
```
Open Grafana at `http://localhost:3000` to view real-time latency and throughput metrics flowing into the Factory Camera Monitor dashboard.

---

## 📦 Container Registry & Artifacts (Upcoming)

Docker container images for the `inference_api` edge deployment will be automatically built and pushed to **GitHub Packages (GHCR)** via CI/CD automation.

---

## 🔮 Roadmap & What's Next

- [x] FastAPI Edge Inference Engine with OpenCV rendering
- [x] MLflow Integration for artifact tracking & retraining loops
- [x] System Observability Stack (Prometheus + Grafana provisioning)
- [ ] **GitHub Actions CI/CD:** Automated testing (`pytest`), code formatting (`flake8`), and container build verification on push.
- [ ] **GitHub Packages (GHCR):** Automatic Docker image release publishing on tagged releases.