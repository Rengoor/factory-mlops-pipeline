from fastapi import FastAPI, UploadFile, File
import onnxruntime as ort
import numpy as np
from PIL import Image
import io
import pandas as pd
from evidently import Report
from evidently.presets import DataDriftPreset
import os
import subprocess

app = FastAPI(title="Factory-Grade Industrial Camera Pipeline")

SESSION = ort.InferenceSession("models/yolov8n.onnx")

# Storage buffers for Baseline Reference vs Production Stream
REFERENCE_DATA = []
CURRENT_STREAM_BUFFER = []
BASELINE_COLLECTED = False


def preprocess_image(image_bytes: bytes):
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")

    # Calculate simple image feature (average brightness) for drift checking
    gray_img = img.convert("L")
    avg_brightness = float(np.mean(np.array(gray_img)))

    img_resized = img.resize((640, 640))
    img_data = np.array(img_resized, dtype=np.float32) / 255.0
    img_data = np.transpose(img_data, (2, 0, 1))
    img_data = np.expand_dims(img_data, axis=0)

    return img_data, avg_brightness


@app.post("/stream_frame")
async def process_stream_frame(file: UploadFile = File(...)):
    global BASELINE_COLLECTED, REFERENCE_DATA, CURRENT_STREAM_BUFFER

    image_bytes = await file.read()
    input_tensor, avg_brightness = preprocess_image(image_bytes)

    input_name = SESSION.get_inputs()[0].name
    outputs = SESSION.run(None, {input_name: input_tensor})

    # Extract maximum detection score as our model confidence proxy
    raw_preds = outputs[0]  # shape (1, 84, 8400)
    confidences = raw_preds[0, 4, :]  # Objectness / class scores
    max_confidence = float(np.max(confidences))

    frame_metrics = {"avg_brightness": avg_brightness, "max_confidence": max_confidence}

    # Step 1: Lock in first 50 normal frames as baseline reference
    if not BASELINE_COLLECTED:
        REFERENCE_DATA.append(frame_metrics)
        if len(REFERENCE_DATA) >= 50:
            BASELINE_COLLECTED = True
        return {"status": "collecting_baseline", "samples": len(REFERENCE_DATA)}

    # Step 2: Buffer incoming production video frames
    CURRENT_STREAM_BUFFER.append(frame_metrics)

    # Step 3: Trigger EvidentlyAI drift evaluation every 30 frames
    drift_detected = False
    if len(CURRENT_STREAM_BUFFER) >= 30:
        ref_df = pd.DataFrame(REFERENCE_DATA)
        curr_df = pd.DataFrame(CURRENT_STREAM_BUFFER)

        # 1. Run DataDriftPreset
        report = Report(metrics=[DataDriftPreset()])
        eval_result = report.run(reference_data=ref_df, current_data=curr_df)
        report_dict = eval_result.dict()

        # 2. Iterate through generated metrics to locate drift results
        drift_detected = False
        for metric in report_dict.get("metrics", []):
            val = metric.get("value", {})
            if isinstance(val, dict):
                # Check for explicit boolean key or non-zero drifted share
                if "dataset_drift" in val:
                    drift_detected = bool(val["dataset_drift"])
                    break
                elif "share" in val and val.get("share", 0) > 0:
                    drift_detected = True
                    break

        print(f"Drift Check Complete. Drift Detected: {drift_detected}", flush=True)

        CURRENT_STREAM_BUFFER = []

        print(f"Drift Check Complete. Drift Detected: {drift_detected}", flush=True)

        CURRENT_STREAM_BUFFER = []

        if drift_detected:
            print(
                "\n🚨 ALERT: Data Drift Detected! Triggering retraining pipeline...\n"
            )

            # Run retrain.py in background, ensuring it connects to containerized MLflow
            env = os.environ.copy()
            env["MLFLOW_TRACKING_URI"] = "http://mlflow_server:5000"

            subprocess.Popen(["python3", "retrain.py"], env=env)

            return {
                "status": "DRIFT_DETECTED",
                "action": "RETRAINING_TRIGGERED",
                "metrics": frame_metrics,
            }

    return {"status": "ok", "drift_detected": drift_detected, "metrics": frame_metrics}
