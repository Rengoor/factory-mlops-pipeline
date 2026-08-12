# simulate_video_stream.py
import time
import requests
import numpy as np
from PIL import Image, ImageEnhance
import io

API_URL = "http://localhost:8000/stream_frame"


def generate_synthetic_factory_frame(darken=False):
    """Generates a dummy industrial frame (or modifies lighting)."""
    # Create a random RGB image representing a conveyor belt part
    arr = np.random.randint(100, 200, (480, 640, 3), dtype=np.uint8)
    img = Image.fromarray(arr)

    if darken:
        # Simulate factory camera failure / lighting change
        enhancer = ImageEnhance.Brightness(img)
        img = enhancer.enhance(0.2)  # 80% darker

    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    return buf.getvalue()


def start_camera_stream():
    print("🎥 Starting Industrial Video Camera Feed Simulation...")

    # Phase 1: Stream 50 normal frames to establish baseline
    for i in range(50):
        frame_bytes = generate_synthetic_factory_frame(darken=False)
        res = requests.post(
            API_URL, files={"file": ("frame.jpg", frame_bytes, "image/jpeg")}
        )
        print(f"Frame {i+1}/50 [Baseline Phase]:", res.json()["status"])
        time.sleep(0.05)

    print(
        "\n⚠️ SIMULATION EVENT: Factory main light turns off / Camera underexposed!\n"
    )

    # Phase 2: Stream darkened corrupted frames to trigger drift
    for i in range(40):
        frame_bytes = generate_synthetic_factory_frame(darken=True)
        res = requests.post(
            API_URL, files={"file": ("frame.jpg", frame_bytes, "image/jpeg")}
        )
        data = res.json()
        print(f"Frame {i+51} [Production Feed]:", data)

        if data.get("status") == "DRIFT_DETECTED":
            print(
                "\n✅ SUCCESS: Pipeline successfully caught video stream drift in real time!"
            )
            break
        time.sleep(0.05)


if __name__ == "__main__":
    start_camera_stream()
