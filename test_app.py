import io
import pytest
from PIL import Image
from fastapi.testclient import TestClient
from app import app

client = TestClient(app)


def test_metrics_endpoint():
    """Verify Prometheus metrics are exported successfully."""
    response = client.get("/metrics")
    assert response.status_code == 200
    assert "# HELP" in response.text


def test_stream_frame_endpoint():
    """Generate a valid 1x1 RGB image in memory and test frame processing route."""
    # Create a real 1x1 JPEG image using PIL
    img = Image.new("RGB", (1, 1), color="red")
    img_byte_arr = io.BytesIO()
    img.save(img_byte_arr, format="JPEG")
    img_bytes = img_byte_arr.getvalue()

    response = client.post(
        "/stream_frame", files={"file": ("test.jpg", img_bytes, "image/jpeg")}
    )
    assert response.status_code == 200
