from ultralytics import YOLO


def main():

    model = YOLO("models/yolov8n.pt")
    try:
        model.export(format="onnx")
        print("Model exported to ONNX format successfully.")
    except Exception as e:
        print(f"Error exporting model: {e}")


if __name__ == "__main__":
    main()
