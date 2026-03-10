from ultralytics import YOLO

PT_MODEL   = YOLO("yolov8n.pt")
PT_MODEL.export(format="cnn")