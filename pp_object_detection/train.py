from ultralytics import YOLO

# Load a model
model = YOLO('yolov8n.pt')

# Train the model
results = model.train(data="pp.yaml", epochs=50, imgsz=1800)