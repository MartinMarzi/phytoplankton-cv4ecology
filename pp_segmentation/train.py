from ultralytics import YOLO

# Load a model
model = YOLO("yolov8s-seg.pt")

# Train the model
results = model.train(data="pp.yaml", epochs=100, imgsz=640, plot=True)

