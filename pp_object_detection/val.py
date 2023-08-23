from ultralytics import YOLO

# Load a model
model = YOLO("/home/martin/cv4e/pp_object_detection/runs/detect/train8/weights/best.pt")

# Validate the model
metrics = model.val()
metrics.box.maps 