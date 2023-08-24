from ultralytics import YOLO

# Load a model
model = YOLO("/home/martin/cv4e/pp_object_detection/pp_detect/train2/weights/best.pt")

# Validate the model
metrics = model.val(save_json=True, cfg="config.yaml")
metrics.box.maps 