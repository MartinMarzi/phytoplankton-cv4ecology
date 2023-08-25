from ultralytics import YOLO

# Load a model
model = YOLO("/home/martin/cv4e/pp_segmentation/pp_segment/train/weights/best.pt")

# Validate the model
metrics = model.val(save_json=True, cfg="config.yaml")
metrics.box.map
metrics.box.maps   # a list contains map50-95(B) of each category
metrics.seg.map 
metrics.seg.maps 