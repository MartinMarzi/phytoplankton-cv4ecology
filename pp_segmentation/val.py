from ultralytics import YOLO

# Load a model
model = YOLO("/home/martin/cv4e/pp_segmentation/runs/segment/train7/weights/best.pt")

# Validate the model
metrics = model.val()
metrics.box.map
metrics.box.maps   # a list contains map50-95(B) of each category
metrics.seg.map 
metrics.seg.maps 