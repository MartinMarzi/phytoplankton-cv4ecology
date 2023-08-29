from ultralytics import YOLO

# Load a model
model = YOLO("/home/martin/cv4e/pp_segmentation/pp_segment/train3/weights/best.pt")

# Validate the model
metrics = model.val(
    save_json=True, 
    cfg="config.yaml")

metrics.box.map    # map50-95(B)
metrics.box.map50  # map50(B)
metrics.box.map75  # map75(B)
metrics.box.maps   # a list contains map50-95(B) of each category
metrics.seg.map    # map50-95(M)
metrics.seg.map50  # map50(M)
metrics.seg.map75  # map75(M)
metrics.seg.maps   # a list contains map50-95(M) of each category