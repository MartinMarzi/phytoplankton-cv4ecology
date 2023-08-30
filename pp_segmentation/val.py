from ultralytics import YOLO

# Load a model
model = YOLO("/home/martin/cv4e/pp_segmentation/pp_segment/resol=1920_epochs=70/weights/best.pt")

# Validate the model
metrics = model.val(
    save_json=True,
    iou=0.2,
    name="resol=1920_epochs=70_iou=0.2", 
    cfg="config.yaml")

print(metrics.box.map)    # map50-95(B)
print(metrics.box.maps)   # a list contains map50-95(B) of each category

print(metrics.seg.map50)  # map50(M)
print(metrics.seg.maps)   # a list contains map50-95(M) of each category