import comet_ml 
import os
from ultralytics import YOLO

COMET_API_KEY = "oL5iq9NEaBxQ2y6s9LSKz8NuV"
os.environ["COMET_API_KEY"]= COMET_API_KEY

# initialize experiment in comet
comet_ml.init("pp_seg")

# Load a model
model = YOLO("yolov8n-seg.pt")

# Train the model
results = model.train(
    data="pp.yaml", 
    epochs=70, 
    imgsz=1920, 
    cfg="config.yaml",
    name="resol=1920_epochs=70"
    )