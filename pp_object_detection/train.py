import comet_ml 
import os
from ultralytics import YOLO

COMET_API_KEY = "oL5iq9NEaBxQ2y6s9LSKz8NuV"
os.environ["COMET_API_KEY"]= COMET_API_KEY

# initialize experiment in comet
comet_ml.init("pp_detect")

# Load a model
model = YOLO('yolov8n.pt')

# Train the model
results = model.train(data="pp.yaml", epochs=100, imgsz=640, cfg="config.yaml") 
