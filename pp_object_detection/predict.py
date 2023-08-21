from ultralytics import YOLO

# Load  a model
model = YOLO("/home/martin/cv4e/pp_object_detection/runs/detect/train8/weights/best.pt")

# Define path to the image file
source = "/mnt/ssd-cluster/martin/datasets/bbox/images/test/Pn_delicatissima_20230601_0DB2_6.jpg"

# Predict with the model
results = model(source, save=True) 
