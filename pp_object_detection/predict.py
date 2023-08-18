from ultralytics import YOLO

# Load  a model
model = YOLO("/home/martin/cv4e/pp_object_detection/runs/detect/train8/weights/best.pt")

# Define path to the image file
# source = "/mnt/ssd-cluster/martin/datasets/bbox/images/test/Pn_calliantha_20230523_0024_5.jpg"

# Predict with the model
results = model("/mnt/ssd-cluster/martin/datasets/bbox/images/test/Pn_calliantha_20230523_0024_5.jpg", show=True, save=True) 