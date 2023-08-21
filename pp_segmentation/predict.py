from ultralytics import YOLO

# Load model
model = YOLO("/home/martin/cv4e/pp_segmentation/runs/segment/train5/weights/best.pt")

# Define path to the image file
source = "/mnt/ssd-cluster/martin/datasets/mask/images/test/Pn_calliantha_20230523_0035_13.jpg"

# Predict with the model
results = model(source)