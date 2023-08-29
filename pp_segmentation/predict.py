from ultralytics import YOLO

# Load model
model = YOLO(
    "/home/martin/cv4e/pp_segmentation/pp_segment/train3/weights/best.pt"
    )

# Define path to the image file
source = "/mnt/ssd-cluster/martin/datasets/mask/images/test/Pn_calliantha_20230523_0DB2_1.jpg"

# Predict with the model
results = model(
    source, 
    save=True, 
    save_txt = True,
    save_conf=True,
    retina_masks=True, 
    imgsz=1920,
    cfg="config.yaml"
    ) # show=True, 

# View results
count = 0
for result in results:
    masks = result.masks.xyn  # Masks object for segmentation masks outputs
    print(f"mask {count}\nnr. pixesl: {len(masks[count])} \n{masks}")  # print the Masks object containing the detected instance masks
    # print(f"box: {result.boxes.xyxy}")  # print the Boxes object containing the detection bounding boxes
    count=+1

