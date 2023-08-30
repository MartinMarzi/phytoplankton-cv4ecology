from ultralytics import YOLO
import ipdb

# Load model
model = YOLO(
    "/home/martin/cv4e/pp_segmentation/pp_segment/resol=1920_epochs=70/weights/best.pt"
    )

# Define path to the image file
source = "/mnt/ssd-cluster/martin/datasets/mask/images/test/Pn_calliantha_20230523_0DB2_1.jpg"

# Predict with the model
results = model(
    source, 
    save=True, 
    save_txt = True,
    save_conf=False,
    retina_masks=True, 
    imgsz=1920,
    iou = 0.2,
    name="pred_resol=1920_epochs=70_iou=0.2_",
    cfg="config.yaml"
    ) # show=True, 

# View results
for result in results:
    masks = result.masks # Masks object for segmentation masks outputs
    for mask in masks.xyn:
        print(mask)
    # print(f"mask {count}\n{masks}")  # print the Masks object containing the detected instance masks
    # print(f"mask nr. pixels: {len(masks)}")  # print the Masks object containing the detected instance masks
    # print(f"box: {result.boxes.xyxy}")  # print the Boxes object containing the detection bounding boxes
    
    # ipdb.set_trace()


