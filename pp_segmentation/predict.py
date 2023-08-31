from ultralytics import YOLO
import ipdb
import numpy as np

# Load model
model = YOLO(
    "/home/martin/cv4e/pp_segmentation/pp_segment/resol=1920_epochs=70/weights/best.pt"
    )

# Define path to the image file
# source = "/mnt/ssd-cluster/martin/datasets/mask/images/test/Pn_calliantha_20230523_0DB2_1.jpg"
source = "/mnt/ssd-cluster/martin/datasets/mask/images/holdout/Pn_calliantha_20230601_0DB2_11.jpg"

# Predict with the model
results = model(
    source, 
    save=True, 
    save_txt = True,
    save_conf=True,
    retina_masks=True, 
    imgsz=1920,
    iou = 0.2,
    name="pred_resol=1920_epochs=70_iou=0.2_",
    cfg="config.yaml"
    ) # show=True, 

print (type(results))
# View results
result =results[0]

shit = result.masks.cpu() # Masks object for segmentation masks outputs
masks = shit.data.numpy()
print (f"type is {type(masks)} and shape is {masks.shape}")
num_masks = masks.shape[0]
print (f"Number of masks detected is {num_masks}")
for idx in range(0,num_masks):
    print (f"Mask number: {idx+1}")
    individual_mask = masks[idx]
    plt.imshow(individual_mask)
    x_ones,y_ones = np.where(individual_mask == 1)
    number_of_ones = x_ones.shape[0]
    print (f"Number of pixels is {number_of_ones}")
    break

#for mask in masks.xyn:
#    print(mask)
# print(f"mask {count}\n{masks}")  # print the Masks object containing the detected instance masks
# print(f"mask nr. pixels: {len(masks)}")  # print the Masks object containing the detected instance masks
# print(f"box: {result.boxes.xyxy}")  # print the Boxes object containing the detection bounding boxes

# ipdb.set_trace()


