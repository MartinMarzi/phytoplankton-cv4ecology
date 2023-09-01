from ultralytics import YOLO
import os
import numpy as np
import matplotlib.pylab as plt
import pandas as pd

# Load model
model = YOLO(
    "/home/martin/cv4e/pp_segmentation/pp_segment/resol=1920_epochs=70/weights/best.pt"
    )

# Get all image file path names for the test set
dir_path = "/mnt/ssd-cluster/martin/datasets/mask/images/test"
source = [os.path.join(dir_path, f) for f in os.listdir(dir_path)]
# source = "/mnt/ssd-cluster/martin/datasets/mask/images/test/Pn_delicatissima_20230523_0DB2_36.jpg"

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

data = []

for image in results:
    if image:
        interm = image.masks.cpu() # Masks object for segmentation masks outputs
        masks = interm.data.numpy()
        # print (f"type is {type(masks)} and shape is {masks.shape}")
        num_masks = masks.shape[0]
        print (f"Number of masks detected is {num_masks}")
        # get image name
        img_path = image.path
        img_name = os.path.basename(img_path)

        print(f"image name is {img_name}")

        for idx in range(0,num_masks):
            print (f"Mask number: {idx+1}")
            individual_mask = masks[idx]
            plt.imshow(individual_mask)
            # Save the plot to a file
            image_dir = "/home/martin/cv4e/evaluation/evaluation_data/pred_test/"
            output_path = os.path.join(image_dir, img_name.split(".")[0] + f"_{idx}.jpg")
            plt.savefig(output_path)
            # get class
            mask_class = str(int(image.boxes.cls.cpu().numpy()[idx]))
            # print(image.names[mask_class])
            print(f"mask class is: {mask_class}")
            # Calculate the surface
            x_ones,y_ones = np.where(individual_mask == 1)
            surface = x_ones.shape[0]
            confidence = max(image.boxes.conf.cpu().numpy())
            print (f"Number of pixels is {surface}")
        
            data.append([mask_class, surface, confidence, img_name, idx])
       
data = pd.DataFrame(data, columns=['class', 'surface', 'confidence', 'image_name', 'img_inst_id'])

# Save to CSV 
data.to_csv("/home/martin/cv4e/evaluation/evaluation_data/pred_test/pred_test_annotations.csv", index=False)