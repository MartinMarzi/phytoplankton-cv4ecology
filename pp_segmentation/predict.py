from ultralytics import YOLO
import ipdb
import numpy as np

# Load model
model = YOLO(
    "/home/martin/cv4e/pp_segmentation/pp_segment/resol=1920_epochs=70/weights/best.pt"
    )

# Define path to the image file
source = "/mnt/ssd-cluster/martin/datasets/mask/images/test/Pn_calliantha_20230523_0DB2_1.jpg"
# source = "/mnt/ssd-cluster/martin/datasets/mask/images/holdout/Pn_calliantha_20230601_0DB2_11.jpg"

# Predict with the model
results = model(
    source, 
    save=True, 
    # save_txt = True,
    save_conf=True,
    retina_masks=True, 
    imgsz=1920,
    iou = 0.2,
    # classes=list,
    name="pred_resol=1920_epochs=70_iou=0.2_",
    cfg="config.yaml"
    ) # show=True, 

# # Process results list
# for image in results:
#     # org_shape = result.orig_shape
#     masks = result.masks  # Masks object for segmentation masks outputs
#     probability = result.probs
#     mask_class = result.boxes.cls.cpu().numpy()[]
#     img_path = result.path

# Export results

for image in results:
    interm = image.masks.cpu() # Masks object for segmentation masks outputs
    masks = interm.data.numpy()
    # print (f"type is {type(masks)} and shape is {masks.shape}")
    num_masks = masks.shape[0]
    print (f"Number of masks detected is {num_masks}")
    for idx in range(0,num_masks):
        print (f"Mask number: {idx+1}")
        individual_mask = masks[idx]
        plt.imshow(individual_mask)
        # Save the plot to a file
        output_path = "/home/martin/cv4e/evaluation/evaluation_data/pred_test/mask.png"
        plt.savefig(output_path)
        # get class
        mask_class = int(image.boxes.cls.cpu().numpy()[idx])
        # print(image.names[mask_class])
        print(f"mask class is: {mask_class}")
        # Calculate the surface
        x_ones,y_ones = np.where(individual_mask == 1)
        number_of_ones = x_ones.shape[0]
        print (f"Number of pixels is {number_of_ones}")
    



