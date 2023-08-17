# Convert bbox annotations from COCO to YOLO format
import os
import json
import shutil
from PIL import Image

# testing
coco_annotations_path = "/Users/mmarzi/MLprojects/cv4e/dataset/data_bbox/raw/Bounding Box 1/annotations/instances_default.json"
output_folder = "/Users/mmarzi/MLprojects/cv4e/dataset/data_bbox/interim"
image_folder =  "/Users/mmarzi/MLprojects/cv4e/dataset/data_bbox/raw/Bounding Box 1"

def coco2yolo(coco_annotations_path, image_folder, output_folder):
    """
    Converts COCO format to JOLO format
    """
    with open(coco_annotations_path, "r") as f:
        data = json.load(f)

    # Ensure output directory exists
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Create a dictionary for mapping image id to filename
    image_id_to_name = {img["id"]: img["file_name"].replace('.jpg', '') for img in data["images"]}

    # Convert COCO annotations to YOLO format
    for annotation in data["annotations"]:
        # Get image file name
        image_name = image_id_to_name[annotation["image_id"]] 
        # Get bbox values
        x, y, width, height = annotation["bbox"]
        # Convert to YOLO format (center-x, center-y, width, height) and 
        # normalize the values by image width and height        
        img_width = 2584
        img_height = 1936
        x_center = (x + width / 2) / img_width
        y_center = (y + height / 2) / img_height
        width /= img_width
        height /= img_height
    
        # Convert class ID (correct -1)
        class_id = annotation["category_id"] - 1

        # Write output text file
        with open(os.path.join(output_folder, f"{image_name}.txt"), "a") as f:
            f.write(f"{class_id} {x_center} {y_center} {width} {height}\n")
        
def main():
    coco2yolo(coco_annotations_path, image_folder, output_folder)

main()