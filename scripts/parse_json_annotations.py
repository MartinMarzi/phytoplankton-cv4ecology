import json
import pandas as pd
import numpy as np

def extract_annotations_data(json_path):
    with open(json_path, "r") as file:
        data = json.load(file)

    # Create a mapping for image_id to image_name from the 'images' section
    image_id_to_name = {img["id"]: img["file_name"] for img in data["images"]}

    # Create a mapping for category_id to category_name from the 'categories' section
    category_id_to_name = {cat['id']: cat['name'] for cat in data['categories']}

    # Extract required information
    annotations_data = []
    for annotation in data['annotations']:
        image_name = image_id_to_name[annotation['image_id']]
        class_name = category_id_to_name[annotation['category_id']]
        surface = annotation['area']
        confidence = np.nan
        img_inst_id = annotation['id']

        annotations_data.append([class_name, surface, confidence, image_name, img_inst_id])
    
    return annotations_data

# json_path = "/Users/mmarzi/MLprojects/cv4e/evaluation/evaluation_data/test_annotated/coco_labels/instances_default_1.json"
# extract_annotations_data(json_path)

# Paths to the JSON files
json_path1 = "/Users/mmarzi/MLprojects/cv4e/evaluation/evaluation_data/test_annotated/coco_labels/instances_default_1.json"
json_path2 = "/Users/mmarzi/MLprojects/cv4e/evaluation/evaluation_data/test_annotated/coco_labels/instances_default_2.json"

# Extract data from both JSON files
data1 = extract_annotations_data(json_path1)
data2 = extract_annotations_data(json_path2)

# Combine data and convert to DataFrame
combined_data = data1 + data2
df_annotations = pd.DataFrame(combined_data, columns=['class', 'surface', 'confidence', 'image_name', 'img_inst_id'])

# Save to CSV 
df_annotations.to_csv("GT_annotations.csv", index=False)
