import cv2
import os

def display_yolo_segmentations(image_dir, yolo_annotations_dir, image_name):
    # Load the image
    image_path = os.path.join(image_dir, image_name + ".jpg")
    image = cv2.imread(image_path)

    if image is None:
        print(f"Failed to load image from path: {image_path}")
        return

    img_height, img_width = image.shape[:2]

    # Read YOLO annotations for the image
    annotation_path = os.path.join(yolo_annotations_dir, image_name + ".txt")
    with open(annotation_path, 'r') as file:
        lines = file.readlines()
        for line in lines:
            data = line.strip().split()
            class_id = int(data[0])
            polygon_points = [(int(float(data[i]) * img_width), int(float(data[i+1]) * img_height)) for i in range(1, len(data), 2)]
            for i in range(0, len(polygon_points)-1, 2):
                cv2.rectangle(image, polygon_points[i], polygon_points[i+1], (0, 255, 0), 2)
            cv2.putText(image, str(class_id), polygon_points[0], cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

    # Save the image to the destination
    destination_path = "/mnt/ssd-cluster/martin/cv4e/conversion_test"
    filename_on_desktop = os.path.join(destination_path, image_name + "_annotated.jpg")

    # Check if the directory exists
    if not os.path.exists(destination_path):
        print(f"Directory {destination_path} does not exist. Creating it now.")
        os.makedirs(destination_path, exist_ok=True)

    # Save and check the result
    success = cv2.imwrite(filename_on_desktop, image)
    if success:
        print(f"Annotated image saved at: {filename_on_desktop}")
    else:
        print(f"Failed to save the image at: {filename_on_desktop}")

# Example usage
image_directory = "/mnt/ssd-cluster/martin/datasets/mask/images/test"
yolo_annotation_directory = "/mnt/ssd-cluster/martin/datasets/mask/labels/test"
sample_image_name = "Pn_calliantha_20230523_0DB2_1"
display_yolo_segmentations(image_directory, yolo_annotation_directory, sample_image_name)
