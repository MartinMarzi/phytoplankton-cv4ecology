# Convert bbox annotations from COCO to YOLO format
import os
import json
import shutil
from PIL import Image

def coco2yolo(coco_annotations_path)