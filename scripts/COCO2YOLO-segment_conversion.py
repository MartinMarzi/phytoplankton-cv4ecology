from ultralytics.data.converter import convert_coco

convert_coco(labels_dir='../coco/annotations/', use_segments=True)