from ultralytics.models.yolo.detect import DetectionTrainer

from ultralytics import YOLO

# Load a model
model = YOLO('yolov8n.pt')
model.trainer.plot_training_labels()