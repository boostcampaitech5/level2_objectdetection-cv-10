from ultralytics import YOLO

# Load YOLO v8 model
model = YOLO("yolov8l.pt")  # ! Load Pre-trained YOLO v8
# model = YOLO("yolov8l.yaml")  # ! Load YOLO v8 with scratch version -- Not trained

# Train YOLO v8
model.train(data="recycle.yaml", epochs=30)  # ! Augmentation 이미 존재
