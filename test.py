from ultralytics import YOLO

# Load a model
model = YOLO("yolov8n-seg.pt")  # load an official model
# model = YOLO("path/to/best.pt")  # load a custom model

# Predict with the model
results = model("G:\science_data\datasets\RicePestsv1\VOCdevkit\images/train/147645_22-07-04-00-02-10_1.jpg")  # predict on an image