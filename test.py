
from ultralytics import YOLO

# Load a model
model = YOLO("E:\code\YOLO\mainline\\runs\detect\yolov10s\weights\\best.pt")  # pretrained YOLOv8n model

# Run batched inference on a list of images
results = model(["G:\science_data\datasets\RicePestv3_category\YuMiMing\\147700_22-08-25-03-22-10_1_3072_3000_1024.jpg"])  # return a list of Results objects

# Process results list
for result in results:
    boxes = result.boxes  # Boxes object for bounding box outputs
    masks = result.masks  # Masks object for segmentation masks outputs
    keypoints = result.keypoints  # Keypoints object for pose outputs
    probs = result.probs  # Probs object for classification outputs
    obb = result.obb  # Oriented boxes object for OBB outputs
    result.show()  # display to screen
    result.save(filename="result.jpg")  # save to disk