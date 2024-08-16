import cv2
import numpy as np
import sys
sys.path.append('../ultralytics')
from ultralytics import YOLO


def sliding_window(image, stepSize, windowSize):
    # 遍历图像中的每个窗口
    for y in range(0, image.shape[0], stepSize):
        for x in range(0, image.shape[1], stepSize):
            # 提取当前窗口
            yield (x, y, image[y:y + windowSize[1], x:x + windowSize[0]])

# 示例：在一张图像上应用滑动窗口
image = cv2.imread('G:\science_data\datasets\RicePestsv1\VOCdevkit\images/train/147645_22-07-04-00-02-10_1.jpg')
winW, winH = 1000, 1000
stepSize = 1000
source = []
for (x, y, window) in sliding_window(image, stepSize=stepSize, windowSize=(winW, winH)):
    # 在此处可以进行目标检测处理
    source.append(window)
model = YOLO('../runs\detect/v3\yolov10s\weights/best.pt')
results = model(source)
for result in results:
    boxes = result.boxes  # Boxes object for bounding box outputs
    masks = result.masks  # Masks object for segmentation masks outputs
    keypoints = result.keypoints  # Keypoints object for pose outputs
    probs = result.probs  # Probs object for classification outputs
    obb = result.obb  # Oriented boxes object for OBB outputs
    result.show()  # display to screen
    result.save(filename="result.jpg")  # save to disk