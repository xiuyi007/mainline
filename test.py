
import argparse

from ultralytics import YOLO

if __name__ == '__main__':
    model = YOLO('runs/detect/train/weights/best.pt')
    model.export(format='onnx', opset=11)
