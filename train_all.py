import argparse

from ultralytics import YOLO


if __name__ == '__main__':
    # Load a pre-trained YOLOv10n model
    parser = argparse.ArgumentParser(description='YOLO Version argument')
    parser.add_argument('--epochs', type=int, default=100, help='epochs number')
    opt = parser.parse_args()
    for v in [8, 10]:
        model = YOLO(f"yolov{v}s.pt")
        model.train(data="./cfg/datasets/pest_v4.yaml", epochs=opt.epochs, imgsz=640, batch=32)
