import argparse
import os

from ultralytics import YOLO


if __name__ == '__main__':
    # Load a pre-trained YOLOv10n model
    parser = argparse.ArgumentParser(description='YOLO Version argument')
    parser.add_argument('--epochs', type=int, default=100, help='epochs number')
    opt = parser.parse_args()
    model_dir = 'cfg/models/'
    i = 0
    for dir_path, dir_name, files in os.walk(model_dir):
        for file in files:
            if not file.startswith('yolo'):
                continue
            name = os.path.join(dir_path, file)
            model = YOLO(name).load('yolov10s.pt')
            model.train(data="cfg/datasets/pest_v3.yaml", epochs=opt.epochs, imgsz=640, batch=32)
