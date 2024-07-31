
import argparse

from ultralytics import YOLO

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='YOLO Version argument')
    parser.add_argument('--model', type=str, default=None, help='yolo model')
    parser.add_argument('--epochs', type=int, default=20, help='epoch number')
    # parser.print_help()

    opt = parser.parse_args()
    model = YOLO(opt.model).load('yolov10s.pt')
    model.train(data="cfg/datasets/pest_v3.yaml", epochs=opt.epochs, imgsz=640, batch=32)
