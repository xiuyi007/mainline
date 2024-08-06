
import argparse

from ultralytics import YOLO

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='YOLO Version argument')
    parser.add_argument('--data', type=str, default='cfg/datasets/pest_v3.yaml', help='dataset yaml')
    parser.add_argument('--model', type=str, default=None, help='yolo model')
    parser.add_argument('--epochs', type=int, default=20, help='epoch number')
    parser.add_argument('--imgsz', type=int, default=640, help='image size')
    parser.add_argument('--batch', type=int, default=32, help='batch size')
    # parser.print_help()

    opt = parser.parse_args()
    model = YOLO(opt.model).load('yolov10s.pt')
    model.train(data=opt.yaml, epochs=opt.epochs, imgsz=opt.imgsz, batch=opt.batch)
