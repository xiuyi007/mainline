
import argparse
import os
from ultralytics import YOLO



def check_dataVersion(path):
    """检查yolo数据集的版本, 如cfg/datasets/pest_v3.yaml路径中提取v3

    Args:
        str (_type_): 数据路径
    """
    version = path.split(os.path.sep)[-1].split('_')[-1].split('.')[0]
    return version


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='YOLO Version argument')
    parser.add_argument('--data', type=str, default='cfg/datasets/pest_v3.yaml', help='dataset yaml')
    parser.add_argument('--model', type=str, default=None, help='yolo model')
    parser.add_argument('--epochs', type=int, default=20, help='epoch number')
    parser.add_argument('--imgsz', type=int, default=640, help='image size')
    parser.add_argument('--batch', type=int, default=32, help='batch size')
    parser.add_argument('--mpretrained', type=str, default='yolov10s.pt', help='pretrained model')
    # parser.print_help()
    opt = parser.parse_args()

    version = check_dataVersion(opt.data)
    project_dir = os.path.join('runs', 'detect', version)
    
    model = YOLO(opt.model).load(opt.mpretrained)
    model.train(data=opt.data, epochs=opt.epochs, imgsz=opt.imgsz, batch=opt.batch, project=project_dir)
