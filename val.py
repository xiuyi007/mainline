from ultralytics import YOLO
import argparse



if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='YOLO Version argument')
    parser.add_argument('--model', type=str, default=None, help='yolo model')
    # parser.print_help()

    opt = parser.parse_args()

    model = YOLO(opt.model)
    model.val(data='cfg/datasets/pest_v3.yaml')
