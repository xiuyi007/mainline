from ultralytics import YOLO


def freeze_layer(trainer):
    model = trainer.model
    num_freeze = 11
    print(f"Freezing {num_freeze} layers")
    freeze = [f'model.{x}.' for x in range(num_freeze)]  # layers to freeze
    for k, v in model.named_parameters():
        v.requires_grad = True  # train all layers
        if any(x in k for x in freeze):
            print(f'freezing {k}')
            v.requires_grad = False
    print(f"{num_freeze} layers are freezer.")


def retraining(trainer):
    model = trainer.model
    reinitial = [f'model.{x}.' for x in range(12, 22)]
    for n, m in model.named_modules():
        if any(x in n for x in reinitial):
            if hasattr(m, "reset_parameters"):
                m.reset_parameters()
                print(f'reinitial {n}')

    num_freeze = 11
    print(f"Freezing {num_freeze} layers")
    freeze = [f'model.{x}.' for x in range(num_freeze)]  # layers to freeze
    for k, v in model.named_parameters():
        v.requires_grad = True  # train all layers
        if any(x in k for x in freeze):
            print(f'freezing {k}')
            v.requires_grad = False
    print(f"{num_freeze} layers are freezer.")


if __name__ == '__main__':
    model = YOLO('runs\detect\\yolov10s\weights\\best.pt')
    model.add_callback("on_train_start", freeze_layer)
    model.train(data="cfg/datasets/pest_v3.yaml", epochs=150, imgsz=640, batch=32, hsv_h=0.0, hsv_s=0.0, hsv_v=0.0, degrees=0.0, translate=0.0, scale=0.0, shear=0.0, perspective=0.0, flipud=0.0,fliplr=0.0, mosaic=0.0, mixup=0.0)