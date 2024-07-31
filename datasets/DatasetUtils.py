import os
import random

# 设定图片目录
image_dir = 'G:\science_data\datasets\RicePestsv3\VOCdevkit\VOC2007\images'

# 获取所有的图片文件名
images = [f for f in os.listdir(image_dir) if f.endswith(('.jpg', '.jpeg', '.png'))]

# 打乱图片顺序
random.shuffle(images)

# 设定划分比例
train_ratio = 0.7
val_ratio = 0.2
test_ratio = 0.1

# 计算各数据集的大小
train_size = int(len(images) * train_ratio)
val_size = int(len(images) * val_ratio)
test_size = len(images) - train_size - val_size

# 划分数据集
train_images = images[:train_size]
val_images = images[train_size:train_size + val_size]
test_images = images[train_size + val_size:]

# 写入文件
def write_paths_to_file(paths, file_name):
    with open(file_name, 'w') as f:
        for path in paths:
            f.write(f"{image_dir}/{path}\n")

# 写入train.txt, val.txt, test.txt
write_paths_to_file(train_images, 'RicePestV4/train.txt')
write_paths_to_file(val_images, 'RicePestV4/val.txt')
write_paths_to_file(test_images, 'RicePestV4/test.txt')

print(f"训练集大小: {len(train_images)}")
print(f"验证集大小: {len(val_images)}")
print(f"测试集大小: {len(test_images)}")
