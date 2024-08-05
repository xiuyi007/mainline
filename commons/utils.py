import difflib
import os
from collections import Counter
from tqdm import tqdm
import shutil
import matplotlib.pyplot as plt


def _extract_label(file):
    rst = []
    with open(file, 'r') as f:
        for line in f.readlines():
            label = line.strip().split()[0]
            rst.append(int(label))
    return rst


def extract_pic_byclass(label_dir, category):
    """
    找出标签目录下（存在了txt标签的目录）指定类别的文件
    :param label_dir:
    :param category:
    :return: list，指定类别的样本的路径列表
    """
    rst = []
    for file in tqdm(os.listdir(label_dir)):
        label_path = os.path.join(label_dir, file)
        if category in _extract_label(label_path):
            rst.append(label_path)
    return rst


def compare_file(f1, f2):
    with open(f1, 'r', encoding='UTF-8') as file1, open(f2, 'r', encoding='UTF-8') as file2:
        diff = difflib.ndiff(file1.readlines(), file2.readlines())
    print('\n'.join(diff))


def read_file2list(file):
    results = []
    with open(file, 'r', encoding='UTF-8') as f:
        rst = f.readlines()
        for line in rst:
            line = line.strip()
            results.append(line)
    return results


def class_inGroup(path):
    """
    对path路径下的label按照类别进行划分，得到counter()，key是类别，value是这个类别的paths，虽然能得到路径，但是由于//
    一个图片会有多个物体，所以一个label路径会多次出现在counter中，这意味着数据集中一个图片会多次使用。
    :param path:
    :return:
    """
    files = os.listdir(path)
    category_paths = Counter()
    for file in tqdm(files):
        label = os.path.join(path, file)
        with open(label, 'r') as f:
            for line in f.readlines():
                category_id = int(line.split()[0])
                if category_paths[category_id] == 0:
                    category_paths[category_id] = []
                category_paths[category_id].append(label)
    return category_paths


def _count_label(label_file, mapping=None):
    """
    统计一个yolo label文件中各个类别的数量
    :param label_file:
    :param mapping:
    :return: counter
    """
    c = Counter()
    with open(label_file) as f:
        lines = f.readlines()
        for line in lines:
            category_id = int(line.split()[0])
            if mapping:
                category_id = mapping[category_id]
            c[category_id] += 1
    return c

def class_count_bytxt(file, mapping=None):
    """
    统计yolo的txt文件指定的数据集的类被情况
    :param file:
    :param mapping:
    :return: Counter，包含了每个类别的数量
    """
    if mapping:
        mapping = read_file2list(mapping)
    with open(file, 'r') as f:
        c = Counter()
        lines = f.readlines()
        for line in lines:
            path = line.strip().replace('images', 'labels').replace('.jpg', '.txt')
            c += _count_label(path, mapping)
    return c



def class_count(path, class_mapping=None):
    """
        统计指定路径下（标签）各个类别的数量

        args:
            data_path: 类似这样的路径，'G:\science_data\datasets\RicePestsv3_tail\VOCdevkit\labels'
        return:
            先yield训练集的结果，再yield验证集的结果
            print(next(class_count(data_path)))
    """
    if class_mapping:
        class_mapping = read_file2list(class_mapping)
    label_files = [os.path.join(path, f) for f in os.listdir(path) if f.endswith('.txt')]
    # 统计每个类别的样本数量
    category_counts = Counter()

    for label_file in tqdm(label_files):
        category_counts += _count_label(label_file, class_mapping)

    # print(f"{t} data count: ", category_counts)
    return category_counts


def extract_tail_datasets(images_dir, labels_dir, new_images_dir, new_labels_dir, target_classes):
    """
        从原来的数据集中创建子数据，只由部分类别构成
    """
    # 创建新数据集目录
    os.makedirs(new_images_dir, exist_ok=True)
    os.makedirs(new_labels_dir, exist_ok=True)
    # 遍历标签文件，提取目标类别的数据
    for label_file in tqdm(os.listdir(labels_dir)):
        if label_file.endswith('.txt'):
            label_path = os.path.join(labels_dir, label_file)
            with open(label_path, 'r') as f:
                lines = f.readlines()

            # 过滤出包含目标类别的行
            filtered_lines = [line for line in lines if int(line.split()[0]) in target_classes]

            if filtered_lines:
                # 复制图像文件
                image_file = label_file.replace('.txt', '.jpg')
                image_path = os.path.join(images_dir, image_file)
                new_image_path = os.path.join(new_images_dir, image_file)
                shutil.copy(image_path, new_image_path)

                # 创建新的标签文件
                new_label_path = os.path.join(new_labels_dir, label_file)
                with open(new_label_path, 'w') as f:
                    f.writelines(filtered_lines)

    print("新数据集创建完成")
    g = class_count(new_labels_dir.split('train')[0])
    print("class count:", next(g))


def make_tail_datasets():
    old_img = 'G:\science_data\datasets\RicePestsv3\VOCdevkit\images\\train'
    old_labels = 'G:\science_data\datasets\RicePestsv3\VOCdevkit\labels\\train'
    new_img = 'G:\science_data\datasets\RicePestsv3_tail\VOCdevkit\images\\train'
    new_labels = 'G:\science_data\datasets\RicePestsv3_tail\VOCdevkit\labels\\train'
    extract_tail_datasets(old_img, old_labels, new_img, new_labels, [3, 4, 5, 6, 7, 8])


def visual_counts(*counter):
    """
    对Counter对象中的类别数量进行一个可视化呈现。
    :param counter:
    :return: None
    """
    for n, c in enumerate(counter):
        print(f'{n}: {c}')
        length = len(counter)
        plt.subplot(length, 1, n+1)
        plt.xlabel('class')
        plt.ylabel('count')
        plt.barh(c.keys(), c.values())
    plt.show()


if __name__ == '__main__':
    # label_mapping = read_file2list('G:\science_data\datasets\RicePestsv3\VOCdevkit\VOC2007\classes.txt')
    # print(label_mapping)
    # for category in tqdm(range(9)):
    #     value_mapping = label_mapping[category]
    #     paths = extract_pic_byclass('G:\science_data\datasets\RicePestsv3\VOCdevkit\VOC2007\labels', category)
    #     img_paths = []
    #     for path in paths:
    #         img_paths.append(path.replace('labels', 'images').replace('txt', 'jpg'))
    #     for img in img_paths:
    #         shutil.copyfile(img, os.path.join(f'G:\science_data\datasets\RicePestv3_category\\{value_mapping}', img.split('\\')[-1]))
    path = 'G:\science_data\datasets\RicePestsv3_easy\images'
    lines = os.listdir(path)
    lines = [line.strip().replace('images', 'labels').replace('jpg', 'txt') for line in lines]
    for line in lines:
        shutil.copyfile(os.path.join('G:\science_data\datasets\RicePestsv3\VOCdevkit\VOC2007\labels', line), f'G:\science_data\datasets\RicePestsv3_easy\\labels\\{line}')
