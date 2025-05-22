import json
import os
from pathlib import Path
from typing import Dict, Tuple
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
from PIL import Image
from torch.utils.data import DataLoader, Dataset

import numpy as np
import random
import pickle
import platform
import os
import torch
from matplotlib import pyplot as plt


from ram import get_transform


class VOC2012_handler(Dataset):
    def __init__(self, X, Y, input_size, transform=None):
        self.X = X
        self.Y = Y
        self.transform = get_transform(input_size)
        self.data_path = '../datasets/voc/VOCdevkit/VOC2012'

    def __getitem__(self, index):
        x = Image.open(self.data_path + '/JPEGImages/' + self.X[index]).convert('RGB')
        if self.transform is not None:
            x = self.transform(x)
        y = self.Y[index]
        return x, y

    def __len__(self):
        return len(self.X)


class COCO2014_handler(Dataset):
    def __init__(self, X, Y, input_size, data_path, transform=None):
        self.X = X
        self.Y = Y
        self.transform = get_transform(input_size)
        self.data_path = data_path

    def __getitem__(self, index):
        x = Image.open(self.data_path + '/' + self.X[index]).convert('RGB')
        if self.transform is not None:
            x = self.transform(x)
        y = self.Y[index]
        return x, y

    def __len__(self):
        return len(self.X)


class CUB_200_2011_handler(Dataset):
    def __init__(self, X, Y, input_size, transform=None):
        self.X = X
        self.Y = Y
        self.transform = get_transform(input_size)
        self.data_path = '../datasets/cub/CUB_200_2011/images'

    def __getitem__(self, index):
        x = Image.open(self.data_path + '/' + self.X[index]).convert('RGB')
        x = self.transform(x)
        y = self.Y[index]
        return x, y

    def __len__(self):
        return len(self.X)







torch.manual_seed(1)
np.random.seed(1)






def load_cifar100_G():  # 输入你“cifar-100-python”的路径

    with open("D:/code/data/cifar/cifar-100-python/train", 'rb') as f:
        data_train = pickle.load(f, encoding='latin1')  # 训练集，不同分类的数据，不同类别序号，
    with open("D:/code/data/cifar/cifar-100-python/test", 'rb') as f:
        data_test = pickle.load(f, encoding='latin1')  # 测试集，不同分类的数据，不同类别序号，
    with open("D:/code/data/cifar/cifar-100-python/meta", 'rb') as f:
        data_meta = pickle.load(f, encoding='latin1')  # 100分类与20分类的标签
    return data_train, data_test, data_meta



def load_cifar100():  # 输入你“cifar-100-python”的路径

    with open("./datasets/CIFAR100/train", 'rb') as f:
        data_train = pickle.load(f, encoding='latin1')  # 训练集，不同分类的数据，不同类别序号，
    with open("./datasets/CIFAR100/test", 'rb') as f:
        data_test = pickle.load(f, encoding='latin1')  # 测试集，不同分类的数据，不同类别序号，
    with open("./datasets/CIFAR100/meta", 'rb') as f:
        data_meta = pickle.load(f, encoding='latin1')  # 100分类与20分类的标签
    return data_train, data_test, data_meta


def read_data_cifar_100():
    random.seed(1)

    data_train, data_test, data_meta = load_cifar100()
    train_data = data_train['data'].reshape((data_train['data'].shape[0], 3, 32, 32))#.transpose((0,1,3,2))
    test_data = data_test['data'].reshape((data_test['data'].shape[0], 3, 32, 32))#.transpose((0,1,3,2))
    train_label = data_train["fine_labels"]
    test_label = data_test["fine_labels"]



    return train_data, train_label, test_data, test_label

class CIFAR100_handler(Dataset):
    def __init__(self, X, Y, input_size, transform=None):
        self.X = X
        self.Y = Y
        self.transform = get_transform(input_size)

    def __getitem__(self, index):
        # print(self.X[index].shape)
        x = Image.fromarray(np.uint8(self.X[index]).transpose((1, 2, 0)))
        # x = Image.open(self.X[index]).convert('RGB')
        x = self.transform(x)
        y = self.Y[index]
        return x, y

    def __len__(self):
        return len(self.X)



def data_gen_tf(train_set, train_label, test_set, test_label,input_size,batch_size):
    train_dataset = CIFAR100_handler(train_set, train_label, input_size)
    train_loader = DataLoader(dataset=train_dataset,
                              batch_size=batch_size,
                              shuffle=True)

    test_dataset = CIFAR100_handler(test_set, test_label, input_size)
    test_loader = DataLoader(dataset=test_dataset,
                             batch_size=batch_size,
                             shuffle=True)
    return train_loader, test_loader


def load_datasets(
        dataset: str,
        model_type: str,
        pattern: str,
        input_size: int,
        batch_size: int,
        num_workers: int
) -> Tuple[DataLoader, Dict]:
    dataset_root = "./datasets/" + dataset
    # Label system of tag2text contains duplicate tag texts, like
    # "train" (noun) and "train" (verb). Therefore, for tag2text, we use
    # `tagid` instead of `tag`.
    if model_type == "ram_plus" or model_type == "ram":
        tag_file = dataset_root + f"/{dataset}_ram_taglist.txt"
    else:
        tag_file = dataset_root + f"/{dataset}_tag2text_tagidlist.txt"

    with open(tag_file, "r", encoding="utf-8") as f:
        taglist = [line.strip() for line in f]

    # imglist = np.load(os.path.join(f'./datasets/{dataset}/', r'train'))

    train_data, train_label, test_data, test_label = read_data_cifar_100()




    if pattern == "train":
        loader, _ = data_gen_tf(train_data, train_label, test_data, test_label, input_size, batch_size)
    if pattern == "val":
        _, loader = data_gen_tf(train_data, train_label, test_data, test_label, input_size, batch_size)

    open_tag_des = dataset_root + f"/{dataset}_llm_tag_descriptions.json"
    if os.path.exists(open_tag_des):
        with open(open_tag_des, 'rb') as fo:
            tag_des = json.load(fo)

    else:
        tag_des = None
    info = {
        "taglist": taglist,
        # "imglist": imglist,
        # "annot_file": annot_file,
        # "img_root": img_root,
        "tag_des": tag_des
    }

    return loader, info
