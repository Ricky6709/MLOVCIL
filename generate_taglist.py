# import json
import os
import numpy as np
# from PIL import Image
import pickle

def load_data():  # 输入你“cifar-100-python”的路径

    with open("D:/code/data/cifar/cifar-100-python/train", 'rb') as f:
        data_train = pickle.load(f, encoding='latin1')  # 训练集，不同分类的数据，不同类别序号，
    with open("D:/code/data/cifar/cifar-100-python/test", 'rb') as f:
        data_test = pickle.load(f, encoding='latin1')  # 测试集，不同分类的数据，不同类别序号，
    with open("D:/code/data/cifar/cifar-100-python/meta", 'rb') as f:
        data_meta = pickle.load(f, encoding='latin1')  # 100分类与20分类的标签
    return data_train, data_test, data_meta



_, _, imglist = load_data()
for key in imglist:
    if key == 'fine_label_names':
        for i in range(len(imglist[key])):
            print(imglist[key][i])
