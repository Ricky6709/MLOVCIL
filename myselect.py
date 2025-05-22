import numpy as np
import torch
import torchvision
from torch.utils.data import Subset, DataLoader
from PIL import Image
import random

# 设置随机种子，以便复现结果
random.seed(42)
# np.random.seed(42)
# torch.manual_seed(42)

class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, cifar_dataset, labeled_indices, target_new):
        self.cifar_dataset = cifar_dataset
        self.labeled_indices = labeled_indices
        self.target_new = target_new

    def __getitem__(self, index):
        img, label = self.cifar_dataset[self.labeled_indices[index]]
        target_new = self.target_new[self.labeled_indices[index]]
        return img, label, target_new

    def __len__(self):
        return len(self.labeled_indices)

    @property
    def data(self):
        return [img for img, _, _ in self]

    @property
    def targets(self):
        return [label for _, label, _ in self]

    @property
    def newtargets(self):
        return [new_label for _, _, new_label in self]

# 下载CIFAR-100数据集
cifar100_train = torchvision.datasets.CIFAR100(root='./datasets/cifar100/', train=True, download=True)

# 获取CIFAR-100的类别列表
classes = cifar100_train.classes

# 将类别分为有标签和无标签
labeled_classes = classes[:90]
unlabeled_classes = classes[90:]

# 随机选择两张有标签数据的图片
labeled_indices = []
for cls in labeled_classes:
    cls_indices = [i for i, label in enumerate(cifar100_train.targets) if label == classes.index(cls)]
    labeled_indices.extend(random.sample(cls_indices, 2))

# 创建新的标签信息
# target_new = [classes.index(cls) if i in labeled_indices else 100 for i in range(len(cifar100_train.targets))]
target_new = [cifar100_train.targets[i] if i in labeled_indices else 100 for i in range(len(cifar100_train.targets))]

# 按照原始数据集中的顺序对索引进行排序,将数据集分为有标签和无标签
labeled_indices.sort()
unlabeled_indices = [i for i in range(len(cifar100_train)) if i not in labeled_indices]
labeled_dataset = CustomDataset(cifar100_train, labeled_indices, target_new)
unlabeled_dataset = CustomDataset(cifar100_train, unlabeled_indices, target_new)

# 将数据集分为有标签和无标签
# labeled_dataset = CustomDataset(cifar100_train, labeled_indices, target_new)
# unlabeled_dataset = CustomDataset(cifar100_train, [i for i in range(len(cifar100_train)) if i not in labeled_indices], target_new)


# 使用DataLoader加载数据
labeled_dataloader = DataLoader(labeled_dataset, batch_size=64, shuffle=True)
unlabeled_dataloader = DataLoader(unlabeled_dataset, batch_size=64, shuffle=True)

# 通过labeled_dataset.data[0]获取第一张有标签图片
print("Image for the first labeled image::", labeled_dataset.data[0])

# first_labeled_img_data, first_labeled_img_label, first_labeled_img_target_new = labeled_dataset[0]
# first_labeled_img_pil.show()
print("Label for the first labeled image:", labeled_dataset.targets[0])
print("New label for the first labeled image:", labeled_dataset.newtargets[0])

# 通过unlabeled_dataset.data[0]获取第一张无标签图片的PIL对象
print("Label for the first unlabeled image:", unlabeled_dataset.targets[0])
print("New label for the first unlabeled image:", unlabeled_dataset.newtargets[0])
