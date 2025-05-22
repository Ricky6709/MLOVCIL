import torch
from torchvision import datasets, transforms
from pycocotools.coco import COCO
import numpy as np


class CustomCocoDataset(datasets.CocoDetection):
    def __init__(self, root, annFile, labeled_number, transform=None):
        super(CustomCocoDataset, self).__init__(root, annFile, transform)
        self.coco = COCO(annFile)

        # 获取所有类别ID和类别名称
        all_classes = sorted(self.coco.getCatIds())
        self.visible_classes = set(all_classes[:labeled_number])
        self.invisible_classes = set(all_classes[labeled_number:])

    def __getitem__(self, index):
        img, target = super(CustomCocoDataset, self).__getitem__(index)
        classes = []
        for t in target:
            if t['category_id'] in self.visible_classes:
                classes.append(t['category_id'])
            else:
                classes.append(-1)
        return img, classes

    def collate_fn(self, batch):
        return tuple(zip(*batch))


def get_coco_dataloader(data_dir, labeled_number, batch_size=4, shuffle=True, transform=None):
    if transform is None:
        transform = transforms.Compose([
            transforms.ToTensor()
        ])

    # 注释文件路径
    # ann_file = f'{data_dir}/annotations/instances_train2017.json'
    ann_file = f'./annotations/instances_train2017.json'

    # 加载自定义数据集
    dataset = CustomCocoDataset(root=f'./train2017/train2017', annFile=ann_file, labeled_number=labeled_number,
                                transform=transform)

    # 创建DataLoader
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle,
                                             collate_fn=dataset.collate_fn)

    return dataloader


# 使用示例
if __name__ == "__main__":
    data_dir = './datasets'  # 修改为你的CoCo数据集路径
    labeled_number = 40  # 可见类的数量
    batch_size = 16

    dataloader = get_coco_dataloader(data_dir, labeled_number, batch_size)

    # 打印一些样本以测试

    for images, classes in dataloader:
        for cls in classes:
            print(cls)
        break
