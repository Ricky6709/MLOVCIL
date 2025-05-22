import torch
from pycocotools.coco import COCO
from PIL import Image
import os


class CustomCocoDataset(torch.utils.data.Dataset):
    def __init__(self, root, annFile, labeled_number, transform=None):
        self.root = root
        self.coco = COCO(annFile)
        self.ids = list(self.coco.imgs.keys())
        self.labeled_number = labeled_number
        self.transform = transform

        # 获取所有类别 ID，并生成类别 ID 到索引的映射
        self.all_categories = sorted(self.coco.getCatIds())
        self.category_id_to_index = {cat_id: idx for idx, cat_id in enumerate(self.all_categories)}

        # 选取前 labeled_number 个类别作为可见类别
        self.visible_classes = set(self.all_categories[:labeled_number])
        self.invisible_classes = set(self.all_categories[labeled_number:])

    def __getitem__(self, index):
        coco = self.coco
        img_id = self.ids[index]
        ann_ids = coco.getAnnIds(imgIds=img_id)
        anns = coco.loadAnns(ann_ids)

        path = coco.loadImgs(img_id)[0]['file_name']
        img = Image.open(os.path.join(self.root, path)).convert('RGB')

        if self.transform is not None:
            img = self.transform(img)

        # 初始化标签矩阵为 0
        labels = [0] * len(self.all_categories)

        # 遍历每个注释并更新标签矩阵
        for ann in anns:
            category_id = ann['category_id']
            if category_id in self.visible_classes:
                labels[self.category_id_to_index[category_id]] = 1
            elif category_id in self.invisible_classes:
                labels[self.category_id_to_index[category_id]] = -1

        return img, labels, img_id

    def __len__(self):
        return len(self.ids)

    def collate_fn(self, batch):
        img, labels, img_ids = zip(*batch)
        return img, labels, img_ids
