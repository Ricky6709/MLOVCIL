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

        # 获取所有类别 ID，并选取前 labeled_number 个作为可见类别
        self.all_classes = sorted(self.coco.getCatIds())
        self.visible_classes = set(self.all_classes[:self.labeled_number])

    def __getitem__(self, index):
        coco = self.coco
        img_id = self.ids[index]
        ann_ids = coco.getAnnIds(imgIds=img_id)
        anns = coco.loadAnns(ann_ids)

        path = coco.loadImgs(img_id)[0]['file_name']
        img = Image.open(os.path.join(self.root, path)).convert('RGB')

        if self.transform is not None:
            img = self.transform(img)

        # 提取类别标签，如果类别不在可见类别中，设置为 -1
        classes = [ann['category_id'] if ann['category_id'] in self.visible_classes else -1 for ann in anns]

        return img, list(classes)

    def __len__(self):
        return len(self.ids)

    def collate_fn(self, batch):
        return tuple(zip(*batch))
