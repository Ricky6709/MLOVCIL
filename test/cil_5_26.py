import numpy as np
import torch
from torchvision import transforms
from coco_5_26 import CustomCocoDataset
from pycocotools.coco import COCO


def create_incremental_tasks(data_dir, labeled_number, n, batch_size=4):
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    transform = transforms.Compose([
        transforms.ToTensor(),
        normalize,
    ])

    # 注释文件路径
    # ann_file = f'{data_dir}/annotations/instances_train2017.json'
    ann_file = f'./annotations/instances_train2017.json'

    # 加载CoCo数据集
    coco = COCO(ann_file)

    # 获取所有类别ID和类别名称
    all_classes = sorted(coco.getCatIds())
    first_n_classes = set(all_classes[:n])

    # 自定义数据集类，用于增量学习任务划分
    class IncrementalCocoDataset(CustomCocoDataset):
        def __init__(self, root, annFile, visible_classes, transform=None):
            super(IncrementalCocoDataset, self).__init__(root, annFile, labeled_number, transform)
            self.visible_classes = visible_classes

        def __getitem__(self, index):
            img, classes = super(CustomCocoDataset, self).__getitem__(index)
            print(classes)
            new_targets = []
            for cls in classes:
                if any(t not in self.visible_classes for t in cls):
                    return img, None
                new_targets.extend([t for t in cls if t in self.visible_classes])

            return img, new_targets
            # for cls in classes:
            #     for t in cls:
            #         if t in self.invisible_classes:
            #             new_target = t
            # # new_target = [t for t in classes if t in self.visible_classes]
            #         else:
            #             return img, None
            #         # if not new_target:  # 如果过滤后没有类别，返回 img 和 None
            #         #     return img, None
            #         return img, new_target

    # 创建任务一的数据集，过滤掉不包含前n个可见类的图像
    task1_dataset = IncrementalCocoDataset(
        # root=f'{data_dir}/train2017',
        root=f'./train2017/train2017',
        annFile=ann_file,
        visible_classes=first_n_classes,
        transform=transform
    )

    task1_filtered_indices = []
    for idx in range(min(sample_size, len(task1_dataset))):
        img, target = task1_dataset[idx]
        if target:
            task1_filtered_indices.append(idx)

    task1_filtered_dataset = torch.utils.data.Subset(task1_dataset, task1_filtered_indices)

    # 创建任务二的数据集
    task2_dataset = CustomCocoDataset(
        # root=f'{data_dir}/train2017',
        root=f'./train2017/train2017',
        annFile=ann_file,
        labeled_number=labeled_number,
        transform=transform
    )

    # 获取任务一中所有的图片索引
    task1_all_indices = set(task1_filtered_indices)
    task2_remaining_indices = [idx for idx in range(len(task2_dataset)) if idx not in task1_all_indices]
    task2_combined_dataset = torch.utils.data.Subset(task2_dataset, task2_remaining_indices)

    return task1_filtered_dataset, task2_combined_dataset

    # # 创建DataLoader
    # task1_loader = torch.utils.data.DataLoader(task1_filtered_dataset, batch_size=batch_size, shuffle=True,
    #                                            collate_fn=task1_filtered_dataset.dataset.collate_fn)
    # task2_loader = torch.utils.data.DataLoader(task2_combined_dataset, batch_size=batch_size, shuffle=True,
    #                                            collate_fn=task2_combined_dataset.dataset.collate_fn)

    # return task1_loader, task2_loader


def save_data_to_npy(loader, file_prefix):
    all_targets = []

    for _, targets in loader:
        all_targets.append(targets)

    np.save(f'{file_prefix}_labels.npy', all_targets)



# 使用示例
if __name__ == "__main__":
    data_dir = './datasets'  # 修改为你的CoCo数据集路径
    labeled_number = 40  # 可见类的数量
    n = 20  # 任务一中的类别数量
    batch_size = 64
    sample_size = 1000

    # task1_loader, task2_loader = create_incremental_tasks(data_dir, labeled_number, n, batch_size)

    task1, task2 = create_incremental_tasks(data_dir, labeled_number, n, batch_size)

    # 创建DataLoader
    task1_loader = torch.utils.data.DataLoader(task1, batch_size=batch_size, shuffle=True,
                                               collate_fn=task1.dataset.collate_fn)
    task2_loader = torch.utils.data.DataLoader(task2, batch_size=batch_size, shuffle=True,
                                               collate_fn=task2.dataset.collate_fn)


    # # 保存任务一的数据
    # print("Saving Task 1 data...")
    # save_data_to_npy(task1_loader, 'task1')
    # print("Task 1 data saved.")
    #
    # # 保存任务二的数据
    # print("Saving Task 2 data...")
    # save_data_to_npy(task2_loader, 'task2')
    # print("Task 2 data saved.")

    # 打印任务一的一些样本以测试
    print("Task 1 samples:")
    for images, classes in task1_loader:
        for cls in classes:
            print(cls)
        break

    # 打印任务二的一些样本以测试
    print("Task 2 samples:")
    for images, classes in task2_loader:
        for cls in classes:
            print(cls)
        break
