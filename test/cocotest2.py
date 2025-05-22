import numpy as np
import torch
from torchvision import transforms
from cocotest1 import CustomCocoDataset
from pycocotools.coco import COCO


def create_incremental_tasks(data_dir, labeled_number, n, batch_size=64):
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    transform = transforms.Compose([
        transforms.ToTensor(),
        normalize,
    ])

    # 注释文件路径
    # ann_file = f'{data_dir}/annotations/instances_train2017.json'
    ann_file = f'./annotations/image_info_test2017.json'

    # 加载 COCO 数据集
    coco = COCO(ann_file)

    # 获取所有类别 ID 和名称
    all_classes = sorted(coco.getCatIds())
    first_n_classes = set(all_classes[:n])

    # 用于增量学习任务分区的自定义数据集类
    class IncrementalCocoDataset(CustomCocoDataset):
        def __init__(self, root, annFile, visible_classes, transform=None):
            super(IncrementalCocoDataset, self).__init__(root, annFile, labeled_number, transform)
            self.visible_classes = visible_classes

        def __getitem__(self, index):
            img, labels, img_id = super(IncrementalCocoDataset, self).__getitem__(index)
            if -1 in labels:
                return img, None
            else:
                new_targets = [idx for idx, label in enumerate(labels) if label == 1]
                return img, new_targets

    # 创建任务 1 数据集，过滤掉不包含前 n 个可见类别的图像
    task1_dataset = IncrementalCocoDataset(
        root=f'test2017/test2017',
        annFile=ann_file,
        visible_classes=first_n_classes,
        transform=transform
    )

    task1_filtered_indices = []
    for idx in range(len(task1_dataset)):
        img, target = task1_dataset[idx]
        if target:
            task1_filtered_indices.append(idx)


    # 创建任务 2 数据集
    task2_dataset = CustomCocoDataset(
        root=f'test2017/test2017',
        annFile=ann_file,
        labeled_number=labeled_number,
        transform=transform
    )

    # 获取任务 1 的所有索引
    task1_all_indices = set(task1_filtered_indices)
    task2_remaining_indices = [idx for idx in range(len(task2_dataset)) if idx not in task1_all_indices]

    return task1_filtered_indices, task2_remaining_indices

def save_indices_to_npy(indices, file_prefix):
    np.save(f'{file_prefix}_indices.npy', indices)

def load_task_dataset(data_dir, ann_file, labeled_number, indices, transform):
    dataset = CustomCocoDataset(
        root=f'test2017/test2017',
        annFile=ann_file,
        labeled_number=labeled_number,
        transform=transform
    )
    filtered_dataset = torch.utils.data.Subset(dataset, indices)
    return filtered_dataset


# def save_data_to_npy(loader, file_prefix):
#     all_targets = []
#
#     for _, targets in loader:
#         all_targets.append(targets)
#
#     np.save(f'{file_prefix}_labels.npy', all_targets)


# 示例用法
if __name__ == "__main__":
    data_dir = './datasets'  # 修改为你的 COCO 数据集路径
    labeled_number = 80  # 可见类别数
    n = 20  # 任务 1 中的类别数量
    batch_size = 64


    task1_indices, task2_indices = create_incremental_tasks(data_dir, labeled_number, n, batch_size)

    # 保存任务 1 和任务 2 的索引
    save_indices_to_npy(task1_indices, 'task1e')
    save_indices_to_npy(task2_indices, 'task2e')

    # task1_filtered_dataset = torch.utils.data.Subset(task1_dataset, task1_indices)
    # task2_combined_dataset = torch.utils.data.Subset(task2_dataset, task2_indices)


    #
    #
    # # 创建 DataLoader
    # task1_loader = torch.utils.data.DataLoader(task1, batch_size=batch_size, shuffle=True,
    #                                            collate_fn=task1.dataset.collate_fn)
    # task2_loader = torch.utils.data.DataLoader(task2, batch_size=batch_size, shuffle=True,
    #                                            collate_fn=task2.dataset.collate_fn)

    # # 测试任务 1 中的一些样本
    # print("任务 1 样本:")
    # for images, classes in task1_loader:
    #     for cls in classes:
    #         print(cls)
    #     break
    #
    # # 测试任务 2 中的一些样本
    # print("任务 2 样本:")
    # for images, classes in task2_loader:
    #     for cls in classes:
    #         print(cls)
    #     break
