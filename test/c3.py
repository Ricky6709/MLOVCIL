import numpy as np
import torch
from torchvision import transforms
from c2 import CustomCocoDataset,load_task_dataset


def create_data_loaders(data_dir, labeled_number, batch_size):
    # 加载任务 1 和任务 2 的索引
    task1_loaded_indices = np.load('task1v_indices.npy')
    task2_loaded_indices = np.load('task2v_indices.npy')

    # ann_file = f'{data_dir}/annotations/instances_train2017.json'
    ann_file = f'./annotations/instances_val2017.json'

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # 加载任务 1 和任务 2 的数据集
    task1_loaded = load_task_dataset(data_dir, ann_file, labeled_number, task1_loaded_indices, transform)
    task2_loaded = load_task_dataset(data_dir, ann_file, labeled_number, task2_loaded_indices, transform)

    # 创建 DataLoader
    task1_loader = torch.utils.data.DataLoader(task1_loaded, batch_size=batch_size, shuffle=True,
                                               collate_fn=task1_loaded.dataset.collate_fn)
    task2_loader = torch.utils.data.DataLoader(task2_loaded, batch_size=batch_size, shuffle=True,
                                               collate_fn=task2_loaded.dataset.collate_fn)

    return task1_loader, task2_loader

# 示例用法
if __name__ == "__main__":
    data_dir = './datasets'  # 修改为你的 COCO 数据集路径
    labeled_number = 40  # 可见类别数
    batch_size = 64

    task1_loader, task2_loader = create_data_loaders(data_dir, labeled_number, batch_size)


    # 测试任务 1 中的一些样本
    print("任务 1 样本:")
    for images, labels, img_ids in task1_loader:
        print(labels)
        print(img_ids)
        break

    # 测试任务 2 中的一些样本
    print("任务 2 样本:")
    for images, labels, img_ids in task2_loader:
        print(labels)
        print(img_ids)
        break


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
