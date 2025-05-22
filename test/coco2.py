import numpy as np
import torch
from torchvision import transforms
from c2 import CustomCocoDataset, load_task_dataset

def create_data_loaders(data_dir, labeled_number, batch_size):
    # Load indices for Task 1 and Task 2
    task1_loaded_indices = np.load('task1v_indices.npy')
    task2_loaded_indices = np.load('task2v_indices.npy')

    # Annotation file path
    # ann_file = f'{data_dir}/annotations/instances_train2017.json'
    ann_file = f'./annotations/instances_val2017.json'

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Load datasets for Task 1 and Task 2
    task1_loaded = load_task_dataset(data_dir, ann_file, labeled_number, task1_loaded_indices, transform)
    task2_loaded = load_task_dataset(data_dir, ann_file, labeled_number, task2_loaded_indices, transform)

    # Create DataLoader
    task1_loader = torch.utils.data.DataLoader(task1_loaded, batch_size=batch_size, shuffle=True,
                                               collate_fn=task1_loaded.dataset.collate_fn)
    task2_loader = torch.utils.data.DataLoader(task2_loaded, batch_size=batch_size, shuffle=True,
                                               collate_fn=task2_loaded.dataset.collate_fn)

    return task1_loader, task2_loader

def save_task_data(loader, labels_path, images_path):
    all_labels = []
    all_img_ids = []

    for images, labels, img_ids in loader:
        all_labels.extend(labels)
        all_img_ids.extend(img_ids)

    # Convert lists to numpy arrays
    all_labels = np.array(all_labels)
    all_img_ids = np.array(all_img_ids)

    # Save to .npy files
    np.save(labels_path, all_labels)
    np.save(images_path, all_img_ids)


if __name__ == "__main__":
    data_dir = './datasets'  # Adjust to your COCO dataset path
    labeled_number = 40  # Number of visible classes
    batch_size = 64

    task1_loader, task2_loader = create_data_loaders(data_dir, labeled_number, batch_size)

    # Save Task 1 labels and image IDs
    save_task_data(task1_loader, 'formatted_task1v_labels.npy', 'formatted_task1v_image.npy')

    # Save Task 2 labels and image IDs
    save_task_data(task2_loader, 'formatted_task2v_labels.npy', 'formatted_task2v_image.npy')

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
