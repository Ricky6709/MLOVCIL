from PIL import Image
import os
import os.path
import numpy as np
import sys
import argparse
import torch

if sys.version_info[0] == 2:
    import cPickle as pickle
else:
    import pickle
import torchvision
import torch.utils.data as data
from torchvision import transforms
# import itertools
# from torch.utils.data.sampler import Sampler
# from __future__ import print_function





class SELECTCIFAR100(torchvision.datasets.CIFAR100):

    def __init__(self, root, labeled=True, labeled_num=50, labeled_ratio=0.5, rand_number=0, transform=None,
                 target_transform=None,
                 download=False, unlabeled_idxs=None):
        super(SELECTCIFAR100, self).__init__(root, True, transform, target_transform, download)

        downloaded_list = self.train_list
        self.data = []
        self.targets = []
        self.target_new = []  # 新的变量
        # now load the picked numpy arrays
        for file_name, checksum in downloaded_list:
            file_path = os.path.join(self.root, self.base_folder, file_name)
            with open(file_path, 'rb') as f:
                entry = pickle.load(f, encoding='latin1')
                self.data.append(entry['data'])
                if 'labels' in entry:
                    self.targets.extend(entry['labels'])
                else:
                    self.targets.extend(entry['fine_labels'])

        self.data = np.vstack(self.data).reshape(-1, 3, 32, 32)
        self.data = self.data.transpose((0, 2, 3, 1))  # convert to HWC

        labeled_classes = range(labeled_num)
        np.random.seed(rand_number)

        if labeled:
            self.labeled_idxs, self.unlabeled_idxs = self.get_labeled_index(labeled_classes, labeled_ratio)
            self.shrink_data(self.labeled_idxs)
            self.targets_new = self.targets.copy()  # 将 targets_new 设置为 targets 的初始值
        else:
            self.shrink_data(unlabeled_idxs)
            self.targets_new = [100] * 100  # 将 targets_new 设置为一个大小为 100 的列表，所有元素为 100

    def get_labeled_index(self, labeled_classes, labeled_ratio):
        labeled_idxs = []
        unlabeled_idxs = []
        for idx, label in enumerate(self.targets):
            if label in labeled_classes and np.random.rand() < labeled_ratio:
                labeled_idxs.append(idx)
            else:
                unlabeled_idxs.append(idx)
                self.target_new.append(100)  # 对 unlabeled 的样本，将 target_new 设置为 100
        return labeled_idxs, unlabeled_idxs

    def shrink_data(self, idxs):
        targets = np.array(self.targets)
        self.targets_new = targets[idxs].tolist()  # 将 targets_new 设置为 targets 的一个子集
        self.targets = targets[idxs].tolist()
        self.data = self.data[idxs, ...]


# Dictionary of transforms
dict_transform = {
    'cifar_train': transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
    ]),
    'cifar_test': transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
    ])
}

# class TransformTwice:
#     def __init__(self, transform):
#         self.transform = transform
#
#     def __call__(self, inp):
#         out1 = self.transform(inp)
#         out2 = self.transform(inp)
#         return out1, out2



def main():
    parser = argparse.ArgumentParser(description='select')
    # parser.add_argument('--milestones', nargs='+', type=int, default=[140, 180])
    parser.add_argument('--labeled-num', default=90, type=int)
    parser.add_argument('--labeled-ratio', default=0.005, type=float)
    parser.add_argument('--seed', type=int, default=1, metavar='S', help='random seed (default: 1)')
    parser.add_argument('-b', '--batch-size', default=512, type=int,
                    metavar='N',
                    help='mini-batch size')
    args = parser.parse_args()
    args.cuda = torch.cuda.is_available()
    device = torch.device("cuda" if args.cuda else "cpu")



    train_label_set = SELECTCIFAR100(root='./datasets/cifar100/', labeled=True,
                                                 labeled_num=args.labeled_num, labeled_ratio=args.labeled_ratio,
                                                 download=True)
    train_unlabel_set = SELECTCIFAR100(root='./datasets/cifar100/', labeled=False, labeled_num=args.labeled_num,
                                                   labeled_ratio=args.labeled_ratio, download=True,
                                                    unlabeled_idxs=train_label_set.unlabeled_idxs)
    test_set = SELECTCIFAR100(root='./datasets/cifar100/', labeled=False, labeled_num=args.labeled_num,
                                          labeled_ratio=args.labeled_ratio, download=True,
                                          transform=dict_transform['cifar_test'],
                                          unlabeled_idxs=train_label_set.unlabeled_idxs)



    labeled_len = len(train_label_set)
    unlabeled_len = len(train_unlabel_set)
    labeled_batch_size = int(args.batch_size * labeled_len / (labeled_len + unlabeled_len))

    # Initialize the splits
    train_label_loader = torch.utils.data.DataLoader(train_label_set, batch_size=labeled_batch_size, shuffle=True, num_workers=2, drop_last=True)
    train_unlabel_loader = torch.utils.data.DataLoader(train_unlabel_set, batch_size=args.batch_size - labeled_batch_size, shuffle=True, num_workers=2, drop_last=True)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=100, shuffle=False, num_workers=1)

    print(train_unlabel_set.targets_new[2])

if __name__ == '__main__':
    main()