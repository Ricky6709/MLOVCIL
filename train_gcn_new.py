from argparse import ArgumentParser
from pathlib import Path
from typing import Dict, List, Optional, TextIO, Tuple

import torch
from PIL import Image, UnidentifiedImageError
from torch import Tensor
from torch.nn import Module, Parameter
from torch.nn.functional import relu, sigmoid
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import torch.nn.functional as F
import os
import json
import numpy as np
import torch.nn as nn
import sys
import torch.backends.cudnn as cudnn
import torch.distributed as dis
import random
import clip

from ram import get_transform
from ram.models import ram_plus, ram, tag2text, cls_network
from ram.utils import build_openset_llm_label_embedding, build_openset_label_embedding, get_mAP, get_PR
from ram.utils.metrics import get_mAP_ua
from utils import step_lr_schedule, get_rank

from losses import *

device = "cuda" if torch.cuda.is_available() else "cpu"


model_clip, preprocess = clip.load("ViT-B/32", device=device)


class _Dataset(Dataset):
    def __init__(self, imglist, input_size):
        self.imglist = imglist
        self.transform = get_transform(input_size)

    def __len__(self):
        return len(self.imglist)

    def __getitem__(self, index):
        try:
            img = Image.open(self.imglist[index] + ".jpg")
        except (OSError, FileNotFoundError, UnidentifiedImageError):
            img = Image.new('RGB', (10, 10), 0)
            print("Error loading image:", self.imglist[index])
        return self.transform(img)


class VOC2012_handler(Dataset):
    def __init__(self, X, Y, input_size, data_path, transform=None):
        self.X = X
        self.Y = Y
        self.transform = get_transform(input_size)
        self.data_path = data_path

    def __getitem__(self, index):
        img_path = os.path.join(self.data_path, self.X[index][0] if isinstance(self.X[index], np.ndarray)
        else self.X[index])
        x = Image.open(img_path).convert('RGB')
        if self.transform is not None:
            x = self.transform(x)
        y = self.Y[index]
        return x, y

    def __len__(self):
        return len(self.X)


class COCO2014_handler(Dataset):
    def __init__(self, X, Y, input_size, data_path, transform=None):
        self.X = X
        self.Y = Y
        self.transform = get_transform(input_size)
        self.data_path = data_path

    def __getitem__(self, index):
        img_path = os.path.join(self.data_path, self.X[index][0] if isinstance(self.X[index], np.ndarray)
        else self.X[index])
        x = Image.open(img_path).convert('RGB')
        if self.transform is not None:
            x = self.transform(x)
        y = self.Y[index]
        return x, y

    def __len__(self):
        return len(self.X)


def parse_args():
    parser = ArgumentParser()
    # model
    parser.add_argument("--model-type",
                        type=str,
                        choices="ram",
                        required=True)
    parser.add_argument("--checkpoint",
                        type=str,
                        required=True)
    parser.add_argument("--backbone",
                        type=str,
                        choices=("swin_l", "swin_b"),
                        default=None,
                        help="If `None`, will judge from `--model-type`")
    parser.add_argument("--open-set",
                        type=bool,
                        # action="store_true",
                        default=True,
                        help=(
                            "Treat all categories in the taglist file as "
                            "unseen and perform open-set classification. Only "
                            "works with RAM."
                        ))
    # data
    parser.add_argument("--dataset",
                        type=str,
                        choices=(
                            "voc",
                            "coco",
                        ),
                        required=True)
    # parser.add_argument("--input-size",
    #                     type=int,
    #                     default=384)
    parser.add_argument("--input-size",
                        type=int,
                        default=224)
    # threshold
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--threshold",
                       type=float,
                       default=None,
                       help=(
                           "Use custom threshold for all classes. Mutually "
                           "exclusive with `--threshold-file`. If both "
                           "`--threshold` and `--threshold-file` is `None`, "
                           "will use a default threshold setting."
                       ))
    group.add_argument("--threshold-file",
                       type=str,
                       default=None,
                       help=(
                           "Use custom class-wise thresholds by providing a "
                           "text file. Each line is a float-type threshold, "
                           "following the order of the tags in taglist file. "
                           "See `ram/data/ram_tag_list_threshold.txt` as an "
                           "example. Mutually exclusive with `--threshold`. "
                           "If both `--threshold` and `--threshold-file` is "
                           "`None`, will use default threshold setting."
                       ))
    # miscellaneous
    parser.add_argument("--output-dir", type=str, default="./SPMLUA_outputs")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument('--seed', default=42, type=int)

    args = parser.parse_args()

    # post process and validity check
    args.model_type = args.model_type.lower()

    assert not (args.model_type == "tag2text" and args.open_set)

    if args.backbone is None:
        args.backbone = "swin_l" if args.model_type == "ram_plus" or args.model_type == "ram" else "swin_b"

    return args


def load_spml_datasets(
        dataset: str,
        model_type: str,
        pattern: str,
        input_size: int,
        batch_size: int,
        num_workers: int
) -> Tuple[DataLoader, Dict]:
    dataset_root = str(Path(__file__).resolve().parent / "datasets")
    img_root = dataset_root + "/imgs"
    # Label system of tag2text contains duplicate tag texts, like
    # "train" (noun) and "train" (verb). Therefore, for tag2text, we use
    # `tagid` instead of `tag`.
    if model_type == "ram_plus" or model_type == "ram":
        if pattern == "task1t" or pattern == "task1v":
            tag_file = dataset_root + f"/{dataset}1/{dataset}_ram_taglist.txt"
            imglist = np.load(os.path.join(f'./datasets/{dataset}/npy', f'formatted_{pattern}_images.npy'))
        if pattern == "task2t" or pattern == "task2v":
            tag_file = dataset_root + f"/{dataset}2/{dataset}_ram_taglist.txt"
            imglist = np.load(os.path.join(f'./datasets/{dataset}/npy', f'formatted_{pattern}_images.npy'))
    else:
        tag_file = dataset_root + f"/{dataset}_tag2text_tagidlist.txt"
        imglist = np.load(os.path.join(f'./datasets/{dataset}/npy', f'formatted_task1t_images.npy'))

    with open(tag_file, "r", encoding="utf-8") as f:
        taglist = [line.strip() for line in f]

    # imglist = np.load(os.path.join(f'./datasets/{dataset}/npy', f'formatted_{pattern}_images.npy'))

    if dataset == "voc":
        train_dataset1 = VOC2012_handler(X=np.load(os.path.join('./datasets/voc/npy', 'formatted_task1t_images.npy')),
                                         Y=np.load(os.path.join('./datasets/voc/npy', 'formatted_task1t_labels.npy')),
                                         data_path='./datasets/voc/VOCdevkit/VOC2012/JPEGImages',
                                         input_size=input_size)
        test_dataset1 = VOC2012_handler(X=np.load(os.path.join('./datasets/voc/npy', 'formatted_task1v_images.npy')),
                                        Y=np.load(os.path.join('./datasets/voc/npy', 'formatted_task1v_labels.npy')),
                                         data_path='./datasets/voc/VOCdevkit/VOC2012/JPEGImages',
                                         input_size=input_size)
        train_dataset2 = VOC2012_handler(X=np.load(os.path.join('./datasets/voc/npy', 'formatted_task2t_images.npy')),
                                         Y=np.load(os.path.join('./datasets/voc/npy', 'formatted_task2t_labels.npy')),
                                         data_path='./datasets/voc/VOCdevkit/VOC2012/JPEGImages',
                                         input_size=input_size)
        test_dataset2 = VOC2012_handler(X=np.load(os.path.join('./datasets/voc/npy', 'formatted_task2v_images.npy')),
                                        Y=np.load(os.path.join('./datasets/voc/npy', 'formatted_task2v_labels.npy')),
                                         data_path='./datasets/voc/VOCdevkit/VOC2012/JPEGImages',
                                         input_size=input_size)
    elif dataset == "coco":
        train_dataset1 = COCO2014_handler(X=np.load(os.path.join('./datasets/coco/npy', 'formatted_task1t_images.npy')),
                                         Y=np.load(os.path.join('./datasets/coco/npy', 'formatted_task1t_labels.npy')),
                                         data_path='./datasets/coco',
                                         input_size=input_size)
        test_dataset1 = COCO2014_handler(X=np.load(os.path.join('./datasets/coco/npy', 'formatted_task1v_images.npy')),
                                        Y=np.load(os.path.join('./datasets/coco/npy', 'formatted_task1v_labels.npy')),
                                        data_path='./datasets/coco',
                                        input_size=input_size)
        train_dataset2 = COCO2014_handler(X=np.load(os.path.join('./datasets/coco/npy', 'formatted_task2t_images.npy')),
                                         Y=np.load(os.path.join('./datasets/coco/npy', 'formatted_task2t_labels.npy')),
                                         data_path='./datasets/coco',
                                         input_size=input_size)
        test_dataset2 = COCO2014_handler(X=np.load(os.path.join('./datasets/coco/npy', 'formatted_task2v_images.npy')),
                                        Y=np.load(os.path.join('./datasets/coco/npy', 'formatted_task2v_labels.npy')),
                                        data_path='./datasets/coco',
                                        input_size=input_size)
    # elif dataset == "cub":
    #     train_dataset = CUB_200_2011_handler(X=np.load(os.path.join('./datasets/cub', 'formatted_train_images.npy')),
    #                                          Y=np.load(os.path.join('./datasets/cub', 'formatted_train_labels.npy')),
    #                                          input_size=input_size)
    #     test_dataset = CUB_200_2011_handler(X=np.load(os.path.join('./datasets/cub', 'formatted_val_images.npy')),
    #                                         Y=np.load(os.path.join('./datasets/cub', 'formatted_val_labels.npy')),
    #                                         input_size=input_size)
    # elif dataset == "nus":
    #     train_dataset = NUS_WIDE_handler(X=np.load(os.path.join('./datasets/nus', 'formatted_train_images.npy')),
    #                                      Y=np.load(os.path.join('./datasets/nus', 'formatted_train_labels_obs.npy')),
    #                                      input_size=input_size)
    #     test_dataset = NUS_WIDE_handler(X=np.load(os.path.join('./datasets/nus', 'formatted_val_images.npy')),
    #                                     Y=np.load(os.path.join('./datasets/nus', 'formatted_val_labels.npy')),
    #                                     input_size=input_size)

    if pattern == "task1t":
        loader = DataLoader(
            dataset=train_dataset1,
            shuffle=True,
            drop_last=False,
            pin_memory=True,
            batch_size=batch_size,
            num_workers=num_workers
        )
    if pattern == "task1v":
        loader = DataLoader(
            dataset=test_dataset1,
            shuffle=False,
            drop_last=False,
            pin_memory=True,
            batch_size=batch_size,
            num_workers=num_workers
        )
    if pattern == "task2t":
        loader = DataLoader(
            dataset=train_dataset2,
            shuffle=True,
            drop_last=False,
            pin_memory=True,
            batch_size=batch_size,
            num_workers=num_workers
        )
    if pattern == "task2v":
        loader = DataLoader(
            dataset=test_dataset2,
            shuffle=False,
            drop_last=False,
            pin_memory=True,
            batch_size=batch_size,
            num_workers=num_workers
        )
    if pattern == "task1t" or pattern == "task1v":
        open_tag_des = dataset_root + f"/{dataset}1/{dataset}_llm_tag_descriptions.json"
    if pattern == "task2t" or pattern == "task2v":
        open_tag_des = dataset_root + f"/{dataset}2/{dataset}_llm_tag_descriptions.json"

    if os.path.exists(open_tag_des):
        with open(open_tag_des, 'rb') as fo:
            tag_des = json.load(fo)

    else:
        tag_des = None
    info = {
        "taglist": taglist,
        "imglist": imglist,
        # "annot_file": annot_file,
        # "img_root": img_root,
        "tag_des": tag_des
    }

    return loader, info


def get_class_idxs(
        model_type: str,
        open_set: bool,
        taglist: List[str]
) -> Optional[List[int]]:
    """Get indices of required categories in the label system."""
    if model_type == "ram_plus" or model_type == "ram":
        if not open_set:
            model_taglist_file = "ram/data/ram_tag_list.txt"
            with open(model_taglist_file, "r", encoding="utf-8") as f:
                model_taglist = [line.strip() for line in f]
            return [model_taglist.index(tag) for tag in taglist]
        else:
            return None
    else:  # for tag2text, we directly use tagid instead of text-form of tag.
        # here tagid equals to tag index.
        return [int(tag) for tag in taglist]


def load_thresholds(
        threshold: Optional[float],
        threshold_file: Optional[str],
        model_type: str,
        open_set: bool,
        class_idxs: List[int],
        num_classes: int,
) -> List[float]:
    """Decide what threshold(s) to use."""
    if not threshold_file and not threshold:  # use default
        if model_type == "ram_plus" or model_type == "ram":
            if not open_set:  # use class-wise tuned thresholds
                ram_threshold_file = "ram/data/ram_tag_list_threshold.txt"
                with open(ram_threshold_file, "r", encoding="utf-8") as f:
                    idx2thre = {
                        idx: float(line.strip()) for idx, line in enumerate(f)
                    }
                    return [idx2thre[idx] for idx in class_idxs]
            else:
                return [0.5] * num_classes
        else:
            return [0.68] * num_classes
    elif threshold_file:
        with open(threshold_file, "r", encoding="utf-8") as f:
            thresholds = [float(line.strip()) for line in f]
        assert len(thresholds) == num_classes
        return thresholds
    else:
        return [threshold] * num_classes


def gen_pred_file(
        imglist: List[str],
        tags: List[List[str]],
        img_root: str,
        pred_file: str
) -> None:
    """Generate text file of tag prediction results."""
    with open(pred_file, "w", encoding="utf-8") as f:
        for image, tag in zip(imglist, tags):
            # should be relative to img_root to match the gt file.
            s = str(Path(image).relative_to(img_root))
            if tag:
                s = s + "," + ",".join(tag)
            f.write(s + "\n")


def load_ram(
        backbone: str,
        checkpoint: str,
        input_size: int,
        taglist: List[str],
        open_set: bool,
        class_idxs: List[int],
) -> Module:
    model = ram(pretrained=checkpoint, image_size=input_size, vit=backbone)
    # trim taglist for faster inference
    if open_set:
        print("Building tag embeddings ...")
        label_embed, _ = build_openset_label_embedding(taglist)
        model.label_embed = Parameter(label_embed.float())   # shape:(20,512)

    else:
        model.label_embed = Parameter(model.label_embed[class_idxs, :])
    return model.to(device).eval()


@torch.no_grad()
def forward_ram(model: Module, imgs: Tensor) -> Tensor:
    image_embeds = model.image_proj(model.visual_encoder(imgs.to(device)))
    image_atts = torch.ones(
        image_embeds.size()[:-1], dtype=torch.long).to(device)
    label_embed = relu(model.wordvec_proj(model.label_embed)).unsqueeze(0) \
        .repeat(imgs.shape[0], 1, 1)
    tagging_embed, _ = model.tagging_head(
        encoder_embeds=label_embed,
        encoder_hidden_states=image_embeds,
        encoder_attention_mask=image_atts,
        return_dict=False,
        mode='tagging',
    )
    return sigmoid(model.fc(tagging_embed).squeeze(-1))


def print_write(f: TextIO, s: str):
    print(s)
    f.write(s + "\n")


@torch.no_grad()
def test_model_1(ram_model, cla_model, cla_img_model, cla_gcn_model, text_model, test_loader, imglist,  taglist, label_embed_clip, topk,number):
    ram_model.eval()
    text_model.eval()
    cla_model.eval()
    cla_img_model.eval()
    cla_gcn_model.eval()

    median_list = [[] for _ in range(number)]  # 假设有5个类别
    # inference
    logits = torch.empty(len(imglist), len(taglist))   # 预测结果
    targs = torch.empty(len(imglist), len(taglist))    # 真实标签
    pos = 0
    threshold_list = []

    for (imgs, labels) in tqdm(test_loader, desc="Test"):
        labels = labels.to(device)
        image_embeds = ram_model.image_proj(ram_model.visual_encoder(imgs.to(device)))
        image_atts = torch.ones(
            image_embeds.size()[:-1], dtype=torch.long).to(device)
        label_embed = relu(ram_model.wordvec_proj(ram_model.label_embed)).unsqueeze(0) \
            .repeat(imgs.shape[0], 1, 1)
        tagging_embed, _ = ram_model.tagging_head(
            encoder_embeds=label_embed,
            encoder_hidden_states=image_embeds,
            encoder_attention_mask=image_atts,
            return_dict=False,
            mode='tagging',
        )    # 标记嵌入

        # label_embed_clip, cos_sim = generate_matrix(catName_to_catID, model_clip, clip)
        # 开始建图
        label_embed_nor = F.normalize(label_embed_clip, dim=2)
        cos_sim = torch.matmul(label_embed_nor, label_embed_nor.transpose(1, 2))

        # 设置 topk、p，得到稀疏矩阵，缓解过度平滑
        p = 0.2  # 可调
        for i in range(cos_sim.shape[0]):
            matrix = cos_sim[i]
            topk_values, topk_indices = torch.topk(matrix, topk, dim=1)
            result = torch.zeros_like(matrix)
            result.scatter_(1, topk_indices, topk_values)
            matrix.copy_(result)

        # 缓解过度平滑
        for i in range(cos_sim.shape[0]):
            matrix = cos_sim[i]
            diagonal_elements = torch.diagonal(matrix)
            non_diagonal_sum = torch.sum(matrix, dim=1) - diagonal_elements
            scaling_factor = p / non_diagonal_sum
            for j in range(scaling_factor.shape[0]):
                matrix[j, :] = matrix[j, :] * scaling_factor[j]
            matrix.fill_diagonal_(1 - p)

        label_embed = label_embed_clip.repeat(len(labels), 1, 1)
        cos_sim = cos_sim.repeat(len(labels), 1, 1)
        label_embed_ft = text_model(label_embed, cos_sim) + label_embed

        ram_logits = ram_model.fc(tagging_embed).squeeze(-1)
        img_logits = cla_img_model(image_embeds[:, 0, :]).squeeze(1)
        gcn_logits = cla_gcn_model(label_embed_ft).squeeze(-1)


        cat_fea = torch.cat((ram_logits, img_logits, gcn_logits), dim=1)

        fin_logits = cla_model(cat_fea)
        threshold = 0.6

        for class_index in range(fin_logits.shape[1]):
            # 获取当前类别的标签
            label_mask = (labels[:, class_index] == 1).float()
            # 获取当前类别的伪标签概率
            pseudo_label_probs = torch.sigmoid(fin_logits[:, class_index].detach())
            # 计算这些概率的中值，忽略未标记的类别
            if label_mask.sum() > 0:

                valid_probs = pseudo_label_probs[label_mask == 1]
                threshold = valid_probs.median().item()
                threshold = 0.99 if threshold == 1.0 else threshold
                median_list[class_index].append(threshold)


        # # 获取 labels 中第20个类别中标签为1的所有伪标签概率
        # label_range = labels[:, :]
        # label_mask = (label_range == 1).float()
        # pseudo_label_probs = (sigmoid(fin_logits.detach()))
        # # 计算这些概率的均值(中值），忽略未标记的类别
        # if label_mask.sum() > 0:
        #     valid_probs = pseudo_label_probs[label_mask == 1]
        #     threshold = valid_probs.min()
        #     threshold_list.append(threshold.item())
        #     # mean(),max(),min(),median()

        bs = imgs.shape[0]
        logits[pos:pos + bs, :] = sigmoid(fin_logits).cpu()
        targs[pos:pos + bs, :] = labels.cpu()
        pos += bs

    # 计算每个类别的中值的均值
    model_threshold = [sum(medians) / len(medians) if len(medians) > 0 else 0 for medians in median_list]
    # 打印每个类别的阈值
    # print(model_threshold)


    # model_threshold = sum(threshold_list) / len(threshold_list)
    # print(model_threshold)

    # evaluate and record
    mAP, APs = get_mAP_ua(logits.numpy(), targs.numpy(), taglist)
    # CP, CR, Ps, Rs = get_PR(pred_file, annot_file, taglist)

    return mAP, APs, model_threshold

@torch.no_grad()
def test_model_2(ram_model, cla_model, cla_img_model, cla_gcn_model, text_model, test_loader, imglist, taglist,
                 label_embed_clip, topk):
    ram_model.eval()
    text_model.eval()
    cla_model.eval()
    cla_img_model.eval()
    cla_gcn_model.eval()

    # inference
    logits = torch.empty(len(imglist), len(taglist))  # 预测结果
    targs = torch.empty(len(imglist), len(taglist))  # 真实标签
    pos = 0

    for (imgs, labels) in tqdm(test_loader, desc="Test"):
        labels = labels.to(device)
        image_embeds = ram_model.image_proj(ram_model.visual_encoder(imgs.to(device)))
        image_atts = torch.ones(
            image_embeds.size()[:-1], dtype=torch.long).to(device)
        label_embed = relu(ram_model.wordvec_proj(ram_model.label_embed)).unsqueeze(0) \
            .repeat(imgs.shape[0], 1, 1)
        tagging_embed, _ = ram_model.tagging_head(
            encoder_embeds=label_embed,
            encoder_hidden_states=image_embeds,
            encoder_attention_mask=image_atts,
            return_dict=False,
            mode='tagging',
        )  # 标记嵌入

        # label_embed_clip, cos_sim = generate_matrix(catName_to_catID, model_clip, clip)
        # 开始建图
        label_embed_nor = F.normalize(label_embed_clip, dim=2)
        cos_sim = torch.matmul(label_embed_nor, label_embed_nor.transpose(1, 2))

        # 设置 topk、p，得到稀疏矩阵，缓解过度平滑
        p = 0.2  # 可调
        for i in range(cos_sim.shape[0]):
            matrix = cos_sim[i]
            topk_values, topk_indices = torch.topk(matrix, topk, dim=1)
            result = torch.zeros_like(matrix)
            result.scatter_(1, topk_indices, topk_values)
            matrix.copy_(result)

        # 缓解过度平滑
        for i in range(cos_sim.shape[0]):
            matrix = cos_sim[i]
            diagonal_elements = torch.diagonal(matrix)
            non_diagonal_sum = torch.sum(matrix, dim=1) - diagonal_elements
            scaling_factor = p / non_diagonal_sum
            for j in range(scaling_factor.shape[0]):
                matrix[j, :] = matrix[j, :] * scaling_factor[j]
            matrix.fill_diagonal_(1 - p)

        label_embed = label_embed_clip.repeat(len(labels), 1, 1)
        cos_sim = cos_sim.repeat(len(labels), 1, 1)
        label_embed_ft = text_model(label_embed, cos_sim) + label_embed

        ram_logits = ram_model.fc(tagging_embed).squeeze(-1)
        img_logits = cla_img_model(image_embeds[:, 0, :]).squeeze(1)
        gcn_logits = cla_gcn_model(label_embed_ft).squeeze(-1)

        cat_fea = torch.cat((ram_logits, img_logits, gcn_logits), dim=1)

        fin_logits = cla_model(cat_fea)

        bs = imgs.shape[0]
        logits[pos:pos + bs, :] = sigmoid(fin_logits).cpu()
        targs[pos:pos + bs, :] = labels.cpu()
        pos += bs
    # evaluate and record
    mAP, APs = get_mAP_ua(logits.numpy(), targs.numpy(), taglist)
    # CP, CR, Ps, Rs = get_PR(pred_file, annot_file, taglist)

    return mAP, APs


def pseudo_label_generation(ram_model, cla_model, cla_img_model, cla_gcn_model, text_model, imgs, label_embed_clip):

    # GCN_embed = []
    # GCN_embed = torch.tensor(GCN_embed).to(device)
    #
    # for cat in catName_to_catID_1:
    #     template = "a photo of a {}."
    #     cat_text = template.format(cat)
    #     text = clip.tokenize(cat_text).to(device)
    #
    #     pre_GCN_embed = []
    #     pre_GCN_embed = torch.tensor(pre_GCN_embed).to(device)
    #
    #     with torch.no_grad():
    #         text_features = model_clip.encode_text(text).float()
    #     pre_GCN_embed = torch.cat((pre_GCN_embed, text_features), dim=0)
    #     GCN_embed_nor = F.normalize(pre_GCN_embed, dim=1)
    #
    #     # 节点接在一起
    #     GCN_embed = torch.cat((GCN_embed, GCN_embed_nor), dim=0)
    # # (classes,512)
    # label_embed_clip = GCN_embed.clone()
    # label_embed_clip = label_embed_clip.unsqueeze(0)

    image_embeds = ram_model.image_proj(ram_model.visual_encoder(imgs.to(device)))
    image_atts = torch.ones(
        image_embeds.size()[:-1], dtype=torch.long).to(device)
    label_embed = relu(ram_model.wordvec_proj(ram_model.label_embed)).unsqueeze(0) \
        .repeat(imgs.shape[0], 1, 1)
    tagging_embed, _ = ram_model.tagging_head(
        encoder_embeds=label_embed,
        encoder_hidden_states=image_embeds,
        encoder_attention_mask=image_atts,
        return_dict=False,
        mode='tagging',
    )

    # 开始建图
    label_embed_nor = F.normalize(label_embed_clip, dim=2)
    cos_sim = torch.matmul(label_embed_nor, label_embed_nor.transpose(1, 2))

    # 设置 topk、p，得到稀疏矩阵，缓解过度平滑
    topk = 4  # 可调
    p = 0.2  # 可调
    for i in range(cos_sim.shape[0]):
        matrix = cos_sim[i]
        topk_values, topk_indices = torch.topk(matrix, topk, dim=1)
        result = torch.zeros_like(matrix)
        result.scatter_(1, topk_indices, topk_values)
        matrix.copy_(result)

    # 缓解过度平滑
    for i in range(cos_sim.shape[0]):
        matrix = cos_sim[i]
        diagonal_elements = torch.diagonal(matrix)
        non_diagonal_sum = torch.sum(matrix, dim=1) - diagonal_elements
        scaling_factor = p / non_diagonal_sum
        for j in range(scaling_factor.shape[0]):
            matrix[j, :] = matrix[j, :] * scaling_factor[j]
        matrix.fill_diagonal_(1 - p)

    # label_embed_clip, cos_sim = generate_matrix(catName_to_catID_1, model_clip, clip)
    label_embed = label_embed_clip.repeat(len(imgs), 1, 1)
    cos_sim = cos_sim.repeat(len(imgs), 1, 1)
    label_embed_ft = text_model(label_embed, cos_sim) + label_embed


    ram_logits = ram_model.fc(tagging_embed).squeeze(-1)
    img_logits = cla_img_model(image_embeds[:, 0, :]).squeeze(1)
    gcn_logits = cla_gcn_model(label_embed_ft).squeeze(-1)

    cat_fea = torch.cat((ram_logits, img_logits, gcn_logits), dim=1)
    fin_logits = cla_model(cat_fea)

    return fin_logits


def generate_label_embed_clip(catName_to_catID, model_clip, clip):
    GCN_embed = []
    GCN_embed = torch.tensor(GCN_embed).to(device)

    for cat in catName_to_catID:
        template = "a photo of a {}."
        cat_text = template.format(cat)
        text = clip.tokenize(cat_text).to(device)

        pre_GCN_embed = []
        pre_GCN_embed = torch.tensor(pre_GCN_embed).to(device)

        with torch.no_grad():
            text_features = model_clip.encode_text(text).float()
        pre_GCN_embed = torch.cat((pre_GCN_embed, text_features), dim=0)
        GCN_embed_nor = F.normalize(pre_GCN_embed, dim=1)

        # 节点接在一起
        GCN_embed = torch.cat((GCN_embed, GCN_embed_nor), dim=0)
    # (classes,512)
    label_embed_clip = GCN_embed.clone()
    label_embed_clip = label_embed_clip.unsqueeze(0)

    return label_embed_clip


# def generate_matrix(catName_to_catID,catName_to_catID_2, model_clip, clip):
#     GCN_embed_1 = []
#     GCN_embed_2 = []
#     GCN_embed_1 = torch.tensor(GCN_embed_1).to(device)
#     GCN_embed_2 = torch.tensor(GCN_embed_2).to(device)
#
#     for cat in catName_to_catID:
#         # img_path = "/root/SJML/prompt/voc/" + str(cat)
#         text = clip.tokenize(cat).to(device)
#
#         pre_GCN_embed = []
#         pre_GCN_embed = torch.tensor(pre_GCN_embed).to(device)
#
#         with torch.no_grad():
#             text_features = model_clip.encode_text(text).float()
#         pre_GCN_embed = torch.cat((pre_GCN_embed, text_features), dim=0)
#
#         # for picture_name in os.listdir(img_path):
#         #     file_name = img_path + "/" + picture_name
#         #     image = Image.open(file_name)
#         #     image_input = preprocess(image).unsqueeze(0).to(device)
#         #     with torch.no_grad():
#         #         image_features = model_clip.encode_image(image_input).float()
#         #         pre_GCN_embed = torch.cat((pre_GCN_embed, image_features), dim=0)
#
#         GCN_embed_nor = F.normalize(pre_GCN_embed, dim=1)
#         # cos_sim = torch.matmul(GCN_embed_nor, GCN_embed_nor.transpose(0, 1)).to(device)
#         # Pre_text_model = GCN(input_dim=512, hidden_dim=256, output_dim=512).to(device)
#         #
#         # # 卷积+池化
#         # GCN_embed_ft = Pre_text_model(pre_GCN_embed, cos_sim) + pre_GCN_embed
#         # GCN_embed_ft = torch.mean(GCN_embed_ft, dim=0, keepdim=True)
#         #
#         # 节点接在一起
#         GCN_embed_1 = torch.cat((GCN_embed_1, GCN_embed_nor), dim=0)
#     # print('GCN_embed.shape', GCN_embed_1.shape)
#     # (20,512)
#     # 20个节点的图进行卷积
#     # GCN_embed_final = F.normalize(GCN_embed, dim=1)
#     # cos_sim = torch.matmul(GCN_embed_final, GCN_embed_final.transpose(0, 1)).to(device)
#     # Pre_text_model = GCN(input_dim=512, hidden_dim=256, output_dim=512).to(device)
#     # GCN_embed_ft_final = Pre_text_model(GCN_embed, cos_sim) + GCN_embed
#     label_embed_clip_1 = GCN_embed_1.clone()
#     label_embed_clip_1 = label_embed_clip_1.unsqueeze(0)
#
#     for cat in catName_to_catID_2:
#         text = clip.tokenize(cat).to(device)
#
#         pre_GCN_embed = []
#         pre_GCN_embed = torch.tensor(pre_GCN_embed).to(device)
#
#         with torch.no_grad():
#             text_features = model_clip.encode_text(text).float()
#         pre_GCN_embed = torch.cat((pre_GCN_embed, text_features), dim=0)
#         GCN_embed_nor = F.normalize(pre_GCN_embed, dim=1)
#
#         # 节点接在一起
#         GCN_embed_2 = torch.cat((GCN_embed_2, GCN_embed_nor), dim=0)
#     # print('GCN_embed.shape', GCN_embed_2.shape)
#     # (80,512)
#     label_embed_clip_2 = GCN_embed_2.clone()
#     label_embed_clip_2 = label_embed_clip_2.unsqueeze(0)
#
#
#     # 开始建图
#     # GPN_embed = GPN_label_embeds[0, :]
#     label_embed_nor_1 = F.normalize(label_embed_clip_1, dim=2)
#     # print('label_embed_nor_1.shape', label_embed_nor_1.shape)
#     label_embed_nor_2 = F.normalize(label_embed_clip_2, dim=2)
#     # print('label_embed_nor_2.shape', label_embed_nor_2.shape)
#
#     cos_sim_init = torch.matmul(label_embed_nor_2, label_embed_nor_1.transpose(1, 2))
#     # print('cos_sim', cos_sim_init)
#     # print('cos_sim.shape', cos_sim_init.shape)
#     # 获取 cos_sim 的形状
#     batch_size, num_rows, num_cols = cos_sim_init.shape
#     # 创建一个全零张量，形状为 [batch_size, num_rows, num_rows]
#     cos_sim = torch.zeros((batch_size, num_rows, num_rows), device=device)
#     # 将 cos_sim 的元素复制到 expanded_cos_sim 的前 num_cols 列中
#     cos_sim[:, :, :num_cols] = cos_sim_init
#     # 将对角线元素设置为 1.0000
#     for i in range(num_rows):
#         cos_sim[:, i, i] = 1.0000
#     # print('expanded_cos_sim', cos_sim)
#     # print('expanded_cos_sim.shape', cos_sim.shape)
#
#     # 设置 topk、p，得到稀疏矩阵，缓解过度平滑
#     topk = 4  # 可调
#     p = 0.2  # 可调
#     for i in range(cos_sim.shape[0]):
#         matrix = cos_sim[i]
#         topk_values, topk_indices = torch.topk(matrix, topk, dim=1)
#         result = torch.zeros_like(matrix)
#         result.scatter_(1, topk_indices, topk_values)
#         matrix.copy_(result)
#
#     # 缓解过度平滑
#     for i in range(cos_sim.shape[0]):
#         matrix = cos_sim[i]
#         diagonal_elements = torch.diagonal(matrix)
#         non_diagonal_sum = torch.sum(matrix, dim=1) - diagonal_elements
#         scaling_factor = p / non_diagonal_sum
#         for j in range(scaling_factor.shape[0]):
#             matrix[j, :] = matrix[j, :] * scaling_factor[j]
#         matrix.fill_diagonal_(1 - p)
#
#     # # 前20类的处理
#     # for i in range(20):
#     #     matrix = cos_sim[0, i, :]  # 获取第 i 类
#     #     topk_values, topk_indices = torch.topk(matrix, topk + 1)  # 选择 topk+1 个最大的值
#     #     result = torch.zeros_like(matrix)
#     #     result.scatter_(0, topk_indices, topk_values)
#     #     cos_sim[0, i, :] = result
#     # # 缓解过度平滑 - 前20类
#     # for i in range(20):
#     #     matrix = cos_sim[0, i, :]
#     #     diagonal_elements = matrix[i]
#     #     non_diagonal_sum = torch.sum(matrix) - diagonal_elements
#     #     scaling_factor = p / non_diagonal_sum
#     #     matrix = matrix * scaling_factor
#     #     matrix[i] = 1 - p
#     #     cos_sim[0, i, :] = matrix
#     #
#     # # 后60类的处理
#     # for i in range(20, 80):
#     #     matrix = cos_sim[0, i, :]  # 获取第 i 类
#     #     topk_values, topk_indices = torch.topk(matrix, topk)  # 选择 topk 个最大的值
#     #     result = torch.zeros_like(matrix)
#     #     result.scatter_(0, topk_indices, topk_values)
#     #     cos_sim[0, i, :] = result
#     # # 缓解过度平滑 - 后60类
#     # for i in range(20, 80):
#     #     matrix = cos_sim[0, i, :]
#     #     total_sum = torch.sum(matrix)
#     #     scaling_factor = p / total_sum
#     #     matrix = matrix * scaling_factor
#     #     cos_sim[0, i, :] = matrix
#
#     # 结束建图
#     # print('matrix', cos_sim)
#     # print('matrix.shape', cos_sim.shape)
#     # print('label_embed_clip_2.shape', label_embed_clip_2.shape)
#
#     return label_embed_clip_2,cos_sim


def train_task1(train_loader, epochs):
    optimizer = torch.optim.AdamW(
        list(text_model_1.parameters())
        + list(cla_model.parameters())
        + list(cla_gcn_model.parameters())
        + list(cla_img_model.parameters()), lr=1e-2, weight_decay=0.05)

    ram_model_1.to(device).eval()
    text_model_1.to(device).train()
    cla_model.to(device).train()
    cla_gcn_model.to(device).train()
    cla_img_model.to(device).train()

    lambda1 = 0.5
    best_mAP = 0.0
    topk = 4  # 可调

    # start training
    for epoch in range(epochs):

        torch.cuda.empty_cache()

        idx = 0

        for (imgs, labels) in tqdm(train_loader, desc="Train"):
            optimizer.zero_grad()

            labels = labels.to(device)
            image_embeds = ram_model_1.image_proj(ram_model_1.visual_encoder(imgs.to(device)))
            image_atts = torch.ones(
                image_embeds.size()[:-1], dtype=torch.long).to(device)
            label_embed = relu(ram_model_1.wordvec_proj(ram_model_1.label_embed)).unsqueeze(0) \
                .repeat(imgs.shape[0], 1, 1)
            tagging_embed, _ = ram_model_1.tagging_head(
                encoder_embeds=label_embed,
                encoder_hidden_states=image_embeds,
                encoder_attention_mask=image_atts,
                return_dict=False,
                mode='tagging',
            )

            # 开始建图
            label_embed_nor = F.normalize(label_embed_clip_1, dim=2)
            cos_sim = torch.matmul(label_embed_nor, label_embed_nor.transpose(1, 2))

            # 设置 p，得到稀疏矩阵，缓解过度平滑

            p = 0.2  # 可调
            for i in range(cos_sim.shape[0]):
                matrix = cos_sim[i]
                topk_values, topk_indices = torch.topk(matrix, topk, dim=1)
                result = torch.zeros_like(matrix)
                result.scatter_(1, topk_indices, topk_values)
                matrix.copy_(result)

            # 缓解过度平滑
            for i in range(cos_sim.shape[0]):
                matrix = cos_sim[i]
                diagonal_elements = torch.diagonal(matrix)
                non_diagonal_sum = torch.sum(matrix, dim=1) - diagonal_elements
                scaling_factor = p / non_diagonal_sum
                for j in range(scaling_factor.shape[0]):
                    matrix[j, :] = matrix[j, :] * scaling_factor[j]
                matrix.fill_diagonal_(1 - p)

            # label_embed_clip, cos_sim = generate_matrix(catName_to_catID_1, model_clip, clip)
            label_embed = label_embed_clip_1.repeat(len(labels), 1, 1)
            cos_sim = cos_sim.repeat(len(labels), 1, 1)
            label_embed_ft = text_model_1(label_embed, cos_sim) + label_embed

            ram_logits = ram_model_1.fc(tagging_embed).squeeze(-1)
            img_logits = cla_img_model(image_embeds[:, 0, :]).squeeze(1)
            gcn_logits = cla_gcn_model(label_embed_ft).squeeze(-1)

            cat_fea = torch.cat((ram_logits, img_logits, gcn_logits), dim=1)

            fin_logits = cla_model(cat_fea)
            fin_pres = sigmoid(fin_logits).float()

            # l2_loss
            # l2_loss = nn.MSELoss()
            # loss = l2_loss(fin_logits, labels.float())

            # VLPL_loss
            # threshold = 0.75
            # labels = labels.float()
            # # pseudo_label = labels
            # pseudo_label = (sigmoid(ram_logits.detach())).float()
            # pseudo_label = (pseudo_label >= threshold).float()
            # loss = loss_VLPL(logits=fin_pres, pseudo_labels=labels, obs_labels=labels)

            # ASL loss
            loss = loss_asl(logits=fin_pres, obs_labels=labels)

            # BCE loss
            # loss = loss_bce(logits=fin_pres, obs_labels=labels)

            # AN loss
            # loss = loss_an(logits=fin_pres, obs_labels=labels)

            # EM loss
            # loss = loss_EM(logits=fin_pres, obs_labels=labels)

            # EM-APL loss
            # warm_up_epoch = 1
            # theta = 0.9
            # obs_label = labels.clone().detach()
            # fin_pres_ = (sigmoid(fin_logits).float()).clone().detach()
            # if epoch<=warm_up_epoch:
            #     loss = loss_EM(logits=fin_pres, obs_labels=labels)
            # else:
            #
            #     for cls_num in range(fin_pres_.shape[1]):
            #
            #         class_pres = fin_pres_[:,cls_num]
            #         class_labels_obs = obs_label[:,cls_num]
            #         class_idx = torch.arange(class_pres.shape[0])
            #         class_idx = class_idx.to(device)
            #
            #         # select unlabeled data
            #         unlabel_class_preds = class_pres[class_labels_obs == 0]
            #         unlabel_class_idx = class_idx[class_labels_obs == 0]
            #
            #         # select samples
            #         neg_PL_num = int(unlabel_class_preds.shape[0] * theta/3)
            #         _,indices = torch.sort(unlabel_class_preds)
            #         indices = indices[:neg_PL_num]
            #
            #         for loc in indices:
            #             real_loc = unlabel_class_idx[loc]
            #             obs_label[real_loc,cls_num] = -fin_pres_[real_loc,cls_num]
            #
            #     loss = loss_EM_APL(fin_pres, obs_label)

            # # LL-R loss
            # loss = loss_LL_R(fin_logits,labels,epoch)

            loss.backward()
            optimizer.step()

            idx += 1

        # test and save checkpoint
        # mAP, APs = test_model(ram_model_1, cla_model, cla_img_model, cla_gcn_model, test1_loader, test1_imglist,
        #                       test1_taglist)
        mAP, APs, model1_threshold = test_model_1(ram_model_1, cla_model, cla_img_model, cla_gcn_model, text_model_1, test1_loader,
                               test1_imglist, test1_taglist, label_embed_clip_1, topk,number=old_class)



        if mAP >= best_mAP:
            save_obj = {
                'cla_model': cla_model.state_dict(),
                'cla_img_model': cla_img_model.state_dict(),
                'cla_gcn_model': cla_gcn_model.state_dict(),
                'text_model_1': text_model_1.state_dict(),
                'model1_threshold': model1_threshold,
                'epoch': epoch,
                'mAP': mAP,
                'APs': APs
            }

            torch.save(save_obj, os.path.join(output_dir_1, 'checkpoint_task1.pth'))



            with open(ap_file, "w", encoding="utf-8") as f:
                f.write("Tag,AP\n")
                for tag, AP in zip(taglist1, APs):
                    f.write(f"{tag},{AP * 100.0:.2f}\n")

            with open(summary_file, "a", encoding="utf-8") as f:
                print_write(f, f"mAP: {mAP * 100.0}")

            best_mAP = mAP

        print(f"Epoch : {epoch} | Loss : {loss.item()} | mAP : {mAP}")


def train_task2(train_loader, epochs):

    optimizer = torch.optim.AdamW(list(text_model_2.parameters())
        + list(cla_model_2.parameters())
        + list(cla_gcn_model_2.parameters())
        + list(cla_img_model_2.parameters()), lr=1e-2, weight_decay=0.05)

    ram_model_1.to(device).eval()
    ram_model_2.to(device).eval()
    text_model_2.to(device).train()
    cla_model_2.to(device).train()
    cla_gcn_model_2.to(device).train()
    cla_img_model_2.to(device).train()

    lambda1 = 0.5
    best_mAP = 0.0
    topk = 10  # 可调

    # start training
    for epoch in range(epochs):

        torch.cuda.empty_cache()

        idx = 0

        for (imgs, labels) in tqdm(train_loader, desc="Train"):
            optimizer.zero_grad()

            labels = labels.to(device)

            true_label = torch.zeros_like(labels)
            true_label[:, old_class:old_class + label_class] = labels[:, old_class:old_class + label_class]
            labels = true_label

            image_embeds = ram_model_2.image_proj(ram_model_2.visual_encoder(imgs.to(device)))
            image_atts = torch.ones(
                image_embeds.size()[:-1], dtype=torch.long).to(device)
            label_embed = relu(ram_model_2.wordvec_proj(ram_model_2.label_embed)).unsqueeze(0) \
                .repeat(imgs.shape[0], 1, 1)
            tagging_embed, _ = ram_model_2.tagging_head(
                encoder_embeds=label_embed,
                encoder_hidden_states=image_embeds,
                encoder_attention_mask=image_atts,
                return_dict=False,
                mode='tagging',
            )


            # 开始建图
            label_embed_nor = F.normalize(label_embed_clip, dim=2)
            cos_sim = torch.matmul(label_embed_nor, label_embed_nor.transpose(1, 2))

            # 设置 topk、p，得到稀疏矩阵，缓解过度平滑
            p = 0.2  # 可调
            for i in range(cos_sim.shape[0]):
                matrix = cos_sim[i]
                topk_values, topk_indices = torch.topk(matrix, topk, dim=1)
                result = torch.zeros_like(matrix)
                result.scatter_(1, topk_indices, topk_values)
                matrix.copy_(result)

            # 缓解过度平滑
            for i in range(cos_sim.shape[0]):
                matrix = cos_sim[i]
                diagonal_elements = torch.diagonal(matrix)
                non_diagonal_sum = torch.sum(matrix, dim=1) - diagonal_elements
                scaling_factor = p / non_diagonal_sum
                for j in range(scaling_factor.shape[0]):
                    matrix[j, :] = matrix[j, :] * scaling_factor[j]
                matrix.fill_diagonal_(1 - p)

            label_embed = label_embed_clip.repeat(len(labels), 1, 1)
            cos_sim = cos_sim.repeat(len(labels), 1, 1)
            # print('cos_sim',cos_sim.shape)
            label_embed_ft = text_model_2(label_embed, cos_sim) + label_embed

            ram_logits = ram_model_2.fc(tagging_embed).squeeze(-1)
            img_logits = cla_img_model_2(image_embeds[:, 0, :]).squeeze(1)
            gcn_logits = cla_gcn_model_2(label_embed_ft).squeeze(-1)
            cat_fea = torch.cat((ram_logits, img_logits, gcn_logits), dim=1)

            fin_logits = cla_model_2(cat_fea)
            fin_pres = sigmoid(fin_logits).float()

            # l2_loss
            # l2_loss = nn.MSELoss()
            # loss = l2_loss(fin_logits, labels.float())

            # VLPL_loss
            # threshold = model1_threshold

            pseudo_labels_fin = pseudo_label_generation(ram_model_1, cla_model, cla_img_model, cla_gcn_model, text_model_1, imgs,label_embed_clip_1)

            student_subset = fin_logits[:, :old_class]
            cosine_similarity = F.cosine_similarity(pseudo_labels_fin, student_subset, dim=1)
            loss_lt = 1.0 - cosine_similarity.mean()

            # 参数 a 的值，可以在 0 到 1 之间调整
            # a = 0
            # ram_logits[:, :20] = a * ram_logits[:, :20] + (1 - a) * pseudo_labels_task1[:, :20]
            pseudo_labels_task1 = (sigmoid(pseudo_labels_fin.detach())).float()


            final_pseudo_labels = torch.zeros_like(pseudo_labels_task1)
            # 遍历每个类别并设置相应的阈值
            for class_index, threshold in enumerate(model1_threshold):
                final_pseudo_labels[:, class_index] = (pseudo_labels_task1[:, class_index] >= threshold).float()
            pseudo_labels_task1= final_pseudo_labels


            # pseudo_labels_task1 = (pseudo_labels_task1 >= threshold).float()

            # 获取 labels 中第old_class到label_class个类别中标签为1的所有伪标签概率
            # label_range = labels[:, old_class:old_class + label_class]
            # label_mask = (label_range == 1).float()
            # pseudo_label_probs = sigmoid(ram_logits[:, old_class:old_class + label_class].detach())
            # # 计算这些概率的均值(中值），忽略未标记的类别
            # if label_mask.sum() > 0:
            #     valid_probs = pseudo_label_probs[label_mask == 1]
            #     threshold = valid_probs.min()
            threshold = 0.6
            pseudo_label = (sigmoid(ram_logits.detach())).float()
            pseudo_label = (pseudo_label >= threshold).float()
            pseudo_label[:, :old_class] = pseudo_labels_task1[:, :]
            labels = labels.float()
            pseudo_label[:, old_class:old_class + label_class] = labels[:, old_class:old_class + label_class]
            # labels[:, 20:50] = pseudo_labels_task1[:, 20:50]

            # loss = loss_asl(logits=fin_pres, obs_labels=pseudo_label)

            # loss = loss_VLPL(logits=fin_pres, pseudo_labels=pseudo_label, obs_labels=labels)

            # EM loss
            # loss = loss_EM(logits=fin_pres, obs_labels=pseudo_label)

            # EM-APL loss
            warm_up_epoch = 1
            theta = 0.9
            obs_label = pseudo_label.clone().detach()
            fin_pres_ = (sigmoid(fin_logits).float()).clone().detach()
            if epoch<=warm_up_epoch:
                loss = loss_EM(logits=fin_pres, obs_labels=pseudo_label)
            else:

                for cls_num in range(fin_pres_.shape[1]):

                    class_pres = fin_pres_[:,cls_num]
                    class_labels_obs = obs_label[:,cls_num]
                    class_idx = torch.arange(class_pres.shape[0])
                    class_idx = class_idx.to(device)

                    # select unlabeled data
                    unlabel_class_preds = class_pres[class_labels_obs == 0]
                    unlabel_class_idx = class_idx[class_labels_obs == 0]

                    # select samples
                    neg_PL_num = int(unlabel_class_preds.shape[0] * theta/3)
                    _,indices = torch.sort(unlabel_class_preds)
                    indices = indices[:neg_PL_num]

                    for loc in indices:
                        real_loc = unlabel_class_idx[loc]
                        obs_label[real_loc,cls_num] = -fin_pres_[real_loc,cls_num]

                loss = loss_EM_APL(fin_pres, obs_label)

            # # LL-R loss
            # loss = loss_LL_R(fin_logits,pseudo_label,epoch)

            loss = loss + 25 * loss_lt
            loss.backward()
            optimizer.step()

            idx += 1

        # test and save checkpoint
        # mAP, APs = test_model(ram_model_2, cla_model_2, cla_img_model_2, cla_gcn_model_2, test2_loader, test2_imglist,
        #                       test2_taglist)
        mAP, APs = test_model_2(ram_model_2, cla_model_2, cla_img_model_2, cla_gcn_model_2, text_model_2, test2_loader
                              , test2_imglist, test2_taglist, label_embed_clip,topk)


        if mAP >= best_mAP:
            save_obj = {
                'cla_model_2': cla_model_2.state_dict(),
                'cla_img_model_2': cla_img_model_2.state_dict(),
                'cla_gcn_model_2': cla_gcn_model_2.state_dict(),
                'text_model_2': text_model_2.state_dict(),
                'epoch': epoch,
                'mAP': mAP,
                'APs': APs
            }
            torch.save(save_obj, os.path.join(output_dir_2, 'checkpoint_task2.pth'))

            with open(ap_file, "w", encoding="utf-8") as f:
                f.write("Tag,AP\n")
                for tag, AP in zip(taglist2, APs):
                    f.write(f"{tag},{AP * 100.0:.2f}\n")

            with open(summary_file, "a", encoding="utf-8") as f:
                print_write(f, f"mAP: {mAP * 100.0}")

            best_mAP = mAP

        print(f"Epoch : {epoch} | Loss : {loss.item()} | mAP : {mAP}")


if __name__ == "__main__":
    args = parse_args()

    if args.dataset == 'coco':
        args.input_size = 224
        catName_to_catID_1 = {
            'person': 0,
            'bicycle': 1,
            'car': 2,
            'motorcycle': 3,
            'airplane': 4,
            'bus': 5,
            'train': 6,
            'truck': 7,
            'boat': 8,
            'trafficlight': 9,
            'firehydrant': 10,
            'stopsign': 11,
            'parkingmeter': 12,
            'bench': 13,
            'bird': 14,
            'cat': 15,
            'dog': 16,
            'horse': 17,
            'sheep': 18,
            'cow': 19
        }
        # catID_to_catName = {catName_to_catID[k]: k for k in catName_to_catID}
        catName_to_catID_2 = {
            'person': 0,
            'bicycle': 1,
            'car': 2,
            'motorcycle': 3,
            'airplane': 4,
            'bus': 5,
            'train': 6,
            'truck': 7,
            'boat': 8,
            'trafficlight': 9,
            'firehydrant': 10,
            'stopsign': 11,
            'parkingmeter': 12,
            'bench': 13,
            'bird': 14,
            'cat': 15,
            'dog': 16,
            'horse': 17,
            'sheep': 18,
            'cow': 19,
            'elephant': 20,
            'bear': 21,
            'zebra': 22,
            'giraffe': 23,
            'backpack': 24,
            'umbrella': 25,
            'handbag': 26,
            'tie': 27,
            'suitcase': 28,
            'frisbee': 29,
            'skis': 30,
            'snowboard': 31,
            'sportsball': 32,
            'kite': 33,
            'baseballbat': 34,
            'baseballglove': 35,
            'skateboard': 36,
            'surfboard': 37,
            'tennisracket': 38,
            'bottle': 39,
            'wineglass': 40,
            'cup': 41,
            'fork': 42,
            'knife': 43,
            'spoon': 44,
            'bowl': 45,
            'banana': 46,
            'apple': 47,
            'sandwich': 48,
            'orange': 49,
            'broccoli': 50,
            'carrot': 51,
            'hotdog': 52,
            'pizza': 53,
            'donut': 54,
            'cake': 55,
            'chair': 56,
            'couch': 57,
            'pottedplant': 58,
            'bed': 59,
            'diningtable': 60,
            'toilet': 61,
            'tv': 62,
            'laptop': 63,
            'mouse': 64,
            'remote': 65,
            'keyboard': 66,
            'cellphone': 67,
            'microwave': 68,
            'oven': 69,
            'toaster': 70,
            'sink': 71,
            'refrigerator': 72,
            'book': 73,
            'clock': 74,
            'vase': 75,
            'scissors': 76,
            'teddybear': 77,
            'hairdrier': 78,
            'toothbrush': 79
        }
        # catName_to_catID_2 = {
        #     'elephant': 0,
        #     'bear': 1,
        #     'zebra': 2,
        #     'giraffe': 3,
        #     'backpack': 4,
        #     'umbrella': 5,
        #     'handbag': 6,
        #     'tie': 7,
        #     'suitcase': 8,
        #     'frisbee': 9,
        #     'skis': 10,
        #     'snowboard': 11,
        #     'sportsball': 12,
        #     'kite': 13,
        #     'baseballbat': 14,
        #     'baseballglove': 15,
        #     'skateboard': 16,
        #     'surfboard': 17,
        #     'tennisracket': 18,
        #     'bottle': 19,
        #     'wineglass': 20,
        #     'cup': 21,
        #     'fork': 22,
        #     'knife': 23,
        #     'spoon': 24,
        #     'bowl': 25,
        #     'banana': 26,
        #     'apple': 27,
        #     'sandwich': 28,
        #     'orange': 29,
        #     'broccoli': 30,
        #     'carrot': 31,
        #     'hotdog': 32,
        #     'pizza': 33,
        #     'donut': 34,
        #     'cake': 35,
        #     'chair': 36,
        #     'couch': 37,
        #     'pottedplant': 38,
        #     'bed': 39,
        #     'diningtable': 40,
        #     'toilet': 41,
        #     'tv': 42,
        #     'laptop': 43,
        #     'mouse': 44,
        #     'remote': 45,
        #     'keyboard': 46,
        #     'cellphone': 47,
        #     'microwave': 48,
        #     'oven': 49,
        #     'toaster': 50,
        #     'sink': 51,
        #     'refrigerator': 52,
        #     'book': 53,
        #     'clock': 54,
        #     'vase': 55,
        #     'scissors': 56,
        #     'teddybear': 57,
        #     'hairdrier': 58,
        #     'toothbrush': 59
        # }

    if args.dataset == 'voc':
        args.input_size = 224
        catName_to_catID_1 = {
            'aeroplane': 0,
            'bicycle': 1,
            'bird': 2,
            'boat': 3
            # 'bottle': 4
        }
        catName_to_catID_2 = {
            'aeroplane': 0,
            'bicycle': 1,
            'bird': 2,
            'boat': 3,
            'bottle': 4,
            'bus': 5,
            'car': 6,
            'cat': 7,
            'chair': 8,
            'cow': 9,
            'diningtable': 10,
            'dog': 11,
            'horse': 12,
            'motorbike': 13,
            'person': 14,
            'pottedplant': 15,
            'sheep': 16,
            'sofa': 17,
            'train': 18,
            'tvmonitor': 19
        }


    # if args.dataset == "cub":
    #     args.batch_size = 64
    #     args.input_size = 384

    # fix the seed for reproducibility
    seed = args.seed + get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    cudnn.benchmark = True

    # set up output paths
    # output_dir = args.output_dir + "/" + args.dataset + "/" + "Train_with_all_single_positive_labels_L2loss"
    # output_dir_1 = args.output_dir + "/" + "coco_task1" + "/" + "Train_with_full_labels_loss"
    output_dir_1 = args.output_dir + "/" + args.dataset + "_task1" + "/" + "Train_with_full_labels_ASL_loss"
    Path(output_dir_1).mkdir(parents=True, exist_ok=True)
    pred_file, pr_file, ap_file, summary_file, logit_file = [
        output_dir_1 + "/" + name for name in
        ("pred.txt", "pr.txt", "ap.txt", "summary.txt", "logits.pth")
    ]
    with open(summary_file, "w", encoding="utf-8") as f:
        print_write(f, "****************")
        for key in (
                "model_type", "backbone", "checkpoint", "open_set",
                "dataset", "input_size",
                "threshold", "threshold_file",
                "output_dir", "batch_size", "num_workers"
        ):
            print_write(f, f"{key}: {getattr(args, key)}")
        print_write(f, "****************")

    # prepare data1
    loader1, info1 = load_spml_datasets(
        dataset=args.dataset,
        model_type=args.model_type,
        pattern="task1t",
        input_size=args.input_size,
        batch_size=args.batch_size,
        num_workers=args.num_workers
    )
    taglist1, imglist1, tag_des1 = \
        info1["taglist"], info1["imglist"], info1["tag_des"]

    test1_loader, test1_info = load_spml_datasets(
        dataset=args.dataset,
        model_type=args.model_type,
        pattern="task1v",
        input_size=args.input_size,
        batch_size=args.batch_size,
        num_workers=args.num_workers
    )

    test1_taglist, test1_imglist, test1_tag_des = \
        test1_info["taglist"], test1_info["imglist"], test1_info["tag_des"]

    # get class idxs
    class_idxs_1 = get_class_idxs(
        model_type=args.model_type,
        open_set=args.open_set,
        taglist=taglist1
    )

    # set up threshold(s)
    thresholds_1 = load_thresholds(
        threshold=args.threshold,
        threshold_file=args.threshold_file,
        model_type=args.model_type,
        open_set=args.open_set,
        class_idxs=class_idxs_1,
        num_classes=len(taglist1)
    )

    if args.model_type == "ram":
        ram_model_1 = load_ram(
            backbone=args.backbone,
            checkpoint=args.checkpoint,
            input_size=args.input_size,
            taglist=taglist1,
            open_set=args.open_set,
            class_idxs=class_idxs_1
        )

    # freeze
    for params in ram_model_1.parameters():
        params.requires_grad = False

    num_classes = len(taglist1)
    old_class = num_classes
    cla_img_model = cls_network.cls_network(input_size=512, hidden_size=256, output_size=num_classes)
    cla_gcn_model = nn.Linear(512, 1)
    cla_model = nn.Linear(num_classes * 3, num_classes)
    text_model_1 = cls_network.GCN(input_dim=512, hidden_dim=256, output_dim=512)

    label_embed_clip_1 = generate_label_embed_clip(catName_to_catID_1, model_clip, clip)

    # train_task1(train_loader=loader1, epochs=args.epochs)
    # test_model(ram_model_1, cla_model, cla_img_model, cla_gcn_model, text_model_1, test1_loader,
    #                       catName_to_catID_1, test1_imglist, test1_taglist, label_embed_clip_1)

    # 加载权重
    checkpoint = torch.load(os.path.join(output_dir_1, 'checkpoint_task1.pth'))

    # 加载模型权重
    cla_model.load_state_dict(checkpoint['cla_model'])
    cla_img_model.load_state_dict(checkpoint['cla_img_model'])
    cla_gcn_model.load_state_dict(checkpoint['cla_gcn_model'])
    text_model_1.load_state_dict(checkpoint['text_model_1'])
    model1_threshold = checkpoint['model1_threshold']
    cla_model.to(device).eval()
    cla_img_model.to(device).eval()
    cla_gcn_model.to(device).eval()
    text_model_1.to(device).eval()

    # set up output paths
    # output_dir_2 = args.output_dir + "/" + "coco_task2" + "/" + "Train_with_pseudo_labels_VLPLloss"
    output_dir_2 = args.output_dir + "/" + args.dataset + "_task2" + "/" + "Train_with_pseudo_labels_ASL_loss"
    Path(output_dir_2).mkdir(parents=True, exist_ok=True)
    pred_file, pr_file, ap_file, summary_file, logit_file = [
        output_dir_2 + "/" + name for name in
        ("pred.txt", "pr.txt", "ap.txt", "summary.txt", "logits.pth")
    ]
    with open(summary_file, "w", encoding="utf-8") as f:
        print_write(f, "****************")
        for key in (
                "model_type", "backbone", "checkpoint", "open_set",
                "dataset", "input_size",
                "threshold", "threshold_file",
                "output_dir", "batch_size", "num_workers"
        ):
            print_write(f, f"{key}: {getattr(args, key)}")
        print_write(f, "****************")


    # prepare data2
    loader2, info2 = load_spml_datasets(
        dataset=args.dataset,
        model_type=args.model_type,
        pattern="task2t",
        input_size=args.input_size,
        batch_size=args.batch_size,
        num_workers=args.num_workers
    )
    taglist2, imglist2, tag_des2 = \
        info2["taglist"], info2["imglist"], info2["tag_des"]

    test2_loader, test2_info = load_spml_datasets(
        dataset=args.dataset,
        model_type=args.model_type,
        pattern="task2v",
        input_size=args.input_size,
        batch_size=args.batch_size,
        num_workers=args.num_workers
    )

    test2_taglist, test2_imglist, test2_tag_des = \
        test2_info["taglist"], test2_info["imglist"], test2_info["tag_des"]

    # get class idxs
    class_idxs_2 = get_class_idxs(
        model_type=args.model_type,
        open_set=args.open_set,
        taglist=taglist2
    )

    # set up threshold(s)
    thresholds_2 = load_thresholds(
        threshold=args.threshold,
        threshold_file=args.threshold_file,
        model_type=args.model_type,
        open_set=args.open_set,
        class_idxs=class_idxs_2,
        num_classes=len(taglist2)
    )

    if args.model_type == "ram":
        ram_model_2 = load_ram(
            backbone=args.backbone,
            checkpoint=args.checkpoint,
            input_size=args.input_size,
            taglist=taglist2,
            open_set=args.open_set,
            class_idxs=class_idxs_2
        )

    for params in ram_model_2.parameters():
        params.requires_grad = False

    num_classes = len(taglist2)
    label_class = 8
    cla_model_2 = nn.Linear(num_classes * 3, num_classes)
    cla_img_model_2 = cls_network.cls_network(input_size=512, hidden_size=256, output_size=num_classes)
    cla_gcn_model_2 = nn.Linear(512, 1)
    text_model_2 = cls_network.GCN(input_dim=512, hidden_dim=256, output_dim=512)
    label_embed_clip = generate_label_embed_clip(catName_to_catID_2, model_clip, clip)
    train_task2(train_loader=loader2, epochs=args.epochs)

    # mAP, APs = test_model(ram_model_2, cla_model_2, cla_img_model_2, cla_gcn_model_2, text_model_2, test2_loader
    #                       , catName_to_catID_2, test2_imglist, test2_taglist)
