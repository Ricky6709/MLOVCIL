import json
import os
from argparse import ArgumentParser
from pathlib import Path
from typing import Dict, List, Optional, TextIO, Tuple

import torch.optim as optim
from transformers import to_pil_image

from clip import clip
import torch.nn.functional as F
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torch import Tensor
from torch.nn import Module, Parameter
from torch.nn.functional import relu, sigmoid
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from ram import get_transform
from ram.models import ram
from ram.utils import build_openset_label_embedding
from ram.utils.metrics import accuracy
# from ram.utils import build_openset_llm_label_embedding, build_openset_label_embedding, get_mAP, get_PR, build_openset_image_embedding
from utils import step_lr_schedule
from dataset_loader import load_datasets
from select2 import SELECTCIFAR100, dict_transform
from cls_model import Linear, MLP, Cnn, MLP_clip
from torch.optim.lr_scheduler import StepLR

def parse_args():
    parser = ArgumentParser()
    # model
    parser.add_argument("--model-type", type=str, choices=("ram", "ram_plus"), default="ram")
    parser.add_argument("--checkpoint", type=str, default='./pretrained/ram_swin_large_14m.pth')
    parser.add_argument("--backbone", type=str, choices=("swin_l", "swin_b"), default="swin_l",
                        help="If `None`, will judge from `--model-type`")
    parser.add_argument("--open-set", type=bool, default=True,
                        help=(
                            "Treat all categories in the taglist file as "
                            "unseen and perform open-set classification. Only "
                            "works with RAM."
                        ))
    # data
    parser.add_argument("--dataset", type=str, choices=("openimages_common_214", "CIFAR100"), default="CIFAR100")
    parser.add_argument("--input-size", type=int, default=224, choices=(224, 384))
    # threshold
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--threshold", type=float, default=None,
                       help=(
                           "Use custom threshold for all classes. Mutually "
                           "exclusive with `--threshold-file`. If both "
                           "`--threshold` and `--threshold-file` is `None`, "
                           "will use a default threshold setting."
                       ))
    group.add_argument("--threshold-file", type=str, default=None,
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
    parser.add_argument("--output-dir", type=str, default="./tfl_outputs")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=80)

    args = parser.parse_args()

    # post process and validity check
    args.model_type = args.model_type.lower()

    assert not (args.model_type == "tag2text" and args.open_set)

    if args.backbone is None:
        args.backbone = "swin_l" if args.model_type == "ram_plus" or args.model_type == "ram" else "swin_b"

    return args

device = "cuda" if torch.cuda.is_available() else "cpu"
single_template = ["a photo of a {}."]

def article(name):
    return "an" if name[0] in "aeiou" else "a"


def processed_name(name, rm_dot=False):
    res = name.replace("_", " ").replace("/", " or ").lower()
    if rm_dot:
        res = res.rstrip(".")
    return res


def build_clip_label_embedding(model, categories):
    # print("Creating pretrained CLIP image model")
    templates = single_template
    run_on_gpu = torch.cuda.is_available()

    with torch.no_grad():
        openset_label_embedding = []
        for category in categories:
            texts = [
                template.format(
                    processed_name(category, rm_dot=True), article=article(category)
                )
                for template in templates
            ]
            texts = [
                "This is " + text if text.startswith("a") or text.startswith("the") else text
                for text in texts
            ]  # 改造句子
            texts = clip.tokenize(texts)  # tokenize
            if run_on_gpu:
                texts = texts.cuda()
                model = model.cuda()
            text_embeddings = model.encode_text(texts)
            text_embeddings /= text_embeddings.norm(dim=-1, keepdim=True)
            text_embedding = text_embeddings.mean(dim=0)
            text_embedding /= text_embedding.norm()
            openset_label_embedding.append(text_embedding)
        openset_label_embedding = torch.stack(openset_label_embedding, dim=1)
        if run_on_gpu:
            openset_label_embedding = openset_label_embedding.cuda()

    openset_label_embedding = openset_label_embedding.t()
    return openset_label_embedding

def load_clip( ) -> Module:
    model, _ = clip.load("ViT-B/16")
    return model.to(device).eval()


def print_write(f: TextIO, s: str):
    print(s)
    f.write(s + "\n")


def train_model(clip_model, model, labeled_loader, unlabeled_loader, num_epochs=10):

    for params in clip_model.parameters():
        params.requires_grad = False
    for params in model.parameters():
        params.requires_grad = True


    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    model.to(device)
    clip_model.to(device)
    label_embed_or = build_clip_label_embedding(clip_model, taglist)

    for epoch in range(num_epochs):
        model.train()
        step_lr_schedule(optimizer, epoch, init_lr=1e-3, min_lr=5e-6, decay_rate=0.9)
        torch.cuda.empty_cache()

        for batch_idx, (labeled_batch, unlabeled_batch) in enumerate(zip(labeled_loader, unlabeled_loader)):
            labeled_data, labeled_labels = labeled_batch
            unlabeled_data, _ = unlabeled_batch
            label_embed = label_embed_or.repeat(labeled_data.size()[0], 1, 1)
            optimizer.zero_grad()
            labeled_data, labeled_labels, unlabeled_data = labeled_data.to(device), labeled_labels.to(device), unlabeled_data.to(device)
            labeled_image_embeds = clip_model.encode_image(labeled_data).unsqueeze(1)
            # print("labeled_image_embeds shape:", labeled_image_embeds.shape)
            labeled_image_embeds = labeled_image_embeds.to(device)
            label_embed = label_embed.to(device)
            labeled_feature = torch.flatten(torch.cat((labeled_image_embeds, label_embed), dim=1), 1, -1).float()
            logits = model(labeled_feature)
            labeled_loss = criterion(logits, labeled_labels)

            labeled_loss.backward()

            unlabel_embed = label_embed_or.repeat(unlabeled_data.size()[0], 1, 1)
            unlabel_embed = unlabel_embed.to(device)
            unlabeled_image_embeds = clip_model.encode_image(unlabeled_data).unsqueeze(1)
            unlabeled_image_embeds = unlabeled_image_embeds.to(device)
            # print("unlabeled_image_embeds shape:", unlabeled_image_embeds.shape)
            unlabeled_feature = torch.flatten(torch.cat((unlabeled_image_embeds, unlabel_embed), dim=1), 1, -1).float()
            # print("label_embed shape:", label_embed.shape)

            # 生成伪标签和计算余弦相似度
            with torch.no_grad():
                cosine_similarity = F.cosine_similarity(unlabel_embed, unlabeled_image_embeds,
                                                        dim=-1)
                # print("cosine_similarity shape:", cosine_similarity.shape)
                pseudo_labels = torch.argmax(cosine_similarity, dim=-1)
                softmax_similarity = F.softmax(cosine_similarity, dim=-1)
                # print("softmax_similarity shape:", softmax_similarity.shape)

            # 只保留最有信心的对，可以根据阈值进行筛选
            confidence_threshold = 0

            confident_mask = softmax_similarity.max(dim=1)[0] > confidence_threshold
            confident_mask = confident_mask.unsqueeze(1)
            # pseudo_labels = torch.argmax(softmax_similarity, dim=-1)
            pseudo_labels_confident = pseudo_labels[confident_mask[:, 0]]


            # confident_mask = cosine_similarity.max(dim=1)[0] > confidence_threshold
            # confident_mask = confident_mask[:, 0]
            # pseudo_labels_confident = pseudo_labels[confident_mask]
            # print(pseudo_labels_confident.shape)
            # print(pseudo_labels_confident)


            if len(pseudo_labels_confident) > 0:
                unlabeled_outputs = model(unlabeled_feature)
                unlabeled_loss = criterion(unlabeled_outputs, pseudo_labels_confident)

            else:
                unlabeled_loss = 0.0  # 如果没有可用的伪标签，损失为0

            unlabeled_loss.backward()
            optimizer.step()

            # 最终的损失
            loss = labeled_loss + unlabeled_loss


            if batch_idx % 2 == 0:
                print(f"Epoch {epoch}, Batch {batch_idx}, Loss: {loss.item()}")

        # 保存模型状态
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss.item(),
        }, f"model_checkpoint_epoch_{epoch}.pth")


transform = get_transform()

def custom_collate_fn(batch):
    # 定义自定义 collate_fn，确保图像转换为张量
    images, labels = zip(*batch)
    images = [to_pil_image(image) if isinstance(image, torch.Tensor) else image for image in images]
    images = [transform(image) for image in images]
    return torch.stack(images), torch.tensor(labels)


if __name__ == "__main__":
    args = parse_args()

    # set up output paths
    output_dir = args.output_dir + "/" + args.dataset + "/" + "Train"
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    pred_file, pr_file, ap_file, summary_file, logit_file = [
        output_dir + "/" + name for name in
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

    # prepare data
    train_loader, info = load_datasets(
        dataset=args.dataset,
        model_type=args.model_type,
        pattern="train",
        input_size=args.input_size,
        batch_size=args.batch_size,
        num_workers=args.num_workers
    )

    taglist, tag_des = \
        info["taglist"], info["tag_des"]

    # test_loader, _ = load_datasets(
    #     dataset=args.dataset,
    #     model_type=args.model_type,
    #     pattern="val",
    #     input_size=args.input_size,
    #     batch_size=args.batch_size,
    #     num_workers=args.num_workers
    # )

    train_label_set = SELECTCIFAR100(root='./datasets/cifar100/', labeled=True,labeled_num=90, labeled_ratio=0.01,
                                                 download=True,transform=dict_transform['cifar_train'])
    train_unlabel_set = SELECTCIFAR100(root='./datasets/cifar100/', labeled=False, labeled_num=90,labeled_ratio=0.01,
                                       download=True,transform=dict_transform['cifar_train'],
                                       unlabeled_idxs=train_label_set.unlabeled_idxs)
    test_set = SELECTCIFAR100(root='./datasets/cifar100/', labeled=False, labeled_num=90,
                                          labeled_ratio=0.01, download=True,
                                          transform=dict_transform['cifar_test'],
                                          unlabeled_idxs=train_label_set.unlabeled_idxs)
    clip_model = load_clip()
    num_classes = len(taglist)
    print("number of taglist = ", num_classes)
    model = MLP_clip(input_dim=512*(num_classes+1), output_dim=num_classes)

    labeled_len = len(train_label_set)
    unlabeled_len = len(train_unlabel_set)
    labeled_batch_size = int(128 * labeled_len / (labeled_len + unlabeled_len))

    labeled_loader = torch.utils.data.DataLoader(train_label_set, batch_size=labeled_batch_size, shuffle=True, drop_last=True, collate_fn=custom_collate_fn)
    unlabeled_loader = torch.utils.data.DataLoader(train_unlabel_set, batch_size=128-labeled_batch_size, shuffle=True, drop_last=True, collate_fn=custom_collate_fn)
    train_model(clip_model, model, labeled_loader, unlabeled_loader)