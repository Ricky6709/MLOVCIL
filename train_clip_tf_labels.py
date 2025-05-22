import json
import os
from argparse import ArgumentParser
from pathlib import Path
from typing import Dict, List, Optional, TextIO, Tuple

from clip import clip
import torch.nn.functional as F
import numpy as np
import torch
import torch.nn as nn
from PIL import Image, UnidentifiedImageError
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
from cls_model import Linear, MLP, Cnn, MLP_clip
from torch.optim.lr_scheduler import StepLR

device = "cuda" if torch.cuda.is_available() else "cpu"

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

single_template = ["a photo of a {}."]

def article(name):
    return "an" if name[0] in "aeiou" else "a"


def processed_name(name, rm_dot=False):
    # _ for lvis
    # / for obj365
    res = name.replace("_", " ").replace("/", " or ").lower()
    if rm_dot:
        res = res.rstrip(".")
    return res


loss_mse = nn.MSELoss(size_average=True).cuda()
def c_loss(output, label, class_mun):
    one_hot = F.one_hot(label.to(torch.int64), class_mun).cuda() * 2 - 1
    sig_out = output * one_hot
    y_label = torch.ones(sig_out.size()).cuda()
    output = loss_mse(sig_out, y_label)
    return output



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



def load_ram( ) -> Module:
    model, _ = clip.load("ViT-B/16")
    return model.to(device).eval()


def print_write(f: TextIO, s: str):
    print(s)
    f.write(s + "\n")


@torch.no_grad()
def test_model(ram_model, model, test_loader, taglist):
    ram_model.eval()
    model.eval()

    # inference
    final_logits = torch.empty(len(test_loader.dataset), len(taglist))
    targs = torch.empty(len(test_loader.dataset))
    pos = 0
    label_embed_or = build_clip_label_embedding(ram_model, taglist)
    for (imgs, labels) in tqdm(test_loader, desc="Test"):
        imgs = imgs.to(device)
        labels = labels.to(device)
        labels = torch.tensor(labels, dtype=torch.float32)

        image_embeds = ram_model.encode_image(imgs).unsqueeze(1)
        image_embeds = image_embeds.to(device)

        label_embed = label_embed_or.repeat(imgs.size()[0], 1, 1)
        label_embed = label_embed.to(device)

        feature = torch.flatten(torch.cat((image_embeds, label_embed), dim=1), 1, -1).float()
        logits = model(feature)



        bs = imgs.shape[0]
        final_logits[pos:pos + bs, :] = logits.cpu()
        # targs[pos:pos + bs, :] = F.one_hot(labels.to(torch.int64), 100).cpu() * 2 - 1
        targs[pos:pos + bs] = labels.cpu()
        pos += bs



    # evaluate and record
    top1, top5 = accuracy(final_logits, targs, topk=(1, 5))
    return top1, top5


def neg_log(x):
    LOG_EPSILON = 1e-5
    return - torch.log(x + LOG_EPSILON)




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

    test_loader, _ = load_datasets(
        dataset=args.dataset,
        model_type=args.model_type,
        pattern="val",
        input_size=args.input_size,
        batch_size=args.batch_size,
        num_workers=args.num_workers
    )



    if args.model_type == "ram":
        ram_model = load_ram()

    # freeze
    for params in ram_model.parameters():
        params.requires_grad = False

    num_classes = len(taglist)
    print("number of taglist = ", num_classes)
    model = MLP_clip(input_dim=512*(num_classes+1), output_dim=num_classes)
    # model = MLP(input_dim=num_classes, output_dim=num_classes)
    # model = Cnn(input_channels=3, n_outputs=num_classes, dropout_rate=0.25)
    for params in model.parameters():
        params.requires_grad = True
    l2_loss = nn.MSELoss(size_average=True)
    ce_loss = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.05)


    ram_model.to(device)
    model.to(device)

    top1, top5 = 0.0, 0.0
    best_top1 = 0
    # start training
    label_embed_or = build_clip_label_embedding(ram_model, taglist)
    for epoch in range(args.epochs):
        model.train()

        step_lr_schedule(optimizer, epoch, init_lr=1e-3, min_lr=5e-6, decay_rate=0.9)
        torch.cuda.empty_cache()



        for (imgs, labels) in tqdm(train_loader, desc="Train"):
            label_embed = label_embed_or.repeat(imgs.size()[0], 1, 1)
            optimizer.zero_grad()
            imgs = imgs.to(device)
            labels = labels.to(device)

            image_embeds = ram_model.encode_image(imgs).unsqueeze(1)
            image_embeds = image_embeds.to(device)

            label_embed = label_embed.to(device)

            feature = torch.flatten(torch.cat((image_embeds, label_embed), dim = 1), 1, -1).float()

            logits = model(feature)
            # loss = c_loss(logits, labels, num_classes)
            loss = ce_loss(logits, labels)


            loss.backward()
            optimizer.step()


        # test
        top1, top5 = test_model(ram_model, model, test_loader, taglist)

        print(f"Epoch : {epoch}  | top1 : {top1[0]}  | top5 : {top5[0]}")

