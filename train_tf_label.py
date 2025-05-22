import json
import os
from argparse import ArgumentParser
from pathlib import Path
from typing import Dict, List, Optional, TextIO, Tuple
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
# from ram.utils import build_openset_llm_label_embedding, build_openset_label_embedding, get_mAP, get_PR
from utils import step_lr_schedule
from dataset_loader import load_datasets
from cls_model import Linear, MLP, Cnn
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


def get_class_idxs(
        model_type: str,
        open_set: bool,
        taglist: List[str]
) -> Optional[List[int]]:
    """Get indices of required categories in the label system."""
    if model_type == "ram_plus" or model_type == "ram":
        if not open_set:
            model_taglist_file = "../ram/data/ram_tag_list.txt"
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
                ram_threshold_file = "../ram/data/ram_tag_list_threshold.txt"
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


# def gen_pred_file(
#         imglist: List[str],
#         tags: List[List[str]],
#         img_root: str,
#         pred_file: str
# ) -> None:
#     """Generate text file of tag prediction results."""
#     with open(pred_file, "w", encoding="utf-8") as f:
#         for image, tag in zip(imglist, tags):
#             # should be relative to img_root to match the gt file.
#             s = str(Path(image).relative_to(img_root))
#             if tag:
#                 s = s + "," + ",".join(tag)
#             f.write(s + "\n")


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
        model.label_embed = Parameter(label_embed.float())
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
def test_model(ram_model, model, test_loader, taglist):
    ram_model.eval()
    model.eval()

    # inference
    final_logits = torch.empty(len(test_loader.dataset), len(taglist))
    targs = torch.empty(len(test_loader.dataset))
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
        )
        # logits = ram_model.fc(tagging_embed).squeeze(-1)

        # print("logits = ", logits.size())
        logits = model(tagging_embed)
        # print("labels size", F.one_hot(labels.to(torch.int64), 100).size())
        bs = imgs.shape[0]
        final_logits[pos:pos + bs, :] = sigmoid(logits).cpu()
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
    # taglist, imglist, tag_des = \
    #     info["taglist"], info["imglist"], info["tag_des"]
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

    # get class idxs
    class_idxs = get_class_idxs(
        model_type=args.model_type,
        open_set=args.open_set,
        taglist=taglist
    )
    print("class_idxs",class_idxs)
    # set up threshold(s)
    thresholds = load_thresholds(
        threshold=args.threshold,
        threshold_file=args.threshold_file,
        model_type=args.model_type,
        open_set=args.open_set,
        class_idxs=class_idxs,
        num_classes=len(taglist)
    )

    if args.model_type == "ram":
        ram_model = load_ram(
            backbone=args.backbone,
            checkpoint=args.checkpoint,
            input_size=args.input_size,
            taglist=taglist,
            open_set=args.open_set,
            class_idxs=class_idxs
        )

    # freeze
    for params in ram_model.parameters():
        params.requires_grad = False

    num_classes = len(taglist)
    print("number of taglist = ", num_classes)
    model = MLP(input_dim=768*num_classes, output_dim=num_classes)
    # model = MLP(input_dim=num_classes, output_dim=num_classes)
    # model = Cnn(input_channels=3, n_outputs=num_classes, dropout_rate=0.25)
    for params in model.parameters():
        params.requires_grad = True
    l2_loss = nn.MSELoss()
    ce_loss = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.05)
    # scheduler = StepLR(optimizer, step_size=2, gamma=0.9)  ####0.98
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, 100)

    ram_model.to(device)
    model.to(device)

    top1, top5 = 0.0, 0.0
    best_top1 = 0
    # start training
    for epoch in range(args.epochs):
        model.train()
        torch.cuda.empty_cache()

        step_lr_schedule(optimizer, epoch, init_lr=1e-3, min_lr=5e-5, decay_rate=0.9)


        for (imgs, labels) in tqdm(train_loader, desc="Train"):

            optimizer.zero_grad()
            imgs = imgs.to(device)
            labels = labels.to(device)
            labels = torch.tensor(labels, dtype=torch.float32)

            image_embeds = ram_model.image_proj(ram_model.visual_encoder(imgs))
            image_embeds = image_embeds.to(device)
            image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(device)

            label_embed = relu(ram_model.wordvec_proj(ram_model.label_embed)).unsqueeze(0).repeat(imgs.shape[0], 1, 1)
            label_embed = label_embed.to(device)

            tagging_embed, _ = ram_model.tagging_head(
                encoder_embeds=label_embed,
                encoder_hidden_states=image_embeds,
                encoder_attention_mask=image_atts,
                return_dict=False,
                mode='tagging',
            )

            # ram_logits = ram_model.fc(tagging_embed).squeeze(-1)
            print("image size", tagging_embed.size())
            logits = model(tagging_embed)
            # logits = model(ram_logits)
            logits = torch.sigmoid(logits)
            # sigmoid_ram_logits = torch.sigmoid(ram_logits)

            # labels = F.one_hot(labels.to(torch.int64), 100) * 2 - 1
            loss = ce_loss(logits, labels.long())

            loss. backward()
            optimizer.step()


        # test and save checkpoint
        top1, top5 = test_model(ram_model, model, test_loader, taglist)
        # scheduler.step()
        if top1 >= best_top1:
            save_obj = {
                'model': model.state_dict(),
                'epoch': epoch,
                'top1': top1,
                'top5': top5
            }

            torch.save(save_obj, os.path.join(output_dir, 'checkpoint_best.pth'))



            best_top1 = top1

        # print(f"Epoch : {epoch} | Loss : {loss.item()} | mAP : {mAP}")
        print(f"Epoch : {epoch}  | top1 : {top1[0]}  | top5 : {top5[0]}")

