from argparse import ArgumentParser
from pathlib import Path
from typing import Dict, List, Optional, TextIO, Tuple

import torch
from torchvision import models, transforms
from PIL import Image, UnidentifiedImageError
from torch import Tensor
from torch.nn import Module, Parameter
from torch.nn.functional import relu, sigmoid, mse_loss
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
from ram.models import ram_plus, ram, tag2text, cls_network, ICA
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
    def __init__(self, X, Y, input_size, transform=None):
        self.X = X
        self.Y = Y
        self.transform = get_transform(input_size)
        self.data_path = './datasets/voc/VOC2012'

    def __getitem__(self, index):
        x = Image.open(self.data_path + '/JPEGImages/' + self.X[index]).convert('RGB')
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


class CUB_200_2011_handler(Dataset):
    def __init__(self, X, Y, input_size, transform=None):
        self.X = X
        self.Y = Y
        self.transform = get_transform(input_size)
        self.data_path = './datasets/cub/CUB_200_2011/images'

    def __getitem__(self, index):
        x = Image.open(self.data_path + '/' + self.X[index]).convert('RGB')
        x = self.transform(x)
        y = self.Y[index]
        return x, y

    def __len__(self):
        return len(self.X)


class NUS_WIDE_handler(Dataset):
    def __init__(self, X, Y, input_size, transform=None):
        self.X = X
        self.Y = Y
        self.transform = get_transform(input_size)
        self.data_path = './datasets/nus/Flickr'

    def __getitem__(self, index):
        x = Image.open(self.data_path + '/' + self.X[index]).convert('RGB')
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
                            "cub",
                            "nus"
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
    parser.add_argument("--epochs", type=int, default=20)
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
            tag_file = dataset_root + f"/coco1/{dataset}_ram_taglist.txt"
            imglist = np.load(os.path.join(f'./datasets/{dataset}/npy', f'formatted_{pattern}_images.npy'))
        if pattern == "task2t" or pattern == "task2v":
            tag_file = dataset_root + f"/coco2/{dataset}_ram_taglist.txt"
            imglist = np.load(os.path.join(f'./datasets/{dataset}/npy', f'formatted_{pattern}_images.npy'))
    else:
        tag_file = dataset_root + f"/{dataset}_tag2text_tagidlist.txt"
        imglist = np.load(os.path.join(f'./datasets/{dataset}/npy', f'formatted_task1t_images.npy'))

    with open(tag_file, "r", encoding="utf-8") as f:
        taglist = [line.strip() for line in f]

    # imglist = np.load(os.path.join(f'./datasets/{dataset}/npy', f'formatted_{pattern}_images.npy'))

    if dataset == "voc":
        train_dataset = VOC2012_handler(X=np.load(os.path.join('./datasets/voc', 'formatted_train_images.npy')),
                                        Y=np.load(os.path.join('./datasets/voc', 'formatted_train_labels_obs.npy')),
                                        input_size=input_size)
        test_dataset = VOC2012_handler(X=np.load(os.path.join('./datasets/voc', 'formatted_val_images.npy')),
                                       Y=np.load(os.path.join('./datasets/voc', 'formatted_val_labels.npy')),
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
    elif dataset == "cub":
        train_dataset = CUB_200_2011_handler(X=np.load(os.path.join('./datasets/cub', 'formatted_train_images.npy')),
                                             Y=np.load(os.path.join('./datasets/cub', 'formatted_train_labels.npy')),
                                             input_size=input_size)
        test_dataset = CUB_200_2011_handler(X=np.load(os.path.join('./datasets/cub', 'formatted_val_images.npy')),
                                            Y=np.load(os.path.join('./datasets/cub', 'formatted_val_labels.npy')),
                                            input_size=input_size)
    elif dataset == "nus":
        train_dataset = NUS_WIDE_handler(X=np.load(os.path.join('./datasets/nus', 'formatted_train_images.npy')),
                                         Y=np.load(os.path.join('./datasets/nus', 'formatted_train_labels_obs.npy')),
                                         input_size=input_size)
        test_dataset = NUS_WIDE_handler(X=np.load(os.path.join('./datasets/nus', 'formatted_val_images.npy')),
                                        Y=np.load(os.path.join('./datasets/nus', 'formatted_val_labels.npy')),
                                        input_size=input_size)

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
        open_tag_des = dataset_root + f"/coco1/{dataset}_llm_tag_descriptions.json"
    if pattern == "task2t" or pattern == "task2v":
        open_tag_des = dataset_root + f"/coco2/{dataset}_llm_tag_descriptions.json"

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
        model.label_embed = Parameter(label_embed.float())  # shape:(20,512)

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
def test_model(ram_model, ica_model, test_loader, imglist, taglist, session_idx):
    ram_model.eval()
    ica_model.eval()

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
        )
        fin_logits = ica_model(tagging_embed, session_idx)
        bs = imgs.shape[0]
        logits[pos:pos + bs, :] = sigmoid(fin_logits).cpu()
        targs[pos:pos + bs, :] = labels.cpu()
        pos += bs

    # evaluate and record
    mAP, APs = get_mAP_ua(logits.numpy(), targs.numpy(), taglist)
    # CP, CR, Ps, Rs = get_PR(pred_file, annot_file, taglist)

    return mAP, APs


def pseudo_label_generation(ram_model, ica_model, imgs, session_idx):
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

    fin_logits = ica_model(tagging_embed, session_idx)

    return fin_logits


def train_task1(train_loader, epochs,session_idx):
    optimizer = torch.optim.AdamW(
        list(ica_model.parameters()), lr=1e-4, weight_decay=1e-4)

    ram_model_1.to(device).eval()
    ica_model.to(device).train()

    lambda1 = 0.5
    best_mAP = 0.0

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

            fin_pres = ica_model(tagging_embed,  session_idx)
            fin_pres = sigmoid(fin_pres).float()

            # BCE loss
            loss = loss_asl(logits=fin_pres, obs_labels=labels)

            loss.backward()
            optimizer.step()

            idx += 1

        # test and save checkpoint
        mAP, APs = test_model(ram_model_1, ica_model, test1_loader, test1_imglist, test1_taglist, session_idx)

        if mAP >= best_mAP:
            save_obj = {
                'ica_model': ica_model.state_dict(),
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


def train_task2(train_loader, epochs, session_idx):
    optimizer = torch.optim.AdamW(
        list(ica_model_2.parameters()), lr=1e-4, weight_decay=1e-4)

    ram_model_1.to(device).eval()
    ram_model_2.to(device).eval()
    ica_model.to(device).eval()
    ica_model_2.to(device).train()

    lambda1 = 0.5
    best_mAP = 0.0

    # start training
    for epoch in range(epochs):

        torch.cuda.empty_cache()

        idx = 0

        for (imgs, labels) in tqdm(train_loader, desc="Train"):
            optimizer.zero_grad()
            labels = labels.to(device)
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

            fin_logits = ica_model_2(tagging_embed, session_idx)
            fin_pres = sigmoid(fin_logits).float()


            pseudo_labels_fin = pseudo_label_generation(ram_model_1, ica_model, imgs, session_idx=0)
            student_subset = fin_logits[:, :20]

            cosine_similarity = F.cosine_similarity(pseudo_labels_fin, student_subset, dim=1)
            loss_lt = 1.0 - cosine_similarity.mean()

            threshold = 0.75

            pseudo_labels_task1 = (sigmoid(pseudo_labels_fin.detach())).float()
            ram_logits = ram_model_2.fc(tagging_embed).squeeze(-1)
            pseudo_label = (sigmoid(ram_logits.detach())).float()
            pseudo_label[:, :20] = pseudo_labels_task1[:, :]
            pseudo_label = (pseudo_label >= threshold).float()
            labels = labels.float()
            pseudo_label[:, 20:50] = labels[:, 20:50]

            loss = loss_asl(logits=fin_pres, obs_labels=pseudo_label)
            # loss = loss_VLPL(logits=fin_pres, pseudo_labels=pseudo_label, obs_labels=labels)
            n = 100
            loss = loss + n * loss_lt

            loss.backward()
            optimizer.step()

            idx += 1

        # test and save checkpoint
        mAP, APs = test_model(ram_model_2, ica_model_2, test2_loader, test2_imglist, test2_taglist, session_idx=1)

        if mAP >= best_mAP:
            save_obj = {
                'ica_model_2': ica_model_2.state_dict(),
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

    if args.dataset == 'coco' or args.dataset == 'nus':
        args.input_size = 224

    if args.dataset == "cub":
        args.batch_size = 64
        args.input_size = 384

    # fix the seed for reproducibility
    seed = args.seed + get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    cudnn.benchmark = True

    # set up output paths
    # output_dir = args.output_dir + "/" + args.dataset + "/" + "Train_with_all_single_positive_labels_L2loss"
    # output_dir_1 = args.output_dir_1 + "/" + args.dataset + "/" + "Train_with_full_labels_BCEloss"
    output_dir_1 = args.output_dir + "/" + "coco_task1" + "/" + "Train_with_full_labels_test_ica_loss"
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

    c = 768  # 特征图的通道数
    d = 512  # 投影后的特征维度
    h = 8  # 注意力头数
    l = 20  # 特征图的长度
    num_classes_list = [20, 60]  # 每个会话的类别数量

    ica_model = ICA.ICABasedMLCIL(c, d, h, l, num_classes_list)

    # train_task1(train_loader=loader1, epochs=args.epochs, session_idx=0)
    # train_task1(train_loader=loader1, epochs=args.epochs)

    # 加载权重
    checkpoint = torch.load(os.path.join(output_dir_1, 'checkpoint_task1.pth'))
    # 加载模型权重
    ica_model.load_state_dict(checkpoint['ica_model'])

    # set up output paths
    output_dir_2 = args.output_dir + "/" + "coco_task2" + "/" + "Train_with_pseudo_labels_ica_val_VLPLloss"
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

    ica_model_2 = ICA.ICABasedMLCIL(c, d, h, l, num_classes_list)
    train_task2(train_loader=loader2, epochs=args.epochs, session_idx=1)
    # train_task2(train_loader=loader2, epochs=args.epochs)

