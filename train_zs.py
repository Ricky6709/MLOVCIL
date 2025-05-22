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

from ram import get_transform
from ram.models import ram_plus, ram, tag2text, cls_linear, cls_network, cls_test, cls_net_active
from ram.utils import build_openset_llm_label_embedding, build_openset_label_embedding, get_mAP, get_PR
from ram.utils.metrics import get_mAP_ua
from utils import step_lr_schedule, get_rank

from losses import *

device = "cuda" if torch.cuda.is_available() else "cpu"


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
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=30)
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


def test_zs(ram_model, test_loader, imglist, taglist):
    ram_model.eval()
    # cla_model.eval()
    # cla_img_model.eval()
    # cla_gcn_model.eval()

    # inference
    logits = torch.empty(len(imglist), len(taglist))   # 预测结果
    targs = torch.empty(len(imglist), len(taglist))    # 真实标签
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
        )    # 标记嵌入

        ram_logits = ram_model.fc(tagging_embed).squeeze(-1)
        # img_logits = cla_img_model(image_embeds[:, 0, :]).squeeze(1)
        # gcn_logits = cla_gcn_model(label_embed).squeeze(-1)


        # cat_fea = torch.cat((ram_logits, img_logits, gcn_logits), dim=1)

        # fin_logits = cla_model(cat_fea)

        # l2_loss
        # pseudo_label = (sigmoid(ram_logits.detach())).float()
        # loss = l2_loss(fin_logits, labels.float())
        bs = imgs.shape[0]
        logits[pos:pos + bs, :] = sigmoid(ram_logits).cpu()
        targs[pos:pos + bs, :] = labels.cpu()
        pos += bs

    # evaluate and record
    mAP, APs = get_mAP_ua(logits.numpy(), targs.numpy(), taglist)
    # CP, CR, Ps, Rs = get_PR(pred_file, annot_file, taglist)

    return mAP, APs



@torch.no_grad()
def test_model(ram_model, cla_model, cla_img_model, cla_gcn_model, test_loader, imglist, taglist):
    ram_model.eval()
    cla_model.eval()
    cla_img_model.eval()
    cla_gcn_model.eval()

    # inference
    logits = torch.empty(len(imglist), len(taglist))   # 预测结果
    targs = torch.empty(len(imglist), len(taglist))    # 真实标签
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
        )    # 标记嵌入

        ram_logits = ram_model.fc(tagging_embed).squeeze(-1)
        img_logits = cla_img_model(image_embeds[:, 0, :]).squeeze(1)
        gcn_logits = cla_gcn_model(label_embed).squeeze(-1)


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






def train_first(train_loader, epochs):

    l2_loss = nn.MSELoss()


    optimizer = torch.optim.AdamW(
        list(cla_model.parameters())
        + list(cla_gcn_model.parameters())
        + list(cla_img_model.parameters()), lr=1e-2, weight_decay=0.05)

    ram_model.to(device).eval()
    cla_model.to(device).train()
    cla_gcn_model.to(device).train()
    cla_img_model.to(device).train()

    lambda1 = 0.5
    best_mAP = 0.0

    # start training
    for epoch in range(epochs):

        torch.cuda.empty_cache()

        idx = 0

        for (imgs, labels) in tqdm(train_loader, desc="Train"):
            optimizer.zero_grad()

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

            ram_logits = ram_model.fc(tagging_embed).squeeze(-1)
            img_logits = cla_img_model(image_embeds[:, 0, :]).squeeze(1)
            gcn_logits = cla_gcn_model(label_embed).squeeze(-1)

            cat_fea = torch.cat((ram_logits, img_logits, gcn_logits), dim=1)

            fin_logits = cla_model(cat_fea)
            fin_pres = sigmoid(fin_logits).float()

            # l2_loss
            # pseudo_label = (sigmoid(ram_logits.detach())).float()
            # loss = l2_loss(fin_logits, labels.float())

            # # VLPL_loss
            threshold = 0.9
            pseudo_label = (sigmoid(ram_logits.detach())).float()
            pseudo_label = (pseudo_label>=threshold).float()
            loss = loss_VLPL(logits=fin_pres, pseudo_labels=pseudo_label, obs_labels=labels)

            # BCE loss
            # loss = loss_bce(logits=fin_pres, obs_labels=labels)

            # AN loss
            # loss = loss_an(logits=fin_pres, obs_labels=labels)

            # # EM loss
            # loss = loss_EM(logits=fin_pres, obs_labels=labels)

            #             # EM-APL loss
            #             warm_up_epoch = 1
            #             theta = 0.9
            #             obs_label = labels.clone().detach()
            #             fin_pres_ = (sigmoid(fin_logits).float()).clone().detach()
            #             if epoch<=warm_up_epoch:
            #                 loss = loss_EM(logits=fin_pres, obs_labels=labels)
            #             else:

            #                 for cls_num in range(fin_pres_.shape[1]):

            #                     class_pres = fin_pres_[:,cls_num]
            #                     class_labels_obs = obs_label[:,cls_num]
            #                     class_idx = torch.arange(class_pres.shape[0])
            #                     class_idx = class_idx.to(device)

                                # select unlabeled data
                               #  unlabel_class_preds = class_pres[class_labels_obs == 0]
                               #  unlabel_class_idx = class_idx[class_labels_obs == 0]
                               #
                               # # select samples
                               #  neg_PL_num = int(unlabel_class_preds.shape[0] * theta/3)
                               #  _,indices = torch.sort(unlabel_class_preds)
                               #  indices = indices[:neg_PL_num]
                               #
                               #  for loc in indices:
                               #      real_loc = unlabel_class_idx[loc]
                               #      obs_label[real_loc,cls_num] = -fin_pres_[real_loc,cls_num]

            #                 loss = loss_EM_APL(fin_pres, obs_label)

            # # LL-R loss
            # loss = loss_LL_R(fin_logits,labels,epoch)



            loss.backward()
            optimizer.step()

            idx += 1

        # test and save checkpoint
        mAP, APs = test_model(ram_model, cla_model, cla_img_model, cla_gcn_model, test2_loader, test2_imglist,
                              test2_taglist)

        if mAP >= best_mAP:
            save_obj = {
                'cla_model': cla_model.state_dict(),
                'cla_img_model': cla_img_model.state_dict(),
                'cla_gcn_model': cla_gcn_model.state_dict(),
                'epoch': epoch,
                'mAP': mAP,
                'APs': APs
            }

            # torch.save(save_obj, os.path.join(output_dir, 'checkpoint_best.pth'))
            # for param in cla_img_model.linear2.weight[:20].parameters():
            #     param.requires_grad = False
            # for param in cla_img_model.linear2.bias[:20].parameters():
            #     param.requires_grad = False


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
    output_dir = args.output_dir + "/" + args.dataset + "/" + "Train_with_full_labels_L2loss"
    # output_dir = args.output_dir + "/" + args.dataset + "/" + "Train_try"
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
    class_idxs = get_class_idxs(
        model_type=args.model_type,
        open_set=args.open_set,
        taglist=taglist2
    )

    # set up threshold(s)
    thresholds = load_thresholds(
        threshold=args.threshold,
        threshold_file=args.threshold_file,
        model_type=args.model_type,
        open_set=args.open_set,
        class_idxs=class_idxs,
        num_classes=len(taglist2)
    )

    if args.model_type == "ram":
        ram_model = load_ram(
            backbone=args.backbone,
            checkpoint=args.checkpoint,
            input_size=args.input_size,
            taglist=taglist2,
            open_set=args.open_set,
            class_idxs=class_idxs
        )



    # freeze
    for params in ram_model.parameters():
        params.requires_grad = False

    num_classes = len(taglist2)
    # num_classes = 20
    cla_model = nn.Linear(num_classes * 3, num_classes)

    # cla_img_model = cls_net_active.cls_network(input_size=512, hidden_size=256, total_classes=80, active_classes=20)
    cla_img_model = cls_network.cls_network(input_size=512, hidden_size=256, output_size=num_classes)
    cla_gcn_model = nn.Linear(768, 1)
    # text_model = GCN(input_dim=768, hidden_dim=256, output_dim=768)

    # train_first(train_loader=loader1, epochs=args.epochs)

    mAP, APs = test_zs(ram_model, test2_loader, test2_imglist, test2_taglist)

    with open(ap_file, "w", encoding="utf-8") as f:
        f.write("Tag,AP\n")
        for tag, AP in zip(taglist2, APs):
            f.write(f"{tag},{AP * 100.0:.2f}\n")

    with open(summary_file, "a", encoding="utf-8") as f:
        print_write(f, f"mAP: {mAP * 100.0}")

    best_mAP = mAP

    print(f" mAP : {mAP}")