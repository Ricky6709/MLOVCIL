import json
import os
from argparse import ArgumentParser
from pathlib import Path
from typing import Dict, List, Optional, TextIO, Tuple
import sys
import clip

import numpy as np
import torch
import torch.nn as nn
from PIL import Image, UnidentifiedImageError
from torch import Tensor
from torch.nn import Module, Parameter
from torch.nn.functional import relu, sigmoid
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ram import get_transform
from ram.models import ram
from ram.utils import build_openset_label_embedding
from ram.utils.metrics import get_mAP_sjml
from utils import step_lr_schedule
from dataset_loader import load_spml_datasets
import torch.nn.functional as F
from linear import cls_network, GCN

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

catName_to_catID = {
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
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
catID_to_catName = {catName_to_catID[k]: k for k in catName_to_catID}

model_clip, preprocess = clip.load("ViT-B/32", device=device)


def parse_args():
    parser = ArgumentParser()
    # model
    parser.add_argument("--model-type", type=str, choices=("ram", "ram_plus"), default="ram")
    parser.add_argument("--checkpoint", type=str, default='../pretrained/ram_swin_large_14m.pth')
    parser.add_argument("--backbone", type=str, choices=("swin_l", "swin_b"), default="swin_l",
                        help="If `None`, will judge from `--model-type`")
    parser.add_argument("--open-set", type=bool, default=True,
                        help=(
                            "Treat all categories in the taglist file as "
                            "unseen and perform open-set classification. Only "
                            "works with RAM."
                        ))
    # data
    parser.add_argument("--dataset", type=str, choices=("openimages_common_214", "openimages_rare_200",
                                                        "voc", "coco", "cub"), default="coco")
    parser.add_argument("--input-size", type=int, default=384)
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
    parser.add_argument("--output-dir", type=str, default="./SJML_outputs")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=30)

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
def test_model(ram_model, cla_model, cla_img_model, cla_gcn_model, text_model, test_loader, taglist):
    ram_model.eval()
    text_model.eval()
    cla_model.eval()

    # inference
    logits = torch.empty(len(test_loader.dataset), len(taglist))
    targs = torch.empty(len(test_loader.dataset), len(taglist))
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

        # campute adjacency_matrix
        label_embed_nor = F.normalize(label_embed, dim=2)
        cos_sim = torch.matmul(label_embed_nor, label_embed_nor.transpose(1, 2))

        # set topk, p, get sparse matrix and alleviate the over-smoothing
        topk = 5
        p = 0.2
        for i in range(cos_sim.shape[0]):
            matrix = cos_sim[i]
            topk_values, topk_indices = torch.topk(matrix, topk, dim=1)
            result = torch.zeros_like(matrix)
            result.scatter_(1, topk_indices, topk_values)
            matrix.copy_(result)

        # alleviate the over-smoothing
        for i in range(cos_sim.shape[0]):
            matrix = cos_sim[i]
            diagonal_elements = torch.diagonal(matrix)
            non_diagonal_sum = torch.sum(matrix, dim=1) - diagonal_elements
            scaling_factor = p / non_diagonal_sum
            for j in range(scaling_factor.shape[0]):
                matrix[j, :] = matrix[j, :] * scaling_factor[j]
            matrix.fill_diagonal_(1 - p)

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
    mAP, APs = get_mAP_sjml(logits.numpy(), targs.numpy(), taglist)
    # CP, CR, Ps, Rs = get_PR(pred_file, annot_file, taglist)

    return mAP, APs


def neg_log(x):
    LOG_EPSILON = 1e-5
    return - torch.log(x + LOG_EPSILON)


def loss_p(preds, origin_preds, labels):
    loss_mtx = origin_preds * neg_log(preds) + (1 - origin_preds) * neg_log(1 - preds)
    loss_mtx = torch.sum(loss_mtx, dim=1)
    weights = origin_preds[labels == 1]
    weights = 1.0 / (weights * labels.shape[1])

    print(np.shape(loss_mtx), np.shape(weights))

    loss_mtx = weights * loss_mtx
    return loss_mtx.mean()


def loss_n(preds, origin_preds, labels):
    loss_mtx = origin_preds * neg_log(preds) + (1 - origin_preds) * neg_log(1 - preds)
    loss_mtx = torch.sum(loss_mtx, dim=1)
    weights = origin_preds[labels == -1]
    weights = 1.0 / ((1 - weights) * labels.shape[1])

    print(np.shape(loss_mtx), np.shape(weights))
    sys.exit()

    loss_mtx = weights * loss_mtx
    return loss_mtx.mean()


l2_loss = nn.MSELoss()


def loss_bce(logits, Data_labels, LM_labels):
    assert logits.shape == LM_labels.shape

    batch_size = logits.shape[0]
    num_classes = logits.shape[1]
    loss_denom_mtx = (num_classes * batch_size) * torch.ones_like(logits)
    loss_denom_mtx = loss_denom_mtx.to(device)

    # compute loss:
    loss_mtx = torch.zeros_like(logits)
    loss_mtx[Data_labels == 1] = neg_log(logits[Data_labels == 1])
    loss_mtx[Data_labels == 0] = neg_log(1.0 - logits[Data_labels == 0])
    loss_mtx[Data_labels == -1] = l2_loss(logits[Data_labels == -1], LM_labels[Data_labels == -1])
    loss_mtx = loss_mtx.to(device)

    main_loss = (loss_mtx / loss_denom_mtx).sum()

    return main_loss


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
    loader, info = load_spml_datasets(
        dataset=args.dataset,
        model_type=args.model_type,
        pattern="train",
        input_size=args.input_size,
        batch_size=args.batch_size,
        num_workers=args.num_workers
    )
    taglist, imglist, tag_des = \
        info["taglist"], info["imglist"], info["tag_des"]

    test_loader, _ = load_spml_datasets(
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

    cla_model = nn.Linear(num_classes * 3, num_classes)
    cla_img_model = cls_network(input_size=512, hidden_size=256, output_size=num_classes)
    cla_gcn_model = nn.Linear(768, 1)
    text_model = GCN(input_dim=768, hidden_dim=256, output_dim=768)

    optimizer = torch.optim.AdamW(list(text_model.parameters())
                                  + list(cla_model.parameters())
                                  + list(cla_gcn_model.parameters())
                                  + list(cla_img_model.parameters()), lr=1e-2, weight_decay=0.05)

    ram_model.to(device).eval()
    text_model.to(device).train()
    cla_model.to(device).train()
    cla_gcn_model.to(device).train()
    cla_img_model.to(device).train()

    lambda1 = 10
    best_mAP = 0.0

    GCN_embed = []
    GCN_embed = torch.tensor(GCN_embed).to(device)
    for cat in catName_to_catID:
        img_path = "/root/SJML/prompt/voc/" + str(cat)
        text = clip.tokenize(cat).to(device)

        pre_GCN_embed = []
        pre_GCN_embed = torch.tensor(pre_GCN_embed).to(device)

        with torch.no_grad():
            text_features = model_clip.encode_text(text).float()
        pre_GCN_embed = torch.cat((pre_GCN_embed, text_features), dim=0)

        for picture_name in os.listdir(img_path):
            file_name = img_path + "/" + picture_name
            image = Image.open(file_name)
            image_input = preprocess(image).unsqueeze(0).to(device)
            with torch.no_grad():
                image_features = model_clip.encode_image(image_input).float()
                pre_GCN_embed = torch.cat((pre_GCN_embed, image_features), dim=0)

        GCN_embed_nor = F.normalize(pre_GCN_embed, dim=1)
        cos_sim = torch.matmul(GCN_embed_nor, GCN_embed_nor.transpose(0, 1)).to(device)
        Pre_text_model = GCN(input_dim=512, hidden_dim=256, output_dim=512).to(device)

        # 卷积+池化
        GCN_embed_ft = Pre_text_model(pre_GCN_embed, cos_sim) + pre_GCN_embed
        GCN_embed_ft = torch.mean(GCN_embed_ft, dim=0, keepdim=True)

        # 节点接在一起
        GCN_embed = torch.cat((GCN_embed, GCN_embed_ft), dim=0)

    # 20个节点的图进行卷积
    GCN_embed_final = F.normalize(GCN_embed, dim=1)
    cos_sim = torch.matmul(GCN_embed_final, GCN_embed_final.transpose(0, 1)).to(device)
    Pre_text_model = GCN(input_dim=512, hidden_dim=256, output_dim=512).to(device)
    GCN_embed_ft_final = Pre_text_model(GCN_embed, cos_sim) + GCN_embed
    label_embed = GCN_embed_ft_final.clone()

    # start training
    for epoch in range(args.epochs):

        torch.cuda.empty_cache()

        step_lr_schedule(optimizer, epoch, init_lr=1e-2, min_lr=5e-5, decay_rate=0.9)

        idx = 0

        for (imgs, labels) in tqdm(loader, desc="train"):

            optimizer.zero_grad()

            labels = labels.to(device)
            image_embeds = ram_model.image_proj(ram_model.visual_encoder(imgs.to(device)))
            image_atts = torch.ones(
                image_embeds.size()[:-1], dtype=torch.long).to(device)

            label_embed = relu(ram_model.wordvec_proj(ram_model.label_embed)).unsqueeze(0) \
                .repeat(imgs.shape[0], 1, 1)
            # 在这里更改/添加

            tagging_embed, _ = ram_model.tagging_head(
                encoder_embeds=label_embed,
                encoder_hidden_states=image_embeds,
                encoder_attention_mask=image_atts,
                return_dict=False,
                mode='tagging',
            )

            # campute adjacency_matrix

            # 开始建图
            # GPN_embed = GPN_label_embeds[0, :]
            label_embed_nor = F.normalize(label_embed, dim=2)
            cos_sim = torch.matmul(label_embed_nor, label_embed_nor.transpose(1, 2))

            # set topk, p, get sparse matrix and alleviate the over-smoothing
            topk = 5  # 可调
            p = 0.2  # 可调
            for i in range(cos_sim.shape[0]):
                matrix = cos_sim[i]
                topk_values, topk_indices = torch.topk(matrix, topk, dim=1)
                result = torch.zeros_like(matrix)
                result.scatter_(1, topk_indices, topk_values)
                matrix.copy_(result)

            # alleviate the over-smoothing
            for i in range(cos_sim.shape[0]):
                matrix = cos_sim[i]
                diagonal_elements = torch.diagonal(matrix)
                non_diagonal_sum = torch.sum(matrix, dim=1) - diagonal_elements
                scaling_factor = p / non_diagonal_sum
                for j in range(scaling_factor.shape[0]):
                    matrix[j, :] = matrix[j, :] * scaling_factor[j]
                matrix.fill_diagonal_(1 - p)
            # 结束建图

            # for i in range(cos_sim.shape[1]):
            #     for j in range(cos_sim.shape[2]):
            #         if i!=j:
            #             cos_sim[:,i,j]=0

            # print(cos_sim[0])
            # sys.exit()

            label_embed_ft = text_model(label_embed, cos_sim) + label_embed

            ram_logits = ram_model.fc(tagging_embed).squeeze(-1)
            img_logits = cla_img_model(image_embeds[:, 0, :]).squeeze(1)
            gcn_logits = cla_gcn_model(label_embed_ft).squeeze(-1)

            cat_fea = torch.cat((ram_logits, img_logits, gcn_logits), dim=1)

            fin_logits = cla_model(cat_fea)
            """
            sigmoid_logits = torch.sigmoid(fin_logits)
            sigmoid_ram_logits = torch.sigmoid(ram_logits)
            sigmoid_ram_logits[labels == 1] = 1
            sigmoid_ram_logits[labels == -1] = 0
            positive_index = []
            negative_index = []
            for i in range(labels.size()[0]):
                if labels[i].sum() == 1:
                    positive_index.append(i)
                else:
                    negative_index.append(i)
            positive_index = torch.tensor(positive_index).to(device)
            negative_index = torch.tensor(negative_index).to(device)
            labels_p = torch.index_select(labels, dim=0, index=positive_index)
            labels_n = torch.index_select(labels, dim=0, index=negative_index)
            sigmoid_logits_p = torch.index_select(sigmoid_logits, dim=0, index=positive_index)
            sigmoid_logits_n = torch.index_select(sigmoid_logits, dim=0, index=negative_index)
            sigmoid_ram_logits_p = torch.index_select(sigmoid_ram_logits, dim=0, index=positive_index)
            sigmoid_ram_logits_n = torch.index_select(sigmoid_ram_logits, dim=0, index=negative_index)

            loss1 = loss_p(sigmoid_logits_p, sigmoid_ram_logits_p, labels_p)
            loss2 = loss_n(sigmoid_logits_n, sigmoid_ram_logits_n, labels_n)
            """

            pseudo_label = sigmoid(ram_logits.detach()).float()
            fin_logits = sigmoid(fin_logits).float()

            loss = loss_bce(fin_logits, labels, pseudo_label)
            # loss.requires_grad_(True)
            # loss3 = l2_loss(fin_logits, pseudo_label)
            # print(loss1, loss2, sigmoid_logits_p, sigmoid_logits_n)
            # loss = loss1 + loss2

            loss.backward()
            optimizer.step()

            idx += 1

        # test and save checkpoint
        mAP, APs = test_model(ram_model, cla_model, cla_img_model, cla_gcn_model, text_model, test_loader, taglist)

        if mAP >= best_mAP:
            save_obj = {
                'cla_model': cla_model.state_dict(),
                'cla_img_model': cla_img_model.state_dict(),
                'cla_gcn_model': cla_gcn_model.state_dict(),
                'text_model': text_model.state_dict(),
                'epoch': epoch,
                'mAP': mAP,
                'APs': APs
            }

            torch.save(save_obj, os.path.join(output_dir, 'checkpoint_best.pth'))

            with open(ap_file, "w", encoding="utf-8") as f:
                f.write("Tag,AP\n")
                for tag, AP in zip(taglist, APs):
                    f.write(f"{tag},{AP * 100.0:.2f}\n")

            with open(summary_file, "a", encoding="utf-8") as f:
                print_write(f, f"mAP: {mAP * 100.0}")

            best_mAP = mAP

        print(f"Epoch : {epoch} | Loss : {loss.item()} | mAP : {mAP}")
