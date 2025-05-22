import torch
import torch.nn as nn
import torch.nn.functional as F
import math 

device = "cuda" if torch.cuda.is_available() else "cpu"


LOG_EPSILON = 1e-5


def neg_log(x):
    return - torch.log(x + LOG_EPSILON)


def loss_an(logits, obs_labels):
    
    assert logits.shape==obs_labels.shape
    
    batch_size = logits.shape[0]
    num_classes = logits.shape[1]
    loss_denom_mtx = (num_classes * batch_size) * torch.ones_like(logits)
    loss_denom_mtx = loss_denom_mtx.to(device)

    # compute loss:
    loss_mtx = torch.zeros_like(logits)
    loss_mtx[obs_labels == 1] = neg_log(logits[obs_labels == 1])
    loss_mtx[obs_labels == 0] = neg_log(1.0 - logits[obs_labels == 0])
    loss_mtx = loss_mtx.to(device)
    
    main_loss = (loss_mtx / loss_denom_mtx).sum()

    return main_loss

def loss_an_logits(logits, observed_labels):

    assert torch.min(observed_labels) >= 0
    
    batch_size = logits.shape[0]
    num_classes = logits.shape[1]
    loss_denom_mtx = (num_classes * batch_size) * torch.ones_like(logits)
    loss_denom_mtx = loss_denom_mtx.to(device)
    
    # compute loss:
    loss_matrix = F.binary_cross_entropy_with_logits(logits, observed_labels.float(), reduction='none')
    loss_matrix =loss_matrix.to(device)
    
    main_loss = (loss_matrix / loss_denom_mtx).sum()
    
    return loss_matrix, main_loss


def loss_asl(logits, obs_labels):
    assert logits.shape == obs_labels.shape

    gamma_pos = 4
    gamma_neg = 1

    batch_size = logits.shape[0]
    num_classes = logits.shape[1]
    loss_denom_mtx = (num_classes * batch_size) * torch.ones_like(logits)
    loss_denom_mtx = loss_denom_mtx.to(device)

    # compute loss:
    loss_mtx = torch.zeros_like(logits)
    loss_mtx[obs_labels == 1] = (1.0-logits[obs_labels == 1]) ** gamma_pos * neg_log(logits[obs_labels == 1])
    loss_mtx[obs_labels == 0] = (logits[obs_labels == 0]) ** gamma_neg * neg_log(1.0 - logits[obs_labels == 0])
    loss_mtx = loss_mtx.to(device)

    main_loss = (loss_mtx / loss_denom_mtx).sum()

    return main_loss



def loss_bce(logits, obs_labels):
    
    assert logits.shape == obs_labels.shape
    
    batch_size = logits.shape[0]
    num_classes = logits.shape[1]
    loss_denom_mtx = (num_classes * batch_size) * torch.ones_like(logits)
    loss_denom_mtx = loss_denom_mtx.to(device)

    # compute loss:
    loss_mtx = torch.zeros_like(logits)
    loss_mtx[obs_labels == 1] = neg_log(logits[obs_labels == 1])
    loss_mtx[obs_labels == 0] = neg_log(1.0 - logits[obs_labels == 0])
    loss_mtx = loss_mtx.to(device)
    
    main_loss = (loss_mtx / loss_denom_mtx).sum()

    return main_loss


def loss_EM(logits, obs_labels):
    
    assert logits.shape==obs_labels.shape
    
    alpha = 0.01
    
    batch_size = logits.shape[0]
    num_classes = logits.shape[1]
    loss_denom_mtx = (num_classes * batch_size) * torch.ones_like(logits)
    loss_denom_mtx = loss_denom_mtx.to(device)

    # compute loss:
    loss_mtx = torch.zeros_like(logits)
    loss_mtx[obs_labels == 1] = neg_log(logits[obs_labels == 1])
    loss_mtx[obs_labels == 0] = -alpha * ( logits[obs_labels == 0] * neg_log(logits[obs_labels == 0]) +
                                          (1 - logits[obs_labels == 0]) * neg_log(1.0 - logits[obs_labels == 0]) )
    loss_mtx = loss_mtx.to(device)
    
    main_loss = (loss_mtx / loss_denom_mtx).sum()

    return main_loss


def loss_EM_APL(logits, obs_labels):
    
    assert torch.min(obs_labels) >= -1
    
    alpha = 0.01
    beta = 0.4
    
    batch_size = logits.shape[0]
    num_classes = logits.shape[1]
    loss_denom_mtx = (num_classes * batch_size) * torch.ones_like(logits)
    loss_denom_mtx = loss_denom_mtx.to(device)

    loss_mtx = torch.zeros_like(logits)

    loss_mtx[obs_labels == 1] = neg_log(logits[obs_labels == 1])
    loss_mtx[obs_labels == 0] = -alpha * (
            logits[obs_labels == 0] * neg_log(logits[obs_labels == 0]) +
            (1 - logits[obs_labels == 0]) * neg_log(1 - logits[obs_labels == 0])
        )

    soft_label = (-obs_labels[obs_labels < 0]).float()
    loss_mtx[obs_labels < 0] = beta * (
            soft_label * neg_log(logits[obs_labels < 0]) +
            (1 - soft_label) * neg_log(1 - logits[obs_labels < 0])
        )
    
    loss_mtx = loss_mtx.to(device)
    main_loss = (loss_mtx / loss_denom_mtx).sum()
    
    return main_loss


def loss_VLPL(logits, pseudo_labels, obs_labels):
    
    assert logits.shape == pseudo_labels.shape == obs_labels.shape
    
    alpha = 0.2
    beta = 0.7
    
    batch_size = logits.shape[0]
    num_classes = logits.shape[1]
    loss_denom_mtx = (num_classes * batch_size) * torch.ones_like(logits)
    loss_denom_mtx = loss_denom_mtx.to(device)
    
    # compute loss:
    loss_mtx = torch.zeros_like(logits)
    loss_mtx[obs_labels == 1] = neg_log(logits[obs_labels == 1])
    loss_mtx[obs_labels == 0] = -alpha * (logits[obs_labels == 0] * neg_log(logits[obs_labels == 0]) +
                                          (1 - logits[obs_labels == 0]) * neg_log(1.0 - logits[obs_labels == 0]) )
    loss_mtx[pseudo_labels == 1] = beta * neg_log(logits[pseudo_labels == 1])
    loss_mtx = loss_mtx.to(device)
    
    main_loss = (loss_mtx / loss_denom_mtx).sum()

    return main_loss


def loss_LL_R(preds, label_vec, epoch):
    
    assert preds.dim() == 2
    
    batch_size = int(preds.size(0))
    num_classes = int(preds.size(1))
    
    unobserved_mask = (label_vec == 0)
    
    # compute loss for each image and class:
    loss_matrix, _ = loss_an_logits(preds, label_vec.clip(0))
    
    if epoch == 0: 
        final_loss_matrix = loss_matrix
    else:
        k = math.ceil(batch_size * num_classes * (epoch*0.002))
        unobserved_loss = unobserved_mask.bool() * loss_matrix
        topk = torch.topk(unobserved_loss.flatten(), k)
        topk_lossvalue = topk.values[-1]
        zero_loss_matrix = torch.zeros_like(loss_matrix)
        final_loss_matrix = torch.where(unobserved_loss < topk_lossvalue, loss_matrix, zero_loss_matrix)
        
    main_loss = final_loss_matrix.mean()
    
    return main_loss


