# Copyright (c) 2018, Curious AI Ltd. All rights reserved.
#
# This work is licensed under the Creative Commons Attribution-NonCommercial
# 4.0 International License. To view a copy of this license, visit
# http://creativecommons.org/licenses/by-nc/4.0/ or send a letter to
# Creative Commons, PO Box 1866, Mountain View, CA 94042, USA.

"""Custom loss functions"""

import torch
from torch.nn import functional as F
from torch.autograd import Variable
import pdb
import numpy as np

def softmax_mse_loss(input_logits, target_logits):
    """Takes softmax on both sides and returns MSE loss

    Note:
    - Returns the sum over all examples. Divide by the batch size afterwards
      if you want the mean.
    - Sends gradients to inputs but not the targets.
    """
    assert input_logits.size() == target_logits.size()
    input_softmax = F.softmax(input_logits, dim=1)
    target_softmax = F.softmax(target_logits, dim=1)
    num_classes = input_logits.size()[1]
    return F.mse_loss(input_softmax, target_softmax, size_average=False) / num_classes


def softmax_kl_loss(input_logits, target_logits):
    """Takes softmax on both sides and returns KL divergence

    Note:
    - Returns the sum over all examples. Divide by the batch size afterwards
      if you want the mean.
    - Sends gradients to inputs but not the targets.
    """
    assert input_logits.size() == target_logits.size()
    input_log_softmax = F.log_softmax(input_logits, dim=1)
    target_softmax = F.softmax(target_logits, dim=1)
    return F.kl_div(input_log_softmax, target_softmax, size_average=False)


def symmetric_mse_loss(input1, input2):
    """Like F.mse_loss but sends gradients to both directions

    Note:
    - Returns the sum over all examples. Divide by the batch size afterwards
      if you want the mean.
    - Sends gradients to both input1 and input2.
    """
    assert input1.size() == input2.size()
    num_classes = input1.size()[1]
    return torch.sum((input1 - input2)**2) / num_classes


def distance_vectors_pairwise(anchor, positive, negative , squared = True):
    """Given batch of anchor descriptors and positive descriptors calculate distance matrix"""

    a_sq = torch.sum(anchor * anchor, dim=1)
    p_sq = torch.sum(positive * positive, dim=1)
    n_sq = torch.sum(negative * negative, dim=1)

    eps = 1e-8
    # d_a_p = torch.sqrt(a_sq + p_sq - 2*torch.sum(anchor * positive, dim = 1) + eps)
    # d_a_n = torch.sqrt(a_sq + n_sq - 2*torch.sum(anchor * negative, dim = 1) + eps)
    # d_p_n = torch.sqrt(p_sq + n_sq - 2*torch.sum(positive * negative, dim = 1) + eps)


    d_a_p = a_sq + p_sq - 2*torch.sum(anchor * positive, dim = 1)
    d_a_n = a_sq + n_sq - 2*torch.sum(anchor * negative, dim = 1)
    d_p_n = p_sq + n_sq - 2*torch.sum(positive * negative, dim = 1)

    if not squared:
        d_a_p = torch.sqrt(d_a_p + eps)
        d_a_n = torch.sqrt(d_a_n + eps)
        d_p_n = torch.sqrt(d_p_n + eps)

    return d_a_p, d_a_n, d_p_n

def pdist(A, squared = False, eps = 1e-4):
    prod = torch.mm(A, A.t())
    norm = prod.diag().unsqueeze(1).expand_as(prod)
    res = (norm + norm.t() - 2 * prod).clamp(min = 0)
    return res if squared else res.clamp(min = eps).sqrt()

def triplet_loss_old(features, labels, margin = 1.0, isSquared = True, weights = None):

    valid_idx = labels.ne(-1)
    valid_feats = features[valid_idx,:]
    valid_labels = labels[valid_idx]
    # valid_dist = F.pairwise_distance(valid_feats,valid_feats,2)

    d = pdist(valid_feats, squared = isSquared)
    # pos = torch.eq(*[valid_labels.unsqueeze(dim).expand_as(d) for dim in [0, 1]]).type_as(d) - torch.autograd.Variable(torch.eye(len(d))).type_as(d)
    pos = torch.eq(*[valid_labels.unsqueeze(dim).expand_as(d) for dim in [0, 1]]).type_as(d)
    
    T = d.unsqueeze(1).expand(*(len(d),) * 3)
    M = pos.unsqueeze(1).expand_as(T) * (1 - pos.unsqueeze(2).expand_as(T))
    # loss = (M * F.relu(T - T.transpose(1, 2) + margin)).sum() / M.sum()

    # pdb.set_trace()    
    # return loss
    return ((M * F.relu(T - T.transpose(1, 2) + margin)).sum(1).sum(1) , M.sum())


def liftedstruct_loss(features, labels, margin = 1.0, isSquared = False, weights = None):
    valid_idx = labels.ne(-1)
    valid_feats = features[valid_idx,:]
    valid_labels = labels[valid_idx]


    d = pdist(valid_feats, squared = isSquared)
    pos = torch.eq(*[valid_labels.unsqueeze(dim).expand_as(d) for dim in [0, 1]]).type_as(d)
    neg_i = torch.mul((margin - d).exp(), 1 - pos).sum(1).expand_as(d)

    # pdb.set_trace()

    # return torch.sum(F.relu(pos.triu(1) * ((neg_i + neg_i.t()).log() + d)).pow(2)) / (pos.sum() - len(d))
    return (torch.sum(F.relu(pos.triu(1) * ((neg_i + neg_i.t()).log() + d)).pow(2),1) , pos.sum() - len(d))

def contrastive_loss_old(features, labels, margin = 1.0, isSquared = True, weights = None):

    valid_idx = labels.ne(-1)
    valid_feats = features[valid_idx,:]
    valid_labels = labels[valid_idx]
    # valid_dist = F.pairwise_distance(valid_feats,valid_feats,2)

    d = pdist(valid_feats, squared = isSquared)
    # pos = torch.eq(*[valid_labels.unsqueeze(dim).expand_as(d) for dim in [0, 1]]).type_as(d) - torch.autograd.Variable(torch.eye(len(d))).type_as(d)
    pos = torch.eq(*[valid_labels.unsqueeze(dim).expand_as(d) for dim in [0, 1]]).type_as(d)
    neg = torch.ne(*[valid_labels.unsqueeze(dim).expand_as(d) for dim in [0, 1]]).type_as(d)
    

    pweights = torch.ger(weights,weights)

    T = d.unsqueeze(1).expand(*(len(d),) * 3)
    # M = pos.unsqueeze(1).expand_as(T) * (1 - pos.unsqueeze(2).expand_as(T))
    Mp = pos.unsqueeze(1).expand_as(T) 
    Meye = torch.autograd.Variable(torch.eye(len(d))).type_as(d).expand_as(T) 

    Mpos = Meye * Mp
    # loss_pos = (Mpos * T).sum() / Mpos.sum()

    Mneg = Meye * torch.abs(Mp-1)
    # loss_neg = (Mneg * F.relu(margin - T)).sum() / Mneg.sum()

    pdb.set_trace()
    loss_pos = (Mpos * T).sum(1).sum(1)
    loss_neg = (Mneg * F.relu(margin - T)).sum(1).sum(1)

    loss = loss_pos + loss_neg
    pdb.set_trace()

    return loss


def contrastive_loss(features, labels, margin = 1.0, isSquared = True, weights = None):

    valid_idx = labels.ne(-1)
    valid_feats = features[valid_idx,:]
    valid_labels = labels[valid_idx]
    # valid_dist = F.pairwise_distance(valid_feats,valid_feats,2)

    d = pdist(valid_feats, squared = isSquared)
    pos_loss = 0
    neg_loss = 0
    pos_count = 0
    neg_count = 0

    for i in range(len(valid_idx)):
        cur_idx = valid_idx[i].item()
        cur_label = valid_labels[i]
        cur_pos = labels.eq(cur_label)
        cur_neg = labels.ne(cur_label)

        # pdb.set_trace()
        cur_pos_dist = d[cur_idx,cur_pos]
        cur_pos_w = weights[cur_idx] * weights[cur_pos]
        cur_pos_dist = cur_pos_dist * cur_pos_w

        cur_neg_dist = F.relu(d[cur_idx,cur_neg] - margin)
        cur_neg_w = weights[cur_idx] * weights[cur_neg]
        cur_neg_dist = cur_neg_dist * cur_neg_w

        pos_loss += cur_pos_dist.sum()
        neg_loss += cur_neg_dist.sum()

        pos_count += cur_pos.sum() - 1
        neg_count += cur_neg.sum()

    # pdb.set_trace()
    loss = pos_loss / pos_count.float() + neg_loss / neg_count.float()

    return loss


def triplet_loss(features, labels, margin = 1.0, isSquared = True, weights = None):

    valid_idx = labels.ne(-1)
    valid_feats = features[valid_idx,:]
    valid_labels = labels[valid_idx]
    # valid_dist = F.pairwise_distance(valid_feats,valid_feats,2)

    d = pdist(valid_feats, squared = isSquared)

    count = 0
    loss = 0

    for i in range(len(valid_idx)):
        cur_idx = valid_idx[i].item()
        cur_label = valid_labels[i]
        cur_pos = labels.eq(cur_label)
        cur_neg = labels.ne(cur_label)

        cur_pos_dist = d[cur_idx,cur_pos]
        cur_neg_dist = d[cur_idx,cur_neg]

        # pdb.set_trace()

        pos_weights = weights[cur_pos]
        neg_weights = weights[cur_neg]


        total_pos = cur_pos.sum().item()
        total_neg = cur_neg.sum().item()

        for j in range(total_pos):
            cur_weights = weights[cur_idx] * weights[cur_pos][j] * neg_weights
            cur_loss = F.relu( margin + cur_pos_dist[j] - cur_neg_dist) * cur_weights

            loss += cur_loss.sum()
            count += total_neg

    # pdb.set_trace()
    loss = loss / float(count)
    return loss
