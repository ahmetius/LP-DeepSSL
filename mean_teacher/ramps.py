# Copyright (c) 2018, Curious AI Ltd. All rights reserved.
#
# This work is licensed under the Creative Commons Attribution-NonCommercial
# 4.0 International License. To view a copy of this license, visit
# http://creativecommons.org/licenses/by-nc/4.0/ or send a letter to
# Creative Commons, PO Box 1866, Mountain View, CA 94042, USA.

"""Functions for ramping hyperparameters up or down

Each function takes the current training step or epoch, and the
ramp length in the same format, and returns a multiplier between
0 and 1.
"""


import numpy as np


def sigmoid_rampup(current, rampup_length):
    """Exponential rampup from https://arxiv.org/abs/1610.02242"""
    if rampup_length == 0:
        return 1.0
    else:
        current = np.clip(current, 0.0, rampup_length)
        phase = 1.0 - current / rampup_length
        return float(np.exp(-5.0 * phase * phase))


def linear_rampup(current, rampup_length):
    """Linear rampup"""
    assert current >= 0 and rampup_length >= 0
    if current >= rampup_length:
        return 1.0
    else:
        return current / rampup_length


def cosine_rampdown(current, rampdown_length):
    """Cosine rampdown from https://arxiv.org/abs/1608.03983"""
    assert 0 <= current <= rampdown_length
    return float(.5 * (np.cos(np.pi * current / rampdown_length) + 1))


# linearly decrease the value
def linear_rampdown(current, rampdown_length):
    """Linear rampdown"""
    if current >= rampdown_length:
        return 1.0 - current / rampdown_length
    else:
        return 1.0


####ADAM
# exponentially decrease the value
def exponential_decrease(current, rampdown_length, total_epochs, scale = 1.0):
    if current <= rampdown_length:
        return 1.0
    else:
        phase = scale * (current - rampdown_length) / (total_epochs - rampdown_length)
        return float(np.exp(-5.0 * phase * phase))



# exponentially increase the value
def exponential_increase(current, rampup_length):
    if current >= rampup_length:
        return 1.0
    else:
        phase = 1 - current / rampup_length
        return float(np.exp(-5.0 * phase * phase))



# adjust lambda_r according to the paper
def adjust_lambda_r(current, rampup_length, rampdown_length, total_epochs):
    if current <= rampup_length:
        phase = 1 - current / rampup_length
        return float(np.exp(-5.0 * phase * phase))
    elif current >= rampdown_length:
        phase = (current - rampdown_length) / (total_epochs - rampdown_length)
        return float(np.exp(-5.0 * phase * phase))
    else:
        return 1.0



# adjust learning rate according to the paper
def adjust_learning_rate_adam(optimizer, initial_lr, epoch, total_epochs):
    # LR warm-up to handle large minibatch sizes from https://arxiv.org/abs/1706.02677
    lr = initial_lr * linear_rampdown(epoch, total_epochs - total_epochs/3)

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr



# adjust beta1 of Adam according to the paper
def adjust_beta_1(optimizer, initial_beta1, epoch, total_epochs):
    beta1 = initial_beta1 * exponential_decrease(epoch, 0.8 * total_epochs, total_epochs, scale = 0.5)
    
    for param_group in optimizer.param_groups:
        _ , beta2 = param_group['betas']
        param_group['betas'] = (beta1, beta2)
