#!/usr/bin/env python
# from sqlite3 import paramstyle
# from warnings import simplefilter
# simplefilter(action='ignore', category=FutureWarning)
# external
import os
import sys
import inspect
import logging
logger = logging.getLogger(__name__)
import itertools
import functools
import shutil
import json
import importlib
from datetime import datetime
import math
import random
import numpy as np
import pandas as pd
from inspect import getmembers, isclass
from pathlib import Path
from functools import partial as func_partial
from tqdm import tqdm
from sklearn.model_selection import train_test_split

import paddle as mi
import paddle.nn as nn
from paddle.io import Dataset
import paddle.nn.functional as F

# homebrew
import misc
import gwio
import minets_paddle as MiNets
import mitas_utils
import brew_midat


class MyDataset(Dataset):
    """Two modes of operation:
            1) data is a list of samples ready for DataLoader (no item_getter needed)
            2) data is a dataframe to be processed by item_getter (a function)
    """
    def __init__(self, data, item_getter=None):
        super(MyDataset, self).__init__()
        self.data = data # the passed data
        self.data_jit = dict() # used to store baked jit data if item_getter is set
        self.data_jit_full = False # indicator of if all jit data have been baked
        self.item_getter = item_getter

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if self.item_getter is None:
            return self.data[idx]
        elif self.data_jit_full:
            return self.data_jit[idx]
        else:
            if idx not in self.data_jit: # not read yet
                # print(f'JIT Loader: {idx}')
                self.data_jit[idx] = self.item_getter(self.data.iloc[idx:idx+1])[0]
                if len(self.data) == len(self.data_jit): self.data_jit_full = True
            return self.data_jit[idx]


def get_dataset(df, args=misc.Struct(data_genre=None), **kwargs):
    """Obtain a MyDataSet object required for DataLoader.

    Arguments:
        df: the dataframe returned by mitas.load_midat_???
        args: the all-inclusive structure
        kwargs: kwargs > args
    """
    args.update(kwargs)

    if args.data_genre is None or args.data_genre.lower() == 'none':
        return MyDataset(df)

    args.data_genre = args.data_genre.lower()

    if args.data_genre in ['rna2d', 'contarna']:
        baker_fn = mitas_utils.bake_midat_contarna
    elif args.data_genre == 'ab2021':
        baker_fn = mitas_utils.bake_midat_ab2021
    else:
        baker_fn = lambda x: x
        logger.warning(f'data_genre: {args.data_genre} has no baker function!!!')

    if args.jit_loader:
        return MyDataset(df, item_getter=func_partial(baker_fn, args=args))
    else:
        return MyDataset(baker_fn(df, args))


def get_dataloader(miset, **kwargs):
    """A simple wrapper for DataLoader """
    # num_gpus = len([_device for _device in mi.device.get_available_device() if ':' in _device])

    num_gpus = mi.device.cuda.device_count()

    loader_opts = dict(
        shuffle = False,
        drop_last = False,
        batch_size = 1,
        timeout = 60, # appears to be in seconds?
        num_workers = min([4 * min([1, num_gpus]), os.cpu_count()]), # 4*num_gpus?
        # False will corrupt cuda version somehow
        use_buffer_reader = True, # mi.device.get_device() == 'gpu',
        use_shared_memory = True,
    )
    loader_opts.update(kwargs)

    return mi.io.DataLoader(miset, **loader_opts)


def get_optimizer(minet, args):
    """Obtain the optimizer and lr_scheduler (or learning_rate) """
    misc.logger_setlevel(logger, args.verbose)

    if args.lr_scheduler is None or args.lr_scheduler.lower().startswith('non'):
        learning_rate = args.learning_rate
    elif args.lr_scheduler.lower().startswith('reduced'):
        learning_rate = mi.optimizer.lr.ReduceOnPlateau(
            learning_rate = args.learning_rate,
            mode = 'min',
            factor = args.lr_factor,
            patience = args.lr_patience,
            threshold = 3e-4,
            threshold_mode = 'rel',
            min_lr = 1e-7,
            cooldown = 1,
            verbose = True,
            )
    else:
        logger.error(f'Cannot recognize args.lr_scheduler: {args.lr_scheduler}')

    if args.grad_clip is not None:
        args.grad_clip = args.grad_clip.lower()
    if args.grad_clip in ['globalnorm', 'global_norm']:
        grad_clip = mi.nn.ClipGradByGlobalNorm(clip_norm=1.0)
    elif args.grad_clip in ['value']:
        grad_clip = mi.nn.ClipGradByValue(min=-7, max=7)
    elif args.grad_clip in ['norm']:
        grad_clip = mi.nn.ClipGradByNorm(clip_norm=4.2)
    else:
        grad_clip = None

    optim_name = args.optim_fn.upper()
    if optim_name == 'ADAMW':
        # Generate parameter names needed to perform weight decay.
        # All bias and LayerNorm parameters are excluded.
        # decay_params = [
        #     p.name for n, p in minet.named_parameters()
        #     if not any(nd in n for nd in ["bias", "norm"])
        #     ]
        mi_opt = mi.optimizer.AdamW(
            parameters = minet.parameters(),
            weight_decay = 0.01 if not isinstance(args.weight_decay, float) else args.weight_decay,
            learning_rate = learning_rate,
            beta1 = args.beta1,
            beta2 = args.beta2,
            epsilon = args.epsilon,
            grad_clip = grad_clip,
            apply_decay_param_fun=None, # lambda x: x in decay_params,
            lazy_mode = False,
        )
    elif optim_name == 'ADAM':
        weight_decay = None
        if args.l1decay:
            weight_decay = mi.regularizer.L1Decay(args.l1decay)
        if args.l2decay:
            if weight_decay is not None:
                logger.warning(f'L2Decay ({args.l2decay} overwrites L1Decay ({args.l1decay}) !')
            weight_decay = mi.regularizer.L2Decay(args.l2decay)

        mi_opt = mi.optimizer.Adam(
            parameters = minet.parameters(),
            weight_decay = weight_decay,
            learning_rate = learning_rate,
            beta1 = args.beta1,
            beta2 = args.beta2,
            epsilon = args.epsilon,
            grad_clip = grad_clip,
            lazy_mode = False,
            )
    else:
        logger.critical(f'Unrecognized optim_fn: {args.optim_fn} !!!')

    # logger.info(f'======= Optimizer =======')
    logger.info(f'======= {inspect.currentframe().f_code.co_name} =======')
    logger.info(f'          method: {args.optim_fn}')
    logger.info(f'     step_stride: {args.optim_step_stride}')
    logger.info(f'       grad clip: {args.grad_clip}')
    logger.info(f'   learning rate: {args.learning_rate}')
    logger.info(f'    lr scheduler: {args.lr_scheduler}')
    logger.info(f'       lr factor: {args.lr_factor}')
    logger.info(f'     lr patience: {args.lr_patience}')
    logger.info(f' lr warmup steps: {args.lr_warmup_steps}')
    logger.info(f'    weight decay: {args.weight_decay}')
    logger.info(f'         l1decay: {args.l1decay}')
    logger.info(f'         l2decay: {args.l2decay}')

    if args.fleet:
        return fleet.distributed_optimizer(mi_opt), learning_rate
    else:
        return mi_opt, learning_rate


def soft_fscore(guess, label, batch=False, symmetric=False, epsilon=1e-8, mi=mi,
        beta=1.0, auto_beta=False, auto_beta_mode='length', auto_beta_pow=1.0,
        bpp=False, bpp_mask=None, bpp_scale=1.0, bpp_threshold=0.5,
        l2=False, l2_mask=None, l2_scale=1.0, l2_threshold=0.5, # 0.5 may be too harsh
        **kwargs):
    """ beta2=beta is used to weight recall relative to precision. What it does
        is to scale FN with beta2 relative to FP, so with beta>1, it penalizes FN beta^2
        times as much as FP.

        It MAY BE an issue when all labels are negative, which leads to fscore=0
        regardless of the input. This may not be an issue though.
    """
    if kwargs: logger.error(f'kwargs is supposed to be empty: {kwargs}!!!')
    assert guess.ndim == label.ndim, 'Input and Label must have the same ndims!'

    if batch:
        sum_axis = list(range(1, guess.ndim))
        total = guess[0].size
        p = label.sum(sum_axis) # all positive
        pp = guess.sum(sum_axis) # all predicted positive
        tp = (guess * label).sum(sum_axis) # true positive
    else:
        sum_axis = list(range(0, guess.ndim))
        total = guess.size
        p = label.sum() # all positive
        pp = guess.sum() # all predicted positive
        tp = (guess * label).sum() # true positive
    epsilon *= total

    # fp = pp - tp # (input * (1.0 - label)).sum(sum_axis)
    fn = p - tp # ((1.0 - input) * label).sum(sum_axis)
    tn = total - pp - fn # ((1 - input) * (1 - label)).sum()

    # beta value is essentially the level of preference of fp over fn
    # I cannot find a good way to set this value. It should in principle
    # depened on the ratio of negative/positive in the label
    if auto_beta:
        if auto_beta_mode in ['npratio', 'np', 'ratio']:
            # beta2 = total / (p + 1.0) / 2
            beta2 = total / max([1.0, p]) - 1.0
        elif auto_beta_mode in ['length', 'len']:
            beta2 = label.shape[1] - 1
        else:
            beta2 = 1.0
            logger.error(f'Unsupported auto_beta_mode: {auto_beta_mode}!!!')

        if beta2 != 1.0 and auto_beta_pow != 1.0:
            beta2 = math.pow(beta2, auto_beta_pow)
        # if batch: # p is an array of numpy or torch or paddle (maybe should check which type)
        #     beta2 = math.pow(label.shape[1] - 1, auto_beta_pow)
        # else:
        #     beta2 = math.pow(label.shape[1] - 1, auto_beta_pow)
    else:
        beta2 = beta # * beta

    if beta2 == 1.0:
        if not symmetric:
            # (p + pp) == (2.0 * tp + fp + fn)
            fscore = (2.0 * tp + epsilon) / (p + pp + epsilon)
        else:
            # Add fscore for the negative (not sure how much this helps)
            # tn -> tp, fn -> fp, fp -> fn, tp -> tn
            fscore = (tp + epsilon) / (p + pp + epsilon) + \
                     (tn + epsilon) / (2.0 * total - p - pp + epsilon)
    else:
        if not symmetric:
            fscore = ((1.0 + beta2) * tp + epsilon) / (beta2 * p + pp + epsilon)
        else:
            fscore = 0.5 * (
                     ((1.0 + beta2) * tp + epsilon) / (beta2 * p + pp + epsilon) + \
                     ((1.0 + beta2) * tn + epsilon) / ((1.0 + beta2) * total - beta2 * p - pp + epsilon)
                     )
    loss = 1.0 - fscore

    # bpp_loss basically deals with the pairing of a single base with all others
    if bpp:
        # threshold * 2 doesn't work well when threshold < 0.5
        # divide by the number of positive labels + 1
        bpp_loss = F.relu(F.thresholded_relu(guess, bpp_threshold).sum(-1) - bpp_threshold * 2.).sum() / (p + 1)

        # bpp_loss = F.mse_loss(
        #     F.thresholded_relu(F.thresholded_relu(guess, bpp_threshold).sum(-1), bpp_threshold * 2),
        #     label.sum(-1), reduction='mean')
            # reduction='none').mean(sum_axis[:-1]) # only necessary when batch_size > 1
        if bpp_scale == 1.0:
            loss += bpp_loss
        else:
            loss += bpp_scale * bpp_loss

    if l2 and l2_mask is not None:
        mask_mat = l2_mask(guess.shape[-2:])
        l2_loss = F.thresholded_relu(guess * mask_mat, l2_threshold).sum(sum_axis) / mi.sum(mask_mat)
        if l2_scale == 1.0:
            loss += l2_loss
        else:
            loss += l2_scale * l2_loss

    return loss

def rmse(input, label, batch=False):
    """ unnecessary """
    pass

def rmse_metric(input, label, batch=False):
    """ returns np.scalar (batch=False) or np.ndarray of (batch=True) """

    rmse = F.mse_loss(input, label, reduction='none')
    mape = F.l1_loss(input, label, reduction='none') / (mi.abs(label) + 0.1)

    if batch:
        sum_axis = list(range(1, rmse.ndim))
        rmse = np.array(rmse.sum(sum_axis), dtype='float32')
        mape = np.array(mape.sum(sum_axis), dtype='float32')
    else:
        rmse = np.array(rmse.sum(), dtype='float32')
        mape = np.array(mape.sum(), dtype='float32')

    return np.stack([np.sqrt(rmse), mape * 100.0], axis=-1).astype('float32')


def masked_rmse_metric(input, label, batch=False):
    """ returns np.scalar of the rmse, does nothing special for the batch_size dim
    An ad hoc convention:
        the last dim of label is ordered as
            0: mask
            1: raw label (e.g., angle)
            2: sin
            3: cos
    """
    if input.ndim == label.ndim - 1: # predicting angle directly
        angle = input
    elif input.ndim == label.ndim and input.shape[-1] == 2: # predicting sin and cos
        radius = mi.norm(input, axis=-1, keepdim=True)
        input /= radius        
        angle = mi.atan2(input[...,0], input[...,1]) * 57.2957795 # conver to degrees
    else:
        logger.critical(f'Unsupported!')
        return 0.0

    rmse = F.mse_loss(angle, label[...,1], reduction='none')
    rmse *= label[...,0]

    # absolute percentage error
    mape = F.l1_loss(angle, label[...,1], reduction='none') / \
            (mi.abs(label[...,1]) + 0.1) # 0.1 is arbitrary
    mape *= label[...,0]

    if batch:
        sum_axis = list(range(1, rmse.ndim))
        num_angles = label[...,0].sum(sum_axis)
        rmse = np.array(rmse.sum(sum_axis) / num_angles, dtype='float32')
        mape = np.array(mape.sum(sum_axis) / num_angles, dtype='float32')
    else:
        num_angles = label[...,0].sum()
        rmse = np.array(rmse.sum() / num_angles, dtype='float32')
        mape = np.array(mape.sum() / num_angles, dtype='float32')

    return np.stack([np.sqrt(rmse), mape * 100.0], axis=-1)


def masked_angle_mse(input, label, batch=False, radius0=mi.to_tensor(1.0, dtype='float32')):
    """ returns a scalar of the averaged loss, so it does nothing special
        about the batch_size dim
    An ad hoc convention:
        the last dim of label is ordered as
            0: mask
            1: raw label (e.g., angle)
            2: sin
            3: cos
    """
    radius_loss = 0.0
    if input.ndim == label.ndim - 1: # predicting angle directly
        loss = F.mse_loss(input, label[...,1], reduction='none')
        loss *= label[...,0]
        # loss = loss.sum() / label[...,0].sum()
    elif input.ndim == label.ndim and input.shape[-1] == 2: # predicting sin and cos
        # the radius of the circle
        radius = mi.norm(input, p=2, axis=-1, keepdim=True)
        input /= radius
        radius_loss = 0.02 * F.l1_loss(radius * label[...,0:1], label[...,0:1], reduction='mean')
        loss = F.mse_loss(input, label[...,2:4], reduction='none')
        loss *= label[...,0:1]
    else:
        logger.critical(f'Unsupported!')
        loss = 0.0

    if batch:
        sum_axis = list(range(1, loss.ndim))
        loss = loss.sum(sum_axis) / label[...,0].sum(sum_axis)
    else:
        loss = loss.sum() / label[...,0].sum()

    loss += radius_loss
    return loss


def sigmoid_mse(input, label, reduction='none'):
    input = F.sigmoid(input)
    loss = F.mse_loss(input, label, reduction=reduction)
    return loss


def softmax_mse(input, label, label_col=1, reduction='none'):
    """ this only makes sense for input.shape[-1]=2
    label_col is only used if input.ndim == label.ndim + 1
    """

    input = F.softmax(input, axis=-1)

    # only take one axis for loss calculation
    if input.ndim == label.ndim + 1:
        if input.ndim == 2: # yet to find a better way, tensor doesn't accept [...,label_col]
            input = input[:, label_col].squeeze(-1)
        elif input.ndim == 3:
            input = input[:, :, label_col].squeeze(-1)
        elif input.ndim == 4:
            input = input[:, :, :, label_col].squeeze(-1)
        elif input.ndim == 5:
            input = input[:, :, :, :, label_col].squeeze(-1)
        elif input.ndim == 6:
            input = input[:, :, :, :, :, label_col].squeeze(-1)
        else:
            logger.critical(f'Feeling dizzy with too many dimensions: {input.ndim}!')

    loss = F.mse_loss(input, label, reduction=reduction)

    return loss


def softmax_bce(input, label, label_col=1, reduction='none',
            gamma=0.0, # for focal loss, 2-4 works well (may need to adjust it)
            alpha=1.0, auto_alpha=False, auto_alpha_mode='length', auto_alpha_pow=1.0,
            **kwargs):
    """ alpha is the multiplier for the loss of positive labels
        auto_alpha_mode can be length/npratio
        auto_alpha_pow can decrease from 1.0 to 0.0
        gamma is the exponent in focal loss, usually between 2 and 4
    """
    if kwargs: logger.error(f'kwargs is supposed to be empty: {kwargs}!!!')
    assert input.ndim == label.ndim + 1, \
            f"input.ndim:{input.ndim} - label.ndim:{label.ndim} != 1!"
    assert input.shape[-1] == 2,  \
            f"input.shape[-1]:{input.shape[-1]} != 2!"

    y0, y1 = mi.unstack(input, axis=-1)

    if label_col == 1: # which index (default: 1) to compare with positive label
        y_delta = y1 - y0
    else:
        y_delta = y0 - y1

    if auto_alpha: # the ratio between negative and positive labels
        if auto_alpha_mode in ['npratio', 'np', 'ratio']: # can be zero if all positive
            alpha = label.size / max([1.0, label.sum()]) - 1.0
        elif auto_alpha_mode in ['len', 'length']: # just length - 1
            alpha = label.shape[1] - 1
        else:
            logger.error(f'Unsupported auto_alpha_mode: {auto_alpha_mode}!!!')

        if alpha != 1.0 and auto_alpha_pow != 1.0:
            alpha = math.pow(alpha, auto_alpha_pow)

    if alpha == 1.0 and gamma == 0.0:
        # this is a reduced formula, please derive to check
        loss = mi.log(1.0 + mi.exp(y_delta)) - label * y_delta
    elif alpha != 1.0 and gamma == 0.0:
        # Option 1:
        # label_scale*label*log(p_label) + (1-label_scale)(1-label)log(1-p_label)
        # label_scale = alpha / (1 + alpha)
        # multiply 1000 as the value is too small
        # loss = 1000.0* 2.0* ((2.0 * alpha * label - alpha - label + 1.0) * \
        #         mi.log(1.0 + mi.exp(y_delta)) - alpha * label * y_delta)
        # Option 2:
        # alpha*label*log(p_label) + (1-label)log(1-p_label)
        loss = mi.log(1.0 + mi.exp(y_delta)) * (1.0 + (alpha - 1.0) * label) - \
                alpha * label * y_delta
    else: # focal loss
        # multiply 100
        guess = 1.0 - 1.0 / (mi.exp(y_delta) + 1.0)
        loss = -100.0 * 2 ** (gamma + 2.0) * \
            (alpha * mi.pow(1.0 - guess, gamma) * label * mi.log(guess) + \
            mi.pow(guess, gamma) * (1.0 - label) * mi.log(1.0 - guess))

    if reduction == 'mean':
        loss = loss.mean()

    return loss


class MetricFn(nn.Layer):
    """ A container for metric fn
    """
    def __init__(self, fn=mitas_utils.pfarm_metric, name='farp', **kwargs):
        super(MetricFn, self).__init__()

        self.name = name.lower()
        self.fn = fn
        self.kwargs = kwargs

        if self.name in ['farp', 'pfarm']:
            self.labels = ['pre', 'f1', 'acc', 'rec', 'mcc']
        elif self.name in ['rmse_angle', 'rmse']:
            self.labels = ['rmse', 'mape']
        else:
            logger.critical(f'Cannot recognize metric_fn name: {self.name}!')
            self.labels = []
        self.num = len(self.labels)

    def forward(self, input, label, **kwargs):
        self.kwargs.update(kwargs)
        # print(input.size, label.size, kwargs)
        return self.fn(input, label, **self.kwargs)


class SeqLossFn_Agg(nn.Layer):
    """ calculate the aggregated loss for input vs. label
    meant for loss_fn which cannot give point to point loss contributions
    """
    def __init__(self, fn=soft_fscore, name='softmax+fscore',
            loss_mask=False, loss_sqrt=False,
            **fn_kwargs):
        super(SeqLossFn_Agg, self).__init__()

        self.name = name.lower()
        self.loss_mask = loss_mask
        self.loss_sqrt = loss_sqrt
        self.times_called = 0
        self.fn = fn
        self.fn_kwargs = fn_kwargs

    def as_label(self, input):
        if self.name in ['softmax+fscore', 'softmax+fscore+bpp', 'softmax+fscore+bpp+l2']: # should add softmax to the name
            input = F.softmax(input, axis=-1)[...,self.fn_kwargs.get('label_col', 1)]
        elif self.name in ['masked_mse']:
            pass
        else:
            logger.critical(f'Cannot recognize loss_fn name: {self.name}!')

        return input

    def forward(self, input, label, seqs_len=None, **kwargs2fn):
        """ return the loss with the shape as input """

        self.times_called += 1
        batch_size = input.shape[0]
        data_len = input.shape[1]

        fn_kwargs = self.fn_kwargs.copy()
        fn_kwargs.update(kwargs2fn) # self.kwargs.copy()?

        if self.name in ['softmax+fscore', 'softmax+fscore+bpp', 'softmax+fscore+bpp+l2']:
            # the data is supposed to be before softmax (last dimension with shape of 2), and label may have the final dimension as 1
            # if not isinstance(label, mi.Tensor) or label.dtype.name != 'INT64':
                # label = mi.to_tensor(label, dtype='int32')
            input = self.as_label(input)
        elif self.name in ['masked_mse']:
            pass
        else:
            logger.critical(f'loss fn: {self.name} not supported yet!')

        if self.times_called % 10000 == 1 and logger.root.level >= logging.INFO:
            logger.info(f'  ======= Loss Fn (times called: {self.times_called}) =======')
            logger.info(f'         name: {self.name}')
            logger.info(f'     seqs_len: {seqs_len}') # str(seqs_len).replace('\n', '')
            logger.info(f'  input shape: {input.shape}')
            logger.info(f'  label shape: {label.shape}')
            logger.info(f'    loss_mask: {self.loss_mask}')
            logger.info(f'    loss_sqrt: {self.loss_sqrt}')
            logger.info(f'    fn kwargs: {fn_kwargs}')

        # no std for
        std_vs_seq = np.zeros((batch_size), dtype='float32')
        # # deal each instance/sequence separately
        if self.loss_mask and seqs_len is not None:# and any(seqs_len < data_len):

            # TODO: should decide whether to multiply mask or iterate through each instance
            #       depending on how large is the batch size (>4 to use mask?)

            loss_for_backprop = mi.zeros([1], dtype='float32')
            # loss_for_backprop = mi.to_tensor(0.0, dtype='float32', stop_gradient=False)
            loss_vs_seq = np.zeros((batch_size), dtype=np.float32)

            for i in range(batch_size):
                seq_len = int(seqs_len[i]) # must be an integer (not numpy.int)

                if input.ndim == 1: # process loss_mat as a whole
                    seq_loss = self.fn(input[i], label[i], **fn_kwargs)
                elif input.ndim == 2:
                    seq_loss = self.fn(input[i, :seq_len], label[i, :seq_len], **fn_kwargs)
                elif input.ndim == 3:
                    seq_loss = self.fn(input[i, :seq_len, :seq_len],
                                    label[i, :seq_len, :seq_len],
                                    **fn_kwargs)
                elif input.ndim == 4:
                    seq_loss = self.fn(input[i, :seq_len, :seq_len, :seq_len],
                                    label[i, :seq_len, :seq_len, :seq_len],
                                    **fn_kwargs)
                elif input.ndim == 5:
                    seq_loss = self.fn(input[i, :seq_len, :seq_len, :seq_len, :seq_len],
                                    label[i, :seq_len, :seq_len, :seq_len, :seq_len],
                                    **fn_kwargs)
                else:
                    logger.critical('too many dimensions for y_model, unsupported!')

                if self.loss_sqrt:
                    loss_for_backprop += mi.sqrt(seq_loss)
                    loss_vs_seq[i] = np.sqrt(seq_loss.numpy())
                    # std_vs_seq[i] = np.sqrt(seq_loss.numpy().std())
                else:
                    loss_for_backprop += seq_loss
                    loss_vs_seq[i] = seq_loss.numpy()
                    # std_vs_seq[i] = seq_loss.numpy().std()
        else:
            if 'batch' in fn_kwargs: fn_kwargs['batch'] = True
            loss_vs_seq = self.fn(input, label, **fn_kwargs)

            if self.loss_sqrt:
                loss_vs_seq = mi.sqrt(loss_vs_seq)

            loss_for_backprop = mi.sum(loss_vs_seq)
            loss_vs_seq = loss_vs_seq.numpy()

        return loss_for_backprop, loss_vs_seq, std_vs_seq


class SeqLossFn_P2P(nn.Layer):
    """ returns a scalar loss, loss_vs_seq: [N], std_vs_seq: [N].
    Designed for losses that can be defined for every element of
    the guess/label, such as MSE, CE.

    loss_vs_seq[i] is averaged over all elements of the ith seq
    """

    def __init__(self, fn=F.mse_loss, name='mse',
            loss_mask=False, loss_sqrt=False,
            **fn_kwargs):
        super(SeqLossFn_P2P, self).__init__()

        self.name = name.lower()
        self.loss_mask = loss_mask
        self.loss_sqrt = loss_sqrt
        self.times_called = 0
        self.fn = fn
        self.fn_kwargs = fn_kwargs

    def as_label(self, input):
        """ Convert input to the same form as label, as model outputs may need
        to go through sigmoid, softmax, etc.
        """
        if not isinstance(input, mi.Tensor):
            input = mi.to_tensor(input)

        if self.name in ['mse', 'bce']:
            pass

        elif self.name in ['ce', 'crossentropy']:
            logger.critical('Need test! Paddle help is the same as softmax+ce, very odd!!!')

        elif self.name in ['sigmoid+mse', 'sigmoid+bce', 'sigmoid+ce']:
            input = F.sigmoid(input)

        elif self.name in ['softmax+mse']:
            input = F.softmax(input, axis=-1)[...,self.fn_kwargs.get('label_col', 1)]

        elif self.name in ['softmax+bce']:
            input = F.softmax(input, axis=-1)[...,self.fn_kwargs.get('label_col', 1)]

        elif self.name in ['softmax+ce']:
            if self.fn_kwargs.get('soft_label', False):
                input = F.softmax(input)
            else:
                input = F.softmax(input)[...,self.fn_kwargs.get('label_col', 1)]
        else:
            logger.critical(f'Cannot recognize loss_fn name: {self.name}!')

        return input

    def forward(self, input, label, seqs_len=None, **kwargs2fn):
        """ Return the loss with the same shape as input """
        self.times_called += 1
        batch_size = input.shape[0]
        data_len = input.shape[1]

        fn_kwargs = self.fn_kwargs.copy()
        fn_kwargs.update(kwargs2fn)

        # deal with the specific requirements of the loss functions
        # if self.name in ['softmax+ce', 'crossentropy']:

        #     if self.fn_kwargs.get('soft_label', False): # soft label
        #         # should not need this in the future!!!
        #         if input.ndim > label.ndim:
        #             label = mitas_utils.hard2soft_label(label, ntype=input.shape[-1], mi=mi)
        #     else:  # hard label
        #         if not isinstance(label, mi.Tensor) or label.dtype.name != 'INT32':
        #             label = mi.to_tensor(label, dtype='int32')

        #         # if input.ndim > label.ndim:
        #             # label = mi.unsqueeze(label, axis=-1)

        # elif self.name in ['mse', 'bce', 'sigmoid+mse', 'sigmoid+bce', 'sigmoid+ce']:
        #     if input.ndim == label.ndim + 1 and input.shape[-1] == 1:
        #         input = input.squeeze(-1)

        # if not isinstance(y_truth, mi.Tensor) or y_truth.dtype.name != 'FP32':
        #     y_truth = mi.to_tensor(y_truth, dtype='float32')

        if self.times_called % 10000 == 1 and logger.root.level >= logging.INFO:
            logger.info(f'  ======= Loss Fn (times called: {self.times_called}) =======')
            logger.info(f'         name: {self.name}')
            logger.info(f'     seqs_len: {seqs_len}') # str(seqs_len).replace('\n', '')
            logger.info(f'  input shape: {input.shape}')
            logger.info(f'  label shape: {label.shape}')
            logger.info(f'    loss_mask: {self.loss_mask}')
            logger.info(f'    loss_sqrt: {self.loss_sqrt}')
            logger.info(f'    fn kwargs: {fn_kwargs}')

        # calculate all anyway, maybe more efficient for GPU
        loss_mat = self.fn(input, label, **fn_kwargs)

        # deal each instance/sequence separately
        if self.loss_mask and seqs_len is not None and mi.any(seqs_len < data_len):

            # TODO: should decide whether to multiply mask or iterate through each instance
            #       depending on how large is the batch size (>4 to use mask?)

            loss_for_backprop = mi.zeros([1], dtype='float32')
            # loss_for_backprop = mi.to_tensor(0.0, dtype='float32', stop_gradient=False)
            # loss_vs_seq = mi.zeros((batch_size,), dtype='float32')
            loss_vs_seq = np.zeros((batch_size), dtype=np.float32)
            std_vs_seq = np.zeros((batch_size), dtype=np.float32)

            for i in range(batch_size):
                seq_len = int(seqs_len[i]) # must be an integer (not numpy.int)
                if loss_mat.ndim == 1:
                    seq_loss = loss_mat[i] # self.loss_fn(input[i], label[i], **kwargs)
                elif loss_mat.ndim == 2:
                    seq_loss = loss_mat[i, :seq_len]
                elif loss_mat.ndim == 3:
                    seq_loss = loss_mat[i, :seq_len, :seq_len]
                elif loss_mat.ndim == 4:
                    seq_loss = loss_mat[i, :seq_len, :seq_len, :seq_len]
                elif loss_mat.ndim == 5:
                    seq_loss = loss_mat[i, :seq_len, :seq_len, :seq_len, :seq_len]
                else:
                    logger.critical('too many dimensions for y_model, unsupported!')

                if self.loss_sqrt:
                    loss_for_backprop += mi.sqrt(mi.mean(seq_loss))
                    loss_vs_seq[i] = np.sqrt(seq_loss.numpy().mean())
                    std_vs_seq[i] = np.sqrt(seq_loss.numpy().std())
                else:
                    loss_for_backprop += mi.mean(seq_loss)
                    loss_vs_seq[i] = seq_loss.numpy().mean()
                    std_vs_seq[i] = seq_loss.numpy().std()
        # process loss_mat as a whole
        else:
            if loss_mat.ndim == 1:
                # may need to squeeze the loss_mat
                loss_vs_seq = mi.squeeze(loss_mat, -1)
                # the std of the errors for each instance
                std_vs_seq = np.zeros_like(loss_mat, dtype=np.float32)
            else:
                # the axes for each instance, from the 2nd to the last
                inst_axes = tuple(range(1, loss_mat.ndim))
                loss_vs_seq = mi.mean(loss_mat, axis=inst_axes)
                std_vs_seq = loss_mat.numpy().std(axis=inst_axes)

            if self.loss_sqrt:
                loss_vs_seq = mi.sqrt(loss_vs_seq)
                std_vs_seq = np.sqrt(std_vs_seq)

            loss_for_backprop = mi.sum(loss_vs_seq)
            loss_vs_seq = loss_vs_seq.numpy()

        # calculate the next loss_fn as needed
        # if self.loss_fn_next is not None:
        #     loss_for_backprop2, loss_vs_seq2, std_vs_seq2 = self.loss_fn_next(input, label,
        #                 seqs_len=seqs_len, loss_padding=loss_padding, loss_sqrt=loss_sqrt, **self.loss_fn_next_kwargs)

        #     loss_for_backprop += loss_for_backprop2
        #     loss_vs_seq += loss_vs_seq2
        #     std_vs_seq += std_vs_seq2

        return loss_for_backprop, loss_vs_seq, std_vs_seq


def get_metric_fn(args):
    """  """
    misc.logger_setlevel(logger, args.verbose) # for scout_args, etc.
    if args.metric_fn is None: return None

    if isinstance(args.metric_fn, str):
        args.metric_fn = [args.metric_fn]
    args.metric_fn = [_s.lower() for _s in args.metric_fn]

    metric_fns = []
    logger.info(f'======= Metric Function =======')
    logger.info(f'            type: {args.metric_fn}')

    for metric_fn in args.metric_fn:
        if metric_fn in ['farp', 'pfarm']:
            metric_fns.append(MetricFn(mitas_utils.pfarm_metric, name='pfarm',
                    beta=args.metric_beta, threshold=args.metric_threshold,
                    batch=args.batch_size > 1))
            logger.info(f'            beta: {args.metric_beta}')
            logger.info(f'       threshold: {args.metric_threshold}')

        elif metric_fn in ['rmse_angle']:
            metric_fns.append(MetricFn(masked_rmse_metric, name='rmse_angle'))
            # logger.info(f'            beta: {args.metric_beta}')
            # logger.info(f'       threshold: {args.metric_threshold}')

        elif metric_fn in ['rmse']:
            metric_fns.append(MetricFn(rmse_metric, name='rmse'))
            pass
        else:
            logger.critical(f'Unrecognized metric_fn: {metric_fn} !!!')


    args.metric_labels = misc.unpack_list_tuple([_fn.labels for _fn in metric_fns])
    logger.info(f'             num: {len(metric_fns)}')
    logger.info(f'          labels: {args.metric_labels}')

    if len(metric_fns):
        return metric_fns
    else:
        logger.critical(f'Unrecognized metric name: {args.metric_fn} !!!')
        return None


def get_loss_fn(args):
    """  """
    misc.logger_setlevel(logger, args.verbose) # for scout_args, etc.

    if isinstance(args.loss_fn, str): args.loss_fn = [args.loss_fn]
    loss_name_map = {
        'ce': 'crossentropy',
        'f_score': 'fscore',
        'binary_cross_entropy': 'bce',
        'sigmoid_crossentropy': 'sigmoid+ce',
        'softmax_crossentropy': 'softmax+ce',
        'softmax_bce': 'softmax+bce',
        'sotmax_fscore': 'softmax+fscore',
        'sotmax_f_score': 'softmax+fscore',
        'sotmax_fscore_bpp': 'softmax+fscore+bpp',
        'sotmax_f_score_bpp': 'softmax+fscore+bpp',
        'sotmax_fscore_bpp_l2': 'softmax+fscore+bpp+l2',
        'sotmax_f_score_bpp_l2': 'softmax+fscore+bpp+l2',
        }
    args.loss_fn = [loss_name_map.get(_s.lower().replace('+', '_').replace('-', '_'),
                        _s.lower()) for _s in args.loss_fn]
    logger.info(f'======= Loss Function =======')

    # temporary fix for the loss_auto_alpha_pow, loss_auto_beta_pow, and loss_gamma
    fn_fix_array = lambda x, length, x0: \
        [x] + [x0]*(length-1) if not hasattr(x, '__len__') else \
        x[:length] + [x0]*(length-len(x))

    args.loss_auto_alpha_pow = fn_fix_array(args.loss_auto_alpha_pow, 2, 0.0)
    args.loss_auto_beta_pow = fn_fix_array(args.loss_auto_beta_pow, 2, 0.0)
    args.loss_alpha = fn_fix_array(args.loss_alpha, 2, 1.0)
    args.loss_beta = fn_fix_array(args.loss_beta, 2, 1.0)
    args.loss_gamma = fn_fix_array(args.loss_gamma, 2, 0.0)

    loss_fn = []
    for fn_name in args.loss_fn:
        # kwargs is handled in the SeqLossFn class
        kwargs = dict(
            name=fn_name,
            loss_mask=args.loss_mask,
            loss_sqrt=args.loss_sqrt,
            # loss_masked_l2decay=args.loss_masked_l2decay,
            )
        for key, val in kwargs.items(): logger.info(f'{key:>18s}: {val}')

        # fn_kwargs is handled in the loss_fn
        if fn_name == 'mse':
            fn_kwargs = dict(reduction='none')
            loss_fn.append(SeqLossFn_P2P(F.mse_loss, **kwargs, **fn_kwargs))

        elif fn_name == 'bce':
            fn_kwargs = dict(reduction='none', weight=None)
            loss_fn.append(SeqLossFn_P2P(F.binary_cross_entropy, **kwargs, **fn_kwargs))

        elif fn_name == 'crossentropy':
            fn_kwargs = dict(soft_label=args.label_tone.lower() == 'soft' or args.label_hard2soft)
            loss_fn.append(SeqLossFn_P2P(F.cross_entropy, **kwargs, **fn_kwargs))

        elif fn_name == 'sigmoid+bce':
            fn_kwargs = dict(reduction='none', weight=None, pos_weight=None)
            if args.loss_auto_alpha:
                logger.error(f'loss_fn: {fn_name} does not support auto_alpha!!!')
            else:
                if args.loss_alpha is not None and args.loss_alpha[0] != 1.0:
                    fn_kwargs['pos_weight'] = mi.to_tensor(args.loss_alpha[0], dtype='float32')
            loss_fn.append(SeqLossFn_P2P(F.binary_cross_entropy_with_logits, **kwargs, **fn_kwargs))

        elif fn_name == 'sigmoid+ce':
            fn_kwargs = dict(soft_label=args.label_tone.lower() == 'soft' or args.label_hard2soft)
            loss_fn.append(SeqLossFn_P2P(F.cross_entropy, **kwargs, **fn_kwargs))

        elif fn_name == 'sigmoid+mse':
            fn_kwargs = dict(reduction='none')
            loss_fn.append(SeqLossFn_P2P(sigmoid_mse, **kwargs, **fn_kwargs))

        elif fn_name == 'softmax+bce':
            fn_kwargs = dict(label_col=1, reduction='none', gamma=args.loss_gamma[0],
                auto_alpha=args.loss_auto_alpha)
            if args.loss_auto_alpha:
                fn_kwargs.update(
                    auto_alpha_pow=args.loss_auto_alpha_pow[0],
                    auto_alpha_mode=args.loss_auto_alpha_mode,
                    )
            else:
                fn_kwargs.update(alpha=args.loss_alpha[0])
            loss_fn.append(SeqLossFn_P2P(softmax_bce, **kwargs, **fn_kwargs))

        elif fn_name == 'softmax+ce':
            fn_kwargs = dict(soft_label=args.label_tone.lower() == 'soft' or args.label_hard2soft)
            loss_fn.append(SeqLossFn_P2P(F.softmax_with_cross_entropy, **kwargs, **fn_kwargs))

        elif fn_name == 'softmax+mse':
            fn_kwargs = dict(label_col=1, reduction='none')
            loss_fn.append(SeqLossFn_P2P(softmax_mse, **kwargs, **fn_kwargs))

        elif fn_name == 'masked_mse':
            fn_kwargs = dict()
            loss_fn.append(SeqLossFn_Agg(masked_angle_mse, **kwargs, **fn_kwargs))

        elif fn_name == 'softmax+fscore':
            fn_kwargs = dict(
                batch=False, symmetric=args.loss_symmetric,
                auto_beta=args.loss_auto_beta)
            if args.loss_auto_beta:
                fn_kwargs.update(
                    auto_beta_pow=args.loss_auto_beta_pow[0],
                    auto_beta_mode=args.loss_auto_beta_mode,
                    )
            else:
                fn_kwargs.update(beta=args.loss_beta[0])
            loss_fn.append(SeqLossFn_Agg(soft_fscore, **kwargs, **fn_kwargs))

        elif fn_name == 'softmax+fscore+bpp':
            fn_kwargs = dict(
                bpp=True, bpp_scale=args.loss_bpp_scale,
                batch=False, symmetric=args.loss_symmetric,
                auto_beta=args.loss_auto_beta)
            if args.loss_auto_beta:
                fn_kwargs.update(
                    auto_beta_pow=args.loss_auto_beta_pow[0],
                    auto_beta_mode=args.loss_auto_beta_mode,
                    )
            else:
                fn_kwargs.update(beta=args.loss_beta[0])
            loss_fn.append(SeqLossFn_Agg(soft_fscore, **kwargs, **fn_kwargs))

        elif fn_name == 'softmax+fscore+bpp+l2':
            fn_kwargs = dict(
                l2_mask=MiNets.MatDiagonalMaskOut(data_fmt='LL', offset=2, reverse=True),
                l2_scale=args.loss_l2_scale,
                bpp=True, bpp_scale=args.loss_bpp_scale,
                batch=False, symmetric=args.loss_symmetric,
                auto_beta=args.loss_auto_beta)
            if args.loss_auto_beta:
                fn_kwargs.update(
                    auto_beta_pow=args.loss_auto_beta_pow[0],
                    auto_beta_mode=args.loss_auto_beta_mode,
                    )
            else:
                fn_kwargs.update(beta=args.loss_beta[0])
            loss_fn.append(SeqLossFn_Agg(soft_fscore, **kwargs, **fn_kwargs))

        else:
            logger.critical(f'Unsupported loss_fn: {fn_name}!!!')
            loss_fn.append(None)
            fn_kwargs = dict()

        for key, val in fn_kwargs.items(): logger.info(f'{key:>18s}: {val}')

    if len(loss_fn) == 0:
        logger.critical(f'not supported loss functions found in: {args.loss_fn}!')

    args.loss_fn_scale = fn_fix_array(args.loss_fn_scale, len(loss_fn), 1.0)

    return loss_fn


def get_callback_fn(args):
    pass


def post_process_guess(guess, input=None, method=None, threshold=0.99, min_delta_ij=None, seqs_len=None, mi=mi):
    """ guess and input (seq_onehot) have batch_size as the first dim
        threshold is for ufold post_processs
    """
    if type(method) not in (tuple, list):
        method = [method]

    for _method in method:
        if _method in ['cn', 'canon', 'canonical']:
            base_a = input[:, :, 0:1]
            base_u = input[:, :, 1:2]
            base_c = input[:, :, 2:3]
            base_g = input[:, :, 3:4]
            batch = base_a.shape[0]
            length = base_a.shape[1]
            bpmat = mi.matmul(base_a, base_u.reshape((batch, 1, length))) + \
                    mi.matmul(base_c, base_g.reshape((batch, 1, length))) + \
                    mi.matmul(base_u, base_g.reshape((batch, 1, length)))
            bpmat += bpmat.transpose([0,2,1])
            guess *= (bpmat == 1).astype(guess.dtype)
        elif _method in ['nc', 'nocanonical', 'noncanonical']:
            base_a = input[:, :, 0:1]
            base_u = input[:, :, 1:2]
            base_c = input[:, :, 2:3]
            base_g = input[:, :, 3:4]
            batch = base_a.shape[0]
            length = base_a.shape[1]
            bpmat = mi.matmul(base_a, base_c.reshape((batch, 1, length))) + \
                    mi.matmul(base_a, base_g.reshape((batch, 1, length))) + \
                    mi.matmul(base_a, base_u.reshape((batch, 1, length))) + \
                    mi.matmul(base_c, base_g.reshape((batch, 1, length))) + \
                    mi.matmul(base_c, base_u.reshape((batch, 1, length))) + \
                    mi.matmul(base_g, base_u.reshape((batch, 1, length)))
            bpmat += bpmat.transpose([0,2,1]) + \
                    mi.matmul(base_a, base_a.reshape((batch, 1, length))) + \
                    mi.matmul(base_c, base_c.reshape((batch, 1, length))) + \
                    mi.matmul(base_g, base_g.reshape((batch, 1, length))) + \
                    mi.matmul(base_u, base_u.reshape((batch, 1, length)))
            guess *= (bpmat == 1).astype(guess.dtype)
        elif _method in ['gridsearch', 'grid_search']:
            pass
        elif _method.startswith('ufold'):
            import ufold_postpro as ufold

            guess_logits = -mi.log(1.0 / guess - 1.0) # pre-sigmoid logits
            constraint_fn = ufold.constraint_matrix_batch

            if _method in ['ufold_nc', 'ufold_noncanon', 'ufold_noncanonical']:
                constraint_fn = ufold.constraint_matrix_batch_addnc
            elif _method in ['ufold_cn', 'ufold_canon', 'ufold_canonical']:
                constraint_fn = ufold.constraint_matrix_batch
                pass
            else:
                logger.warning(f'Unrecognized post_process method: {_method}!')

            guess = ufold.post_process_paddle(
                guess_logits.numpy(), input.numpy(), threshold=threshold, constraint_fn=constraint_fn,
                lr_min=0.01, lr_max=0.1, num_itr=100, rho=1.6, with_l1=True)
                #seq_ori, 0.01, 0.1, 50, 1, True))
            guess = mi.to_tensor((guess > 0.5).astype('float32'), dtype='float32')
            # the following is used by UFold
            # the threshold is
            # u_no_train = postprocess(pred_contacts,
            #     seq_ori, 0.01, 0.1, 100, 1.6, True, 1.5) sigmoid(1.5) = 0.82
            #     #seq_ori, 0.01, 0.1, 50, 1, True)
            # map_no_train = (u_no_train > 0.5).float()
        else:
            logger.warning(f'Unrecognized post_process method: {_method}!')

    if min_delta_ij:
        x_len = guess.shape[1]
        y_len = guess.shape[2]

        # a matrix of [x_len, y_len] with each element storing the row idx
        row_idx_mat = mi.broadcast_to(
                mi.linspace(1, x_len, x_len, dtype="int32").unsqueeze(-1),
                (x_len, y_len))

        if x_len == y_len:
            guess *= (mi.abs(row_idx_mat - row_idx_mat.transpose([1,0])) >= min_delta_ij).astype(guess.dtype)
        else:
            col_idx_mat = mi.broadcast_to(
                mi.linspace(1, y_len, y_len, dtype='int32').unsqueeze(0),
                (x_len, y_len))

            guess *= (mi.abs(row_idx_mat - col_idx_mat) >= min_delta_ij).astype(guess.dtype)

    return guess


def post_analyze_output(guess, label=None, method=None, objective='f1'):
    """ not a good idea to do post analysis here """
    pass


def compute_loss(loss_fn, input, label, seqs_len=None, shuffle=False, batch_size=23, **kwargs):
    """ Both input and label can be list/array/tensor
    But the first dimension must be the batch_size
    """
    if type(loss_fn) not in (list, tuple): loss_fn = [loss_fn]

    num_data = len(input)
    if seqs_len is None:
        midat = list(zip(input, label))
    else:
        midat = list(zip(input, seqs_len, label))

    if not isinstance(midat, MyDataset):
        midat = get_dataset(midat)

    if isinstance(midat, MyDataset):
        miloader = get_dataloader(midat, batch_size=min([batch_size, len(midat)]),
                    shuffle=shuffle, drop_last=False)
    else:
        miloader = midat

    loss_vs_seq = np.zeros((num_data), dtype=np.float32)
    std_vs_seq = np.zeros((num_data), dtype=np.float32)

    for ibatch, data in enumerate(miloader):
        num_seqs = len(data[0])
        istart = ibatch * batch_size
        iend = istart + num_seqs
        seqlen_batch = None if seqs_len is None else data[1].numpy()

        for one_loss_fn in loss_fn:
            _, _loss_vs_seq, _std_vs_seq = one_loss_fn(data[0], data[-1],
                        seqs_len=seqlen_batch, **kwargs)
            loss_vs_seq[istart:iend] = loss_vs_seq[istart:iend] + _loss_vs_seq
            std_vs_seq[istart:iend] = std_vs_seq[istart:iend] + _std_vs_seq

    return loss_vs_seq, std_vs_seq


def train(model, miset, **kwargs):
    """ miset can be midat, miset, or miloader
        Bad practices:
        1) args procesing is odd (kwargs > model.args > args )
        2) new fields are added to the model structure
    """
    # default settings
    args = misc.Struct(dict(
                trainloss_patience = 7, trainloss_rdiff = 1e-7,
                validloss_patience = 7, validloss_rdiff = 1e-7,
                evaluate_callback = None,
                num_evals_per_epoch = 10,
                lr_scheduler = 'none',
                num_recaps_per_epoch = 30,
                num_epochs = 2,
                batch_size = 2,
                loss_padding = False,
                shuffle = True,
                drop_last = True,
                save_dir = None,
                save_level = 1,
                verbose = 1,
                ))
    misc.logger_setlevel(logger, args.verbose)
    args.update(vars(model.args)) # model.args overwrites default args
    args.update(kwargs) # kwargs overwrites all
    if isinstance(args.save_dir, str): args.save_dir = Path(args.save_dir)
    if args.save_dir: args.save_dir.mkdir(parents=True, exist_ok=True)
    model.args.update(vars(args))
    # args can still change by user tweaks such as learning rate, optim_step_stride, etc
    args = model.args

    # do not need this yet, save it for later
    # num_workers = mi.distributed.ParallelEnv().nranks
    # work_site = mi.CUDAPlace(mi.distributed.ParallelEnv().dev_id) if num_workers > 1 else mi.CUDAPlace(0)
    # exe = mi.Executor(work_site)

    if args.train_size is not None and args.train_size > 0:
        if args.train_size < len(miset):
            logger.info(f'Sampling {args.train_size} data out of a total of {len(miset)}...')
            miset = mitas_utils.random_sample(miset, size=args.train_size, replace=False)
        elif args.train_size == len(miset):
            logger.warning(f'Specified train size: {args.train_size} == data length: {len(miset)}.')
        else:
            logger.warning(f'Specified train size: {args.train_size} > data length: {len(miset)}!')

    if isinstance(miset, MyDataset):
        if args.batch_size > len(miset):
            logger.warning(f'batch_size: {args.batch_size} > data_size: {len(miset)}!!!')
            # args.batch_size = len(miset)
        if args.jit_loader and not miset.data_jit_full:
            logger.info('Setting num_workers=0 for the first epoch with jit_loader=True')
            # set num_worker=0, so that miset is not copied to worker processes
            miloader = get_dataloader(miset, batch_size=min([args.batch_size, len(miset)]),
                    shuffle=args.shuffle, drop_last=False, num_workers=0)
        else:
            miloader = get_dataloader(miset, batch_size=min([args.batch_size, len(miset)]),
                    shuffle=args.shuffle, drop_last=args.drop_last)
    else: # can be any iterable
        miloader = miset

    model.num_samples = len(miset)
    model.num_batches = len(miloader)
    num_inputs = 1 if isinstance(args.input_genre, str) else len(args.input_genre)
    num_labels = 1 if isinstance(args.label_genre, str) else len(args.label_genre)

    # chkpt_interval = max([1, model.num_batches // args.num_evals_per_epoch])
    # recap_interval = model.num_batches // args.num_recaps_per_epoch + 1
    # num_recaps = int(np.ceil(model.num_batches / recap_interval))
    logger.info(f'======= Training =======')
    logger.info(f'        data size: {len(miset)}')
    logger.info(f'      input_genre: {args.input_genre}')
    logger.info(f'      label_genre: {args.label_genre}')
    logger.info(f'       jit loader: {args.jit_loader}')
    logger.info(f'       batch size: {args.batch_size}')
    logger.info(f'          shuffle: {args.shuffle}')
    logger.info(f'        drop_last: {args.drop_last}')
    logger.info(f'     # of batches: {model.num_batches}')
    logger.info(f'# of recaps/epoch: {args.num_recaps_per_epoch}')
    logger.info(f' # of evals/epoch: {args.num_evals_per_epoch}')
    logger.info(f'  max # of epochs: {args.num_epochs}')
    logger.info(f'     label smooth: {args.label_smooth}')
    logger.info(f'        loss mask: {args.loss_mask}')
    logger.info(f'        loss sqrt: {args.loss_sqrt}')
    # logger.info(f'    evaluation interval: {chkpt_interval}')
    # logger.info(f'         recap interval: {recap_interval}')

    if args.visual_dl:
        from visualdl import LogWriter
        visualdl_dir = Path.home() / 'tmp/visual_dl' / datetime.now().strftime('%m-%d-%H-%M-%S')
        visualdl_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f'         visual dir: {visualdl_dir.as_posix()}')
        visualdl_writer = LogWriter(logdir=visualdl_dir.as_posix())

    # model.train_loss is a list of DataFrames (concatenated at the end) keeping all loss data
    if not hasattr(model, 'train_loss'):
        model.train_loss = []
    elif isinstance(model.train_loss, pd.DataFrame):
        model.train_loss = [model.train_loss]
    elif isinstance(model.train_loss, list):
        logger.info('model.train_loss is already a list, nothing to be done!')
    else:
        logger.critical(f'Cannot deal with model.train_loss: {type(model.train_loss)}, it will be cleared to []!!!')
        model.train_loss = []

    # model.eval_trail is a structure storing results from evaluate_in_train() calls
    if not hasattr(model, 'eval_trail'):
        model.eval_trail = misc.Struct(valid_loss=[]) # consider to include this in the model structure?
    elif isinstance(model.eval_trail, misc.Struct):
        logger.info(f'Continuing to use existing model.eval_trail with valid_loss of {model.eval_trail.valid_loss[0].shape} ...')
    else:
        logger.critical(f'Cannot deal with model.eval_trail, it will be reinitialized!!!')
        model.eval_trail = misc.Struct(valid_loss=[])

    # temporary variables for journaling, not saved to files
    train_loss_per_epoch = pd.DataFrame() # for recap and checking whether loss reached plateau
    valid_loss_per_epoch = pd.DataFrame() # for recap ... (average all eval calls in one epoch)

    ####### Tweak Files #######
    sv_file = args.save_dir / 'save' if args.save_dir else None
    br_file = args.save_dir / 'stop' if args.save_dir else None
    bs_file = args.save_dir / 'batch_size' if args.save_dir else None
    lr_file = args.save_dir / 'learning_rate' if args.save_dir else None
    wd_file = args.save_dir / 'weight_decay' if args.save_dir else None
    do_file = args.save_dir / 'dropout' if args.save_dir else None
    ss_file = args.save_dir / 'step_stride' if args.save_dir else None
    ls_file = args.save_dir / 'loss_twargs.json' if args.save_dir else None

    # model.epoch/batch/recap are the global counts across all training stages of the same model
    if not hasattr(model, 'stage'):
        model.stage = int(model.train_loss[-1].iloc[-1]['stage']) if len(model.train_loss) and 'stage' in model.train_loss[-1] else -1
    if not hasattr(model, 'epoch'):
        model.epoch = int(model.train_loss[-1].iloc[-1]['epoch']) if len(model.train_loss) else -1
    if not hasattr(model, 'batch'):
        model.batch = int(model.train_loss[-1].iloc[-1]['batch']) if len(model.train_loss) else -1
    if not hasattr(model, 'recap'):
        model.recap = int(model.train_loss[-1].iloc[-1]['recap']) if len(model.train_loss) else -1

    model.stage += 1
    model.net.train()
    model.optim.clear_grad()
    for iepoch in range(args.num_epochs):
        model.epoch += 1

        if args.save_dir:
            logger.info(f'Tweak via the following files in {args.save_dir}:')
            logger.info(f'  <{sv_file.name}><{br_file.name}><{bs_file.name}:{args.batch_size}>' +
                f'<{lr_file.name}:{args.learning_rate:.3g}><{ss_file.name}:{args.optim_step_stride}>' +
                f'<{wd_file.name}:{args.weight_decay:.2g}><{ls_file.name}>')

        ####### Stop Train #######
        if br_file and br_file.exists():
            logger.info(f'Stop training as directed by file: {br_file.as_posix()}')
            br_file.rename(br_file.with_suffix(f'.epo{model.epoch:03d}'))
            break

        ####### Tweak Data #######
        # One can change batch_size and/or dataloader
        new_batch_size = None
        # 1) read from file "batch_size" in the save_dir
        if bs_file and bs_file.exists():
            with bs_file.open('r') as iofile:
                try:
                    new_batch_size = int(iofile.readline().strip())
                    logger.info(f'Use batch size={new_batch_size} from: {bs_file}')
                except:
                    logger.error(f'Error reading batch size file: {bs_file}!!!')
            # use unlink() to remove
            bs_file.rename(bs_file.with_suffix(f'.epo{model.epoch:03d}'))

        # 2) change to default dataloader if data_jit is full after the first epoch
        if iepoch == 1 and args.jit_loader and miset.data_jit_full:
            logger.info(f'Reverting to regular dataloader after the first epoch ...')
            new_batch_size = args.batch_size

        if new_batch_size is not None:
            model.batch_size = args.batch_size = new_batch_size
            logger.info(f'NEW batch size: {args.batch_size} or new dataloader!!!')
            miloader = get_dataloader(miset, batch_size=args.batch_size, shuffle=args.shuffle,
                    drop_last=args.drop_last)
            model.num_batches = len(miloader)

        ####### Tweak Optimizer #######
        weight_decay = None
        if wd_file and wd_file.exists(): # NOT SURE WHETHER IT ACTUALLY DOES ANYTHING!!! #
            if isinstance(model.optim, mi.optimizer.AdamW):
                with wd_file.open('r') as iofile:
                    try:
                        weight_decay = float(iofile.readline().strip())
                        model.optim._coef = weight_decay
                        logger.info(f'Use weight decay={weight_decay} from: {wd_file}')
                        args.weight_decay = weight_decay
                    except:
                        logger.error(f'Error reading weight_decay file: {wd_file}!!!')
                # use unlink() to remove
                wd_file.rename(wd_file.with_suffix(f'.epo{model.epoch:03d}'))
            else:
                logger.critical(f'Tweaking weight decay is only supported for AdamW!!!')

        # Three ways to change learning rate
        new_lr = None
        # 1) learning rate warmup and/or cooldn
        # Both could apply if args.num_epochs < lr_warmup_steps + lr_cooldn_steps!!!
        new_lr = new_lr if iepoch >= args.lr_warmup_steps else \
                args.learning_rate * (iepoch + 1) / args.lr_warmup_steps

        new_lr = new_lr if (iepoch + args.lr_cooldn_steps) < args.num_epochs else \
                args.learning_rate * (args.num_epochs - iepoch) / args.lr_cooldn_steps

        # 2) from "learning_rate" file in args.save_dir, assigned to ARGS.LEARNING_RATE!!!
        if lr_file and lr_file.exists():
            with lr_file.open('r') as iofile:
                try:
                    new_lr = float(iofile.readline().strip())
                    args.learning_rate = new_lr
                    logger.info(f'Use learning rate={new_lr} from: {lr_file}')
                except:
                    logger.error(f'Error reading learning_rate file: {lr_file}!!!')
            # use unlink() to remove
            lr_file.rename(lr_file.with_suffix(f'.epo{model.epoch:03d}'))

        # 3) mi.optimizer.lr.LRScheduler():
        if new_lr is None:
            if iepoch > args.lr_warmup_steps and not isinstance(model.lr_scheduler, float):
                old_lr = model.lr_scheduler.last_lr
                model.lr_scheduler.step(train_loss_per_epoch.loss.iat[-1])

                # change step stride HERE!!!
                new_lr = model.lr_scheduler.last_lr
                if new_lr < old_lr:
                    args.optim_step_stride = min([args.optim_max_stride, int(args.optim_step_stride * math.pow(old_lr / new_lr, 0.25))])
                    logger.info(f'Auto-set optim step stride={args.optim_step_stride} due to decreased learning rate (old: {old_lr}, new: {new_lr})')
            else:
                # get value to save to loss_this_epoch
                new_lr = model.optim.get_lr() if isinstance(model.lr_scheduler, float) else model.lr_scheduler.last_lr
        else:
            if isinstance(model.lr_scheduler, float):
                old_lr = model.optim.get_lr()
                model.optim.set_lr(new_lr)
            else:
                model.lr_scheduler.last_lr = new_lr


        # Two ways to change the optim_step_stride
        # 1) if the learning_rate has been automatically reduced by lr_scheduler

        # SEE CODE ABOVE

        # 2) read from file "step_stride" in the save_dir
        if ss_file and ss_file.exists():
            with ss_file.open('r') as iofile:
                try:
                    args.optim_step_stride = int(iofile.readline().strip())
                    logger.info(f'Use optim step stride={args.optim_step_stride} from: {ss_file}')
                except:
                    logger.error(f'Error reading step stride file: {ss_file}!!!')
            # use unlink() to remove
            ss_file.rename(ss_file.with_suffix(f'.epo{model.epoch:03d}'))

        ####### Tweak Loss Function #######
        # apply loss cooldown steps
        if args.loss_cooldn_steps > 0 and iepoch > args.lr_warmup_steps:

            # goes down to zero for alpha_pow, beta_pow and gamma
            _cooldn_factor = max([0, args.loss_cooldn_steps + args.lr_warmup_steps - iepoch]) /\
                            args.loss_cooldn_steps
            logger.info(f'Applying loss cooling down factor: {misc.str_color(_cooldn_factor)}...')

            cooldn_fn = lambda x: x[1] + (x[0] - x[1]) * _cooldn_factor

            for _iloss in range(len(model.loss_fn)):
                if 'auto_alpha_pow' in model.loss_fn[_iloss].fn_kwargs:
                    model.loss_fn[_iloss].fn_kwargs['auto_alpha_pow'] = cooldn_fn(args.loss_auto_alpha_pow)

                if 'auto_beta_pow' in model.loss_fn[_iloss].fn_kwargs:
                    model.loss_fn[_iloss].fn_kwargs['auto_beta_pow'] = cooldn_fn(args.loss_auto_beta_pow)

                if 'alpha' in model.loss_fn[_iloss].fn_kwargs:
                    model.loss_fn[_iloss].fn_kwargs['alpha'] = cooldn_fn(args.loss_alpha)

                if 'beta' in model.loss_fn[_iloss].fn_kwargs:
                    model.loss_fn[_iloss].fn_kwargs['beta'] = cooldn_fn(args.loss_beta)

                if 'gamma' in model.loss_fn[_iloss].fn_kwargs:
                    model.loss_fn[_iloss].fn_kwargs['gamma'] = cooldn_fn(args.loss_gamma)


        if ls_file and ls_file.exists():
            try:
                args.loss_twargs = gwio.json2dict(ls_file)
                logger.info(f'Use loss_twargs: {args.loss_twargs} from: {ls_file}')
            except:
                logger.error(f'Error reading loss_twargs file: {ls_file}!!!')
            # use unlink() to remove
            ls_file.rename(ls_file.with_suffix(f'.epo{model.epoch:03d}'))


        ####### Tweak Net #######
        if do_file and do_file.exists():
            with do_file.open('r') as iofile:
                try:
                    args.dropout = float(iofile.readline().strip())
                    logger.info(f'Use dropout= {args.dropout} from: {do_file} (not implemented!!!)')
                    # how to do this? Need to search through all sublayers!!!
                    # Dropout.p = args.dropout?
                except:
                    logger.error(f'Error reading dropout file: {do_file}!!!')
                # use unlink() to remove
            do_file.rename(do_file.with_suffix(f'.epo{model.epoch:03d}'))

        ####### Finally, time to run #######

        # use numpy array for storing results, presumably more efficient?
        loss_for_backprop, loss_backprop_count = None, 0
        loss_this_epoch = np.zeros((model.num_samples, 11), dtype=np.float32)
        loss_this_epoch[:,7] = new_lr
        loss_this_epoch[:,8] = args.optim_step_stride
        loss_this_epoch[:,9] = args.batch_size
        loss_this_epoch[:,10] = model.stage

        if model.metric_fn is not None:
            metric_this_epoch = np.empty((model.num_samples, len(args.metric_labels)), dtype=np.float32)

        batches_since_recap = 0 # how many batches since the last recap
        istart, iend = 0, 0 # the start and end idx of the current batch wrt loss_one_epoch
        for ibatch, batch_data in enumerate(miloader):
            model.batch += 1 # model.epoch * model.num_batches + model.batch

            samples_idx = np.array(batch_data[0][:,0], dtype=np.int32)
            samples_len = batch_data[0][:,1]
            if samples_len.dtype is not mi.int32: # if not str(seqs_len.dtype).endswith('int32'):
                samples_len = samples_len.astype(mi.int32) # works for both numpy and paddle

            x_inputs = batch_data[1:num_inputs+1] # inputs are from the 2nd item
            y_labels = batch_data[-num_labels:] # labels are always at the ends

            # remove padding if possible
            # if args.batch_size == 1 and not args.loss_padding and x.ndim > 1 and \
            #         x.shape[1] > samples_len[0]:
            #     x = mitas_utils.cut_padding(x, samples_len[0])
            #     y_truth = mitas_utils.cut_padding(y_truth, samples_len[0])

            y_guess = model.net(*x_inputs, samples_len)
            if type(y_guess) not in (tuple, list):
                y_guess = [y_guess]

            # it seems to be better to calculate some labels here
            if 'f1' in args.label_genre:
                idx_f1 = args.label_genre.index('f1')
                with mi.no_grad():
                    # ONE PROBLEM is how to deal with dropout! unless calculating again without dropout?
                    y_guess0_as_label = model.loss_fn[0].as_label(y_guess[0])
                    y_labels[idx_f1] = mitas_utils.pfarm_metric(y_guess0_as_label, y_labels[0], batch=True, threshold=0.5)
                    y_labels[idx_f1] = mi.to_tensor(y_labels[idx_f1][:,1], dtype='float32', stop_gradient=True)

            batch_len = x_inputs[0].shape[0]  # may differ from batch_size if drop_last=False
            istart = iend
            iend = istart + batch_len
            # try:
            loss_this_epoch[istart:iend, 0] = model.epoch
            loss_this_epoch[istart:iend, 1] = model.batch
            loss_this_epoch[istart:iend, 2] = model.recap + 1 # goes to the next recap
            loss_this_epoch[istart:iend, 3] = samples_idx
            loss_this_epoch[istart:iend, 4] = samples_len
            # except:
            #     print(loss_this_epoch.shape)
            #     print(x_input[0].shape)
            #     print(istart, iend, samples_len)

            # calc loss
            # Current design: one loss_fn corresponds to one y_guess and one y_label

            # loss_for_backprop = mi.to_tensor(0.0, dtype='float32', stop_gradient=False)
            # loss_per_instance = np.zeros((num_samples), dtype=np.float32)
            # std_per_instance = np.zeros((num_samples), dtype=np.float32)
            for _iloss, loss_fn in enumerate(model.loss_fn):
                _loss_for_backprop, _loss_vs_sample, _std_vs_sample = loss_fn(
                        y_guess[_iloss], y_labels[_iloss], seqs_len=samples_len, **args.loss_twargs)

                # multiply by loss_fn_scale[_i] if needed
                if args.loss_fn_scale[_iloss] == 0.0:
                    _loss_for_backprop = 0.0
                    _loss_vs_sample = 0.0
                    _std_vs_sample = 0.0
                elif args.loss_fn_scale[_iloss] != 1.0:
                    _loss_for_backprop *= args.loss_fn_scale[_iloss]
                    _loss_vs_sample *= args.loss_fn_scale[_iloss]
                    _std_vs_sample *= args.loss_fn_scale[_iloss]
                elif args.loss_fn_scale[_iloss] == 1.0:
                    pass
                else:
                    logger.error(f'Cannot handle args.loss_fn_scale: {args.loss_fn_scale}!!!')

                if loss_for_backprop is None:
                    loss_for_backprop = _loss_for_backprop
                else:
                    loss_for_backprop += _loss_for_backprop
                loss_this_epoch[istart:iend, 5] += _loss_vs_sample
                loss_this_epoch[istart:iend, 6] += _std_vs_sample

            # loss = loss_for_backprop.mean() # + 1 / ( 1 + 2 * mi.square(y_model - 0.5).mean())
            # if len(loss_for_backprop) != 1:
            #     logger.critical('loss_for_backprop is a VECTOR, not a number, check!!!')
            loss_backprop_count += 1
            loss_for_backprop.backward()
            loss_for_backprop = None

            if args.verbose > 1:
                print("Current state of net parameters:")
                for par_n, par_v in model.net.named_parameters():
                    print(f'{par_n:28s} - min: {par_v.min().numpy()[0]:11.6f}, max: {par_v.max().numpy()[0]:11.6f}, ' + \
                            f'grad_min: {par_v.grad.min().item():11.6f}, grad_max: {par_v.grad.max().item():11.6f}')

            if loss_backprop_count == args.optim_step_stride:
                # loss_backprop_count *= args.batch_size
                # if loss_backprop_count > 1:
                    # loss_for_backprop /= loss_backprop_count
                model.optim.step()
                model.optim.clear_grad()
                loss_backprop_count = 0
                # loss_for_backprop = None

            # get the metrics
            # acc = mi.metric.accuracy(mi.flatten(y_model), mi.flatten(y_truth))
            if model.metric_fn is not None:
                with mi.no_grad():
                    _metric_vs_sample = []
                    for _imetric, metric_fn in enumerate(model.metric_fn):
                        y_guess[_imetric] = model.loss_fn[_imetric].as_label(y_guess[_imetric])
                        _metric_vs_sample.append(metric_fn(y_guess[_imetric], y_labels[_imetric], batch=True))
                    metric_this_epoch[istart:iend] = np.concatenate(_metric_vs_sample, axis=-1)

            batches_since_recap += 1
            if logger.level <= logging.INFO and (ibatch + iepoch == 0 or
                    (ibatch + 1) * args.num_recaps_per_epoch % model.num_batches < args.num_recaps_per_epoch):
                model.recap += 1
                # backtrack to the last recap
                istart = max([0, iend - batches_since_recap * args.batch_size])
                batches_since_recap = 0
                if model.metric_fn is not None:
                    metric_recap = ''.join([f', {_label}: {metric_this_epoch[istart:iend, _i].mean():6.4f}'
                        for _i, _label in enumerate(args.metric_labels)])
                else:
                    metric_recap = ''

                loss_recap_mean = loss_this_epoch[istart:iend, 5].mean()
                std_recap_mean = loss_this_epoch[istart:iend, 6].mean()

                logger.info(f'epo/bat/i: {model.epoch:03d}/{model.batch:,}/{ibatch}' + \
                            f'({int(ibatch / model.num_batches * 100):2d}%), ' + \
                            f'loss: \033[0;36m{loss_recap_mean:6.4f}\033[0m, ' + \
                            f'std: {std_recap_mean:6.4f}, lr: {model.optim.get_lr():0.2g}' + \
                            metric_recap)

                # save the first y_model and y_truth (already randomized anyway)
                if args.visual_dl:
                    # mitas_utils.save_model_prediction(y_model[0:1], save_dir=visualdl_dir, labels=seqs_idx[0:1],
                    #     seqs_len=seqs_len[0:1], stem='predict')
                    # mitas_utils.save_model_prediction(y_truth[0:1], save_dir=visualdl_dir, labels=seqs_idx[0:1],
                    #     seqs_len=seqs_len[0:1], stem='input')

                    visualdl_writer.add_scalar(tag='loss', step=model.batch, value=loss_recap_mean)
                    visualdl_writer.add_scalar(tag='std', step=model.batch, value=std_recap_mean)

                # learning_rate tuning
                # if not isinstance(model.lr_scheduler, float): # mi.optimizer.lr.LRScheduler):
                #     model.lr_scheduler.step(loss_for_recap)

            # callback (usually for early stopping)
            if args.evaluate_callback is not None and (ibatch + iepoch == 0 or
                    (ibatch + 1) * args.num_evals_per_epoch % model.num_batches < args.num_evals_per_epoch):
                # ibatch % chkpt_interval == 0 # or model.batch == model.num_batches - 1
                # evaluate() has net.eval() and no_grad()
                model.eval_trail.update(epoch=model.epoch, batch=model.batch, ibatch=ibatch, lr=new_lr, 
                    step_stride=args.optim_step_stride, batch_size=args.batch_size, stage=model.stage)
                model.eval_trail = args.evaluate_callback(model=model, trail=model.eval_trail)
                model.net.train()

        # post-epoch
        if model.metric_fn is None:
            loss_this_epoch = pd.DataFrame(loss_this_epoch[:iend, :],
                    columns=['epoch', 'batch', 'recap', 'idx', 'len', 'loss', 'loss_std', 'lr', 'step_stride', 'batch_size', 'stage'], dtype=np.float32)
        else:
            loss_this_epoch = pd.DataFrame(
                    np.concatenate([loss_this_epoch[:iend, :], metric_this_epoch[:iend, :]], axis=1),
                    columns=['epoch', 'batch', 'recap', 'idx', 'len', 'loss', 'loss_std', 'lr', 'step_stride', 'batch_size', 'stage'] + args.metric_labels,
                    dtype=np.float32)

        # consider to change datatype?
        loss_this_epoch = loss_this_epoch.astype({'epoch':np.int32, 'batch':np.int32, 'recap':np.int32, 'idx':np.int32,
            'len':np.int32, 'loss':np.float32, 'loss_std': np.float32, 'lr': np.float32, 'step_stride': np.int32,
            'batch_size': np.int32, 'stage': np.int32})
        model.train_loss.append(loss_this_epoch) # this will be saved as csv

        # Need to check whether pd.append and pd.grouby retains the order
        train_loss_per_epoch = pd.concat([train_loss_per_epoch, loss_this_epoch.mean().to_frame().T], ignore_index=True)
        # loss_vs_batch = loss_vs_batch.append(loss_one_epoch.groupby('batch').mean(), ignore_index=True)

        # summarize train loss during the last epoch
        if model.metric_fn is None:
            metric_recap = ''
        else:
            metric_recap = ', '.join([f'{_s}: {train_loss_per_epoch[_s].iat[-1]:6.4f}' for _s in args.metric_labels])

        logger.info(f'Epoch {model.epoch:03d} average training loss: ' +
                    misc.str_color(f'{train_loss_per_epoch.loss.iat[-1]:6.4f}', style='reverse')  + ' std: ' +
                    misc.str_color(f'{train_loss_per_epoch.loss_std.iat[-1]:6.4f}', style='reverse') + ', ' + \
                    metric_recap)
                    # f'\033[0;46m{loss_vs_epoch.loss.iat[-1]:6.4f}\033[0m' +
                    # f' std: {loss_vs_epoch.loss_std.iat[-1]:6.4f}')

        # summarize valid loss during the last epoch (only the last call to evaluate is used!!!)
        valid_loss_per_epoch = pd.concat([valid_loss_per_epoch, model.eval_trail.valid_loss_by_call[-1].to_frame().T], axis=0, ignore_index=True)
        # valid_loss_per_epoch = valid_loss_per_epoch.append(
        #     model.eval_trail.valid_loss_by_call[model.eval_trail.valid_loss_by_call['epoch'] == model.epoch].mean(),
        #     ignore_index=True)
        #valid_db.valid_loss_by_call.groupby('epoch').mean().reset_index()

        if model.metric_fn is None:
            metric_recap = ''
        else:
            metric_recap = ', '.join([f'{_s}: {valid_loss_per_epoch[_s].iat[-1]:6.4f}' for _s in args.metric_labels])

        logger.info(f'Epoch {model.epoch:03d} average evaluate loss: ' +
                    misc.str_color(f'{valid_loss_per_epoch.loss.iat[-1]:6.4f}', style='reverse')  + ' std: ' +
                    misc.str_color(f'{valid_loss_per_epoch.loss_std.iat[-1]:6.4f}', style='reverse') + ', ' + \
                    metric_recap)
                    # f'\033[0;46m{valid_vs_epoch.loss.iat[-1]:6.4f}\033[0m' +
                    # f' std: {valid_vs_epoch.loss_std.iat[-1]:6.4f}')

        if args.save_dir and args.save_level > 0:
            mitas_utils.save_loss_csv(args.save_dir / 'train_log.csv', pd.concat(model.train_loss), groupby=['recap'])
            mitas_utils.save_loss_csv(args.save_dir / 'valid_log.csv', pd.concat(model.eval_trail.valid_loss), groupby=['epoch', 'batch'])

        if args.save_dir and args.save_level > 1:
            epoch_log_save_dir = args.save_dir / 'epoch_log'
            epoch_log_save_dir.mkdir(parents=True, exist_ok=True)
            mitas_utils.save_loss_csv(epoch_log_save_dir / f'train_epo{model.epoch:03d}_{args.objective}_{train_loss_per_epoch[args.objective].iat[-1]:6.4f}.csv', loss_this_epoch)
            mitas_utils.save_loss_csv(epoch_log_save_dir / f'valid_epo{model.epoch:03d}_{args.objective}_{valid_loss_per_epoch[args.objective].iat[-1]:6.4f}.csv', model.eval_trail.valid_loss[-1])

        # stop the train if needed
        if (not args.nonstop) and model.epoch >= args.trainloss_patience and model.epoch >= args.validloss_patience:

            if all((train_loss_per_epoch.loss.diff() > 0)[-args.trainloss_patience:]):
                logger.critical(f'Training loss increased for {args.trainloss_patience} consecutive epochs, stopping!!!')
                break
            if all(train_loss_per_epoch.loss.pct_change().abs()[-args.trainloss_patience:] < args.trainloss_rdiff):
                logger.critical(f'Training loss changed < {args.trainloss_rdiff * 100}% for {args.trainloss_patience} consecutive epochs, stopping!!!')
                break
            if all((valid_loss_per_epoch.loss.diff() > 0)[-args.validloss_patience:]):
                logger.critical(f'Validate loss increased for {args.validloss_patience} consecutive epochs, stopping!!!')
                break
            if all(valid_loss_per_epoch.loss.pct_change().abs()[-args.validloss_patience:] < args.validloss_rdiff):
                logger.critical(f'Validate loss changed < {args.validloss_rdiff * 100}% for {args.validloss_patience} consecutive epochs, stopping!!!')
                break

        ####### Save Model if "save" file exists #######
        if sv_file and sv_file.exists() and args.save_dir:
            logger.info(f'Save current model as directed by file: {sv_file.as_posix()}')
            sv_file.unlink()
            if len(model.eval_trail.valid_loss):
                save_model_optim(model, save_dir=args.save_dir / (f'save_epo{model.epoch:03d}_{args.objective}' + \
                    f'_{model.eval_trail.valid_loss[-1][args.objective].mean():6.4f}'))
            else:
                save_model_optim(model, save_dir=args.save_dir / (f'save_epo{model.epoch:03d}_train_{args.objective}' + \
                    f'_{model.train_loss[-1][args.objective].mean():6.4f}'))

    # post-train
    if args.visual_dl:
        visualdl_writer.close()

    # model.train_loss = pd.concat(model.train_loss)
    model.train_loss_per_epoch = train_loss_per_epoch
    model.valid_loss = model.eval_trail.valid_loss
    model.valid_loss_per_epoch = valid_loss_per_epoch
    # model.eval_trail.valid_loss_by_call.groupby('epoch').min().reset_index()

    # get the objective
    objective_loss_by_epoch = valid_loss_per_epoch if len(valid_loss_per_epoch) else train_loss_per_epoch

    if args.objective_direction in ['maximize', 'max', 'up']:
        model.objective = objective_loss_by_epoch[args.objective].max() # model.eval_trail.valid_loss[-1][args.objective].mean()
    elif args.objective_direction in ['minimize', 'min', 'down']:
        model.objective = objective_loss_by_epoch[args.objective].min() # model.eval_trail.valid_loss[-1][args.objective].mean()
    else:
        model.objective = None
        logger.critical(f'Unrecognized args.objective_direction: {args.objective_direction}!!!')

    # save model
    if args.save_dir and args.save_level >= 1:
        logger.info(f'Saving final model state dict in <{args.save_dir}>...')
        save_state_dict(model, save_dir=args.save_dir)
        # args.net_src_file = gwio.copy_text_file_to_dir(args.net_src_file, args.save_dir)
        # mitas_utils.save_loss_csv(args.save_dir / 'train_log.csv', model.train_loss, groupby=args.save_groupby)
        # mitas_utils.save_loss_csv(args.save_dir / 'valid_log.csv', model.valid_loss, groupby=args.save_groupby)

    return model.objective


# @mi.no_grad()
def evaluate(model, miset, **kwargs):
    """ Note: model structure is not changed during this call """
    args = misc.Struct(dict(
        batch_size = 8,
        shuffle = False,
        num_recaps_per_epoch = 10,
        save_dir = None,
        drop_last = False,    # whether to drop last batch if < batch_size
        post_process = None,# whether to post-process return
        return_guess = False, # only a flag for returning predicted y
        return_numpy = False, # only applies to y_guess
        verbose = 1,
        ))
    args.update(vars(model.args))
    args.update(kwargs) # kwargs rule all
    if isinstance(args.save_dir, str): args.save_dir = Path(args.save_dir)
    # model.args.update(vars(args)) # args should not change anymore

    if args.valid_size is not None and args.valid_size > 0:
        if args.valid_size < len(miset):
            miset = mitas_utils.random_sample(miset, size=args.valid_size, replace=False)
        elif args.valid_size == len(miset):
            logger.warning(f'Specified valid size: {args.valid_size} == data length: {len(miset)}.')
        else:
            logger.warning(f'Specified valid size: {args.valid_size} > data length: {len(miset)}!')

    if isinstance(miset, MyDataset):
        if args.batch_size > len(miset):
            logger.warning(f'batch_size: {args.batch_size} > data_size: {len(miset)}!!!')
        if args.jit_loader and not miset.data_jit_full:
            logger.info('Setting num_workers=0 for the first epoch with jit_loader=True')
            # set num_worker=0, so that midata is not copied to worker processes
            miloader = get_dataloader(miset, batch_size=min([args.batch_size, len(miset)]),
                    shuffle=args.shuffle, drop_last=False, num_workers=0)
        else:
            miloader = get_dataloader(miset, batch_size=min([args.batch_size, len(miset)]),
                    shuffle=args.shuffle, drop_last=args.drop_last)
    else:
        miloader = miset

    if isinstance(miloader, mi.io.DataLoader):
        num_samples = len(miloader.dataset)
    else:
        num_samples = len(miloader)

    num_batches = len(miloader)
    num_inputs = 1 if isinstance(args.input_genre, str) else len(args.input_genre)
    num_labels = 1 if isinstance(args.label_genre, str) else len(args.label_genre)

    recap_interval = num_batches // args.num_recaps_per_epoch + 1
    logger.info(f'======= Validating =======')
    logger.info(f'        data size: {num_samples}')
    logger.info(f'       jit loader: {args.jit_loader}')
    logger.info(f'       batch size: {args.batch_size}')
    logger.info(f'          shuffle: {args.shuffle}')
    logger.info(f'        drop_last: {args.drop_last}')
    logger.info(f'     # of batches: {num_batches}')
    logger.info(f'   recap interval: {recap_interval}')
    logger.info(f'        loss mask: {args.loss_mask}')
    logger.info(f'        loss sqrt: {args.loss_sqrt}')
    logger.info(f'     post_process: {args.post_process}')

    y_guess_all = [ [] for _ in args.label_genre ]
    # y_truth_all = []
    loss_this_epoch = np.zeros((num_samples, 5), dtype=np.float32)
    if model.metric_fn is not None:
        metric_this_epoch = np.empty((num_samples, len(args.metric_labels)), dtype=np.float32)

    istart, iend = 0, 0 # the start and end idx of the current batch wrt loss_one_epoch
    model.net.eval()
    with mi.no_grad():
        for ibatch, batch_data in tqdm(enumerate(miloader), total=num_samples, desc=f'Evaluating {args.net}'):

            samples_idx = np.array(batch_data[0][:,0], dtype=np.int32)
            samples_len = batch_data[0][:,1]
            if samples_len.dtype is not mi.int32: # if not str(seqs_len.dtype).endswith('int32'):
                samples_len = samples_len.astype(mi.int32) # works for both numpy and paddle

            x_inputs = batch_data[1:num_inputs+1] # from the 2nd item
            y_labels = batch_data[-num_labels:] # always at the ends

            # if args.batch_size == 1 and x.ndim > 1 and x.shape[0] == 1 \
            #         and x.shape[1] > samples_len[0] and not args.loss_padding:
            #     x = mitas_utils.cut_padding(x, samples_len[0])
            #     y_truth = mitas_utils.cut_padding(y_truth, samples_len[0])

            y_guess = model.net(*x_inputs, samples_len)
            if type(y_guess) not in (tuple, list):
                y_guess = [y_guess]

            # it seems to be better to calculate some labels here
            if 'f1' in args.label_genre:
                idx_f1 = args.label_genre.index('f1')
                with mi.no_grad():
                    y_guess0_as_label = model.loss_fn[0].as_label(y_guess[0])
                    y_labels[idx_f1] = mitas_utils.pfarm_metric(y_guess0_as_label, y_labels[0], batch=True, threshold=0.5)
                    y_labels[idx_f1] = mi.to_tensor(y_labels[idx_f1][:,1], dtype='float32')

            num_samples = x_inputs[0].shape[0]
            istart = iend
            iend = istart + num_samples
            loss_this_epoch[istart:iend, 0] = ibatch
            loss_this_epoch[istart:iend, 3] = samples_len
            loss_this_epoch[istart:iend, 4] = samples_idx

            for _iloss, loss_fn in enumerate(model.loss_fn):
                _, _loss_vs_sample, _std_vs_sample = loss_fn(
                        y_guess[_iloss], y_labels[_iloss], seqs_len=samples_len, **args.loss_twargs)

                # multiply by loss_fn_scale[_i] if needed
                if args.loss_fn_scale[_iloss] == 0.0:
                    _loss_vs_sample = 0.0
                    _std_vs_sample = 0.0
                elif args.loss_fn_scale[_iloss] != 1.0:
                    _loss_vs_sample *= args.loss_fn_scale[_iloss]
                    _std_vs_sample *= args.loss_fn_scale[_iloss]

                loss_this_epoch[istart:iend, 1] += _loss_vs_sample
                loss_this_epoch[istart:iend, 2] += _std_vs_sample

            # loss_per_instance = np.zeros((num_samples), dtype=np.float32)
            # std_per_instance = np.zeros((num_samples), dtype=np.float32)
            # for i in range(num_seqs):
            #     seq_len = int(seqs_len[i])
            #     seq_loss = F.mse_loss(y_model[i, :seq_len], y_truth[i, :seq_len], reduction='none').numpy()
            #     loss_vs_seq[i] = np.sqrt(seq_loss.mean())
            #     std_vs_seq[i] = np.sqrt(seq_loss).std()

            if model.metric_fn is not None or args.return_guess or args.post_process is not None:
                for _iloss, loss_fn in enumerate(model.loss_fn):
                    y_guess[_iloss] = loss_fn.as_label(y_guess[_iloss])
                # print(x[:,:10,4:8])
                # print(y_model.sum(axis=1))

                if model.metric_fn is not None:
                    _metric_vs_sample = []
                    for _imetric, metric_fn in enumerate(model.metric_fn):
                        # y_guess[_imetric] = model.loss_fn[_imetric].as_label(y_guess[_imetric])
                        _metric_vs_sample.append(metric_fn(y_guess[_imetric], y_labels[_imetric], batch=True))
                    metric_this_epoch[istart:iend] = np.concatenate(_metric_vs_sample, axis=-1)

                ### CAUTION: post_process and return_guess are not tested yet!
                # if args.post_process is not None: # x[:,:,4:8] or batch_data is the raw x in onehot format
                #     y_guess[0] = post_process_output(y_guess[0], batch_data[-num_labels-1],
                #         method=args.post_process, seqs_len=samples_len)

                if args.return_guess:   # return y_model_all
                    for _i in range(len(y_guess)):
                        # logger.info('processing y_guess for returning')
                        if args.return_numpy:
                            y_guess_all[_i].extend(y_guess[_i].numpy())
                        else:
                            y_guess_all[_i].extend(y_guess[_i])

            if logger.level <= logging.INFO and ibatch % recap_interval == 0:
                istart = max([0, iend - recap_interval * args.batch_size])
                if model.metric_fn is None:
                    metric_recap = ''
                else:
                    metric_recap = ''.join([f', {_label}: {metric_this_epoch[istart:iend, _i].mean():6.4f}'
                        for _i, _label in enumerate(args.metric_labels)])

                logger.info(f'batch: {ibatch:4d} ({int(ibatch / num_batches * 100):2d}%), ' + \
                            f'loss: \033[0;36m{loss_this_epoch[istart:iend, 1].mean():6.4f}\033[0m, ' + \
                            f'std: {loss_this_epoch[istart:iend, 2].mean():6.4f}{metric_recap}')

    # post-epoch
    if model.metric_fn is None:
        loss_this_epoch = pd.DataFrame(loss_this_epoch[:iend, :],
                columns=['batch', 'loss', 'loss_std', 'len', 'idx'])
        metric_recap = ''
    else:
        loss_this_epoch = pd.DataFrame(
                np.concatenate([loss_this_epoch[:iend, :], metric_this_epoch[:iend, :]], axis=1),
                columns=['batch', 'loss', 'loss_std', 'len', 'idx'] + args.metric_labels)
        metric_recap = ', '.join([f'{_s}: {loss_this_epoch[_s].mean():6.4f}' for _s in args.metric_labels])

    loss_this_epoch = loss_this_epoch.astype({'batch':np.int32, 'idx':np.int32, 'len':np.int32,
        'loss':np.float32, 'loss_std': np.float32})

    if logger.level <= logging.INFO:
        logger.info(f'Validate mean: \033[0;46m{loss_this_epoch.loss.mean():6.4f}\033[0m' +
                    f', std: {loss_this_epoch.loss.std():6.4f}, ' + metric_recap)

    if args.save_dir and args.save_level > 0:
        if not args.save_dir.exists(): args.save_dir.mkdir(parents=True)
        mitas_utils.save_loss_csv(args.save_dir / 'valid_loss.csv', loss_this_epoch)

    if args.return_guess:
        return loss_this_epoch, y_guess_all
    else:
        return loss_this_epoch


# @mi.no_grad()
def predict(model, miset, **kwargs):
    """  """
    args = misc.Struct(dict(
                    batch_size = 8,
                    shuffle = False, # the first two not used yet
                    num_recaps_per_epoch = 10,
                    save_dir = Path.cwd() / 'predict',
                    return_numpy=False,
                    ))
    args.update(vars(model.args))
    args.update(kwargs) # kwargs rule all
    if args.save_dir and not isinstance(args.save_dir, Path):
        args.save_dir = Path(args.save_dir)
    # model.args.update(vars(args))

    if args.predict_size is not None and args.predict_size > 0:
        if args.predict_size < len(miset):
            miset = mitas_utils.random_sample(miset, size=args.predict_size, replace=False)
        elif args.predict_size == len(miset):
            logger.warning(f'Specified predict size: {args.predict_size} == data length: {len(miset)}.')
        else:
            logger.warning(f'Specified predict size: {args.predict_size} > data length: {len(miset)}!')

    if isinstance(miset, MyDataset):
        if args.batch_size > len(miset):
            logger.warning(f'batch_size: {args.batch_size} > data_size: {len(miset)}!!!')
        if args.jit_loader and not miset.data_jit_full:
            logger.info('Setting num_workers=0 for the first epoch with jit_loader=True')
            # set num_worker=0, so that midata is not copied to worker processes
            miloader = get_dataloader(miset, batch_size=min([args.batch_size, len(miset)]),
                    shuffle=args.shuffle, drop_last=False, num_workers=0)
        else:
            miloader = get_dataloader(miset, batch_size=min([args.batch_size, len(miset)]),
                    shuffle=False, drop_last=False)
    else:
        miloader = miset

    recap_interval = len(miloader) // args.num_recaps_per_epoch + 1
    num_samples = len(miset)
    num_inputs = 1 if isinstance(args.input_genre, str) else len(args.input_genre)
    num_labels = 1 if isinstance(args.label_genre, str) else len(args.label_genre)

    logger.info(f'Predicting, data size: {num_samples}')
    logger.info(f'           jit loader: {args.jit_loader}')
    logger.info(f'           batch size: {args.batch_size}')
    logger.info(f'            drop_last: {args.drop_last}')
    logger.info(f'              shuffle: {args.shuffle}')
    logger.info(f'         # of batches: {len(miloader)}')
    logger.info(f'       recap interval: {recap_interval}')
    logger.info(f'       post_process: {args.post_process}')

    if args.save_dir and args.save_level >= 1:
        args.save_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f'Predicted files will be saved in: {args.save_dir}')

    # two returned values
    y_guess_all = [ [] for _ in args.label_genre ] # np.empty((data_size, midata[0][0].shape[0]), dtype=np.float32)
    seqs_len_all = [] # np.empty((data_size), dtype=np.int32)

    model.net.eval()
    with mi.no_grad(), tqdm(total=num_samples, desc=f'Predicting {args.net}', disable=False) as prog_bar:
        for ibatch, batch_data in enumerate(miloader()):

            seqs_idx, seqs_len = batch_data[0][:, 0], batch_data[0][:, 1]
            x_inputs = batch_data[1:num_inputs+1] # from the 2nd item

            # if args.batch_size == 1 and x.ndim > 1 and x.shape[0] == 1 and x.shape[1] > seqs_len[0] and not args.loss_padding:
            #     x = mitas_utils.cut_padding(x, int(seqs_len[0]))
            #     y_truth = mitas_utils.cut_padding(y_truth, int(seqs_len[0]))
                # if 'bpmat' in args.input_genre:
                    # bpmat = mitas_utils.cut_padding(bpmat, seqs_len[0])

            y_guess = model.net(*x_inputs, seqs_len)
            if type(y_guess) not in (tuple, list):
                y_guess = [y_guess]

            # if args.post_process is not None:
            # y_guess = model.loss_fn[0].as_label(y_guess)

            for _iloss, loss_fn in enumerate(model.loss_fn):
                y_guess[_iloss] = loss_fn.as_label(y_guess[_iloss])

            if args.post_process is not None:
                y_guess = post_process_guess(y_guess, batch_data[-num_labels-1], method=args.post_process)

            # done with all calculation, collect and save
            for _i in range(len(y_guess)):
                if args.return_numpy:
                    # y_guess = y_guess.numpy()
                    y_guess_all[_i].extend(y_guess[_i].numpy())
                else:
                    y_guess_all[_i].extend(y_guess[_i])

            # seqs_len_all.extend(seqs_len.numpy())

            if args.save_dir: # obsolete now!!!
                # in the case of multiple loss_fns, their as_label() should give the same results
                # for loss_fn in model.loss_fn:
                mitas_utils.save_predict_matrix(y_guess, args.save_dir, seqs_len=seqs_len,
                            names=seqs_idx.numpy().astype(int))
            prog_bar.update(len(y_guess))

    logger.info(f'Completed prediction of {num_samples} samples')
    return y_guess_all #, np.array(seqs_len_all)


def save_model_optim(model, save_dir=Path.cwd()):
    # saving model always wrecks havoc for me
    # logger.info(f'Saving jit model file in: {args.save_dir}')
    # model.net.train()
    # mi.jit.save(model.net, (args.save_dir / 'model_train').as_posix(),
    #     input_spec=model.net.Embed.in_shapes)
    #     # input_spec=[InputSpec(shape=[2,512, 15], dtype='float32', name='x'),
    #                 # InputSpec(shape=[2, 1], dtype='int32', name='seqs_len')])

    # save the model
    # model_path = new_save_dir.parent / 'model_eval'
    # if not model_path.with_suffix('.pdmodel').exists():
    #     logger.info(f'Saving model diagram to: {model_path} ...')
    #     mi.jit.save(model.net, model_path.as_posix(), input_spec=model.net.Embed.in_shapes)
    #     if not model.args.to_static:
    #         mi.disable_static()

    save_dir = Path(save_dir) if save_dir else Path.cwd()
    save_state_dict(model, save_dir=save_dir)
    # model.args.net_src_file =
    gwio.copy_text_file_to_dir(model.args.net_src_file, save_dir)
    gwio.dict2json(vars(model.args), save_dir / 'args.json')
    if hasattr(model, 'train_loss') and isinstance(model.train_loss, list) and len(model.train_loss):
        mitas_utils.save_loss_csv(save_dir / f'train_epo{model.epoch:03d}_{model.args.objective}_{model.train_loss[-1][model.args.objective].mean():8.6f}.csv', model.train_loss[-1])
    if hasattr(model, 'eval_trail') and hasattr(model.eval_trail, 'valid_loss') and \
        len(model.eval_trail.valid_loss):
        mitas_utils.save_loss_csv(save_dir / f'valid_epo{model.epoch:03d}_{model.args.objective}_{model.eval_trail.valid_loss[-1][model.args.objective].mean():8.6f}.csv', model.eval_trail.valid_loss[-1])
    logger.info(f'Saved model and optim: {save_dir}')
    return None


def save_state_dict(model, save_dir=Path.cwd()):
    """  """
    if isinstance(save_dir, str): save_dir = Path(save_dir)
    if not save_dir.exists(): save_dir.mkdir(parents=True)

    net_state_file = save_dir / 'net.state'
    opt_state_file = save_dir / 'opt.state'

    mi.save(model.net.state_dict(), net_state_file.as_posix())
    mi.save(model.optim.state_dict(), opt_state_file.as_posix())

    logger.info(f'Saved model states in: {save_dir}')


def load_state_dict(model, load_dir=Path.cwd(), load_log=True):
    """ load_log: whether to load train_log.csv and valid_log.csv """
    if isinstance(load_dir, str): load_dir = Path(load_dir)
    logger.info(f'Loading model states from: {load_dir}')

    net_state_file = load_dir / 'net.state'
    opt_state_file = load_dir / 'opt.state'

    try:
        if net_state_file.exists():
            net_state_dict = mi.load(net_state_file.as_posix())
            model.net.set_state_dict(net_state_dict)
            logger.info(f'Loaded net state: {net_state_file}')
    except:
        logger.error('Error in net state_dict loading!!!')

    try:
        if opt_state_file.exists():
            opt_state_dict = mi.load(opt_state_file.as_posix())
            model.optim.set_state_dict(opt_state_dict)
            logger.info(f'Loaded optim state: {opt_state_file}')
    except Exception as err:
        sys.stderr.write(str(err)+'\n')
        logger.error('Error in optim state_dict loading!!!')

    train_loss_file = load_dir / 'train_log.csv'
    if load_log and train_loss_file.exists():
        logger.info(f'Loading train log file: {train_loss_file} ...')
        model.train_loss = pd.read_csv(train_loss_file)
        if 'irecap' in model.train_loss:
            model.train_loss.rename(columns={'irecap':'recap'}, inplace=True)
        model.train_loss = model.train_loss.astype(
            {'epoch':np.int32, 'batch':np.int32, 'recap':np.int32, 'idx':np.int32,
            'len':np.int32, 'loss':np.float32, 'loss_std': np.float32}, errors='ignore')

    valid_loss_file = load_dir / 'valid_log.csv'
    if load_log and valid_loss_file.exists():
        logger.info(f'Loading valid log file: {valid_loss_file} ...')
        valid_loss = pd.read_csv(valid_loss_file)
        if 'irecap' in valid_loss:
            valid_loss.rename(columns={'irecap':'recap'})
        valid_loss = valid_loss.astype({'batch':np.int32, 'idx':np.int32,
            'len':np.int32, 'loss':np.float32, 'loss_std': np.float32}, errors='ignore')
        model.eval_trail = misc.Struct(valid_loss=[valid_loss])

    return model

def get_net(args, MiNets=None, quiet=False):
    """  """
    misc.logger_setlevel(logger, args.verbose)
    if isinstance(args.load_dir, str): args.load_dir = Path(args.load_dir)
    if isinstance(args.net_src_file, str): args.net_src_file = Path(args.net_src_file)
    if MiNets is None: MiNets = globals()['MiNets']

    # get the local src code path
    if args.load_dir and args.net_src_file:
        my_src_code = args.load_dir / args.net_src_file.name
    else:
        my_src_code = Path(args.net_src_file)

    # reload the net if a local copy exits
    if my_src_code.exists() and os.path.exists(MiNets.__file__) and not my_src_code.samefile(MiNets.__file__):
        args.net_src_file = my_src_code
        logger.info(f'Found local net definition: {my_src_code}')

        # a clumsy and fragile way to handle this...
        tmp_src_code = Path('/tmp') / Path(MiNets.__file__).name
        tmp_src_code.unlink(missing_ok=True)
        tmp_src_code.symlink_to(my_src_code.resolve())
        spec = importlib.util.spec_from_file_location('LocalMiNets', tmp_src_code)
        MyNets = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(MyNets)
        # spec.loader.exec_module(sys.modules['LocalMiNets'])
        # MyNets = sys.modules['LocalMiNets']

        # below doesn't work if the file extension is not .py
        # sys.path.insert(0, my_src_code.parent.absolute().as_posix())
        # if my_src_code.name == Path(MiNets.__file__).name:
        #     MyNets = importlib.reload(MiNets)
        # else:
        #     MyNets = importlib.import_module(my_src_code.stem)
        # sys.path.remove(my_src_code.parent.absolute().as_posix())
    else:
        MyNets = MiNets # globals()['MiNets']
        args.net_src_file = MyNets.__file__

    logger.info(f'Used net definition: {misc.str_color(MyNets.__file__, style="reverse")}')

    # locate net classes by name
    net_classes = getmembers(MyNets, isclass)
    net_names = [_s[0].lower() for _s in net_classes]
    idx_net = misc.get_list_index(net_names, args.net.lower())
    if not idx_net:
        idx_net = misc.get_list_index(net_names, args.net.lower() + 'net')
    if not idx_net:
        logger.error(f'No net definition with name: {args.net}(NET) found!')
        return None

    # use the first match
    net_init_fn = net_classes[idx_net[0]][1]
    net = net_init_fn(args)

    # Distributed training
    if args.spawn:
        net = mi.DataParallel(net)
    elif args.fleet:
        net = fleet.distributed_model(net)

    if not quiet and hasattr(net, 'summary') and logger.level <= logging.INFO:
        args.params = net.summary()
        logger.info(f'{args.params}')

    return net


def get_model(args, quiet=False):
    """  """
    # model = mi.Model(upp_net)
    # model.prepare(upp_opt, loss_fn)
    # model.fit(train_dataset, epochs=NUM_EPOCHS, batch_size=BATCH_SIZE, log_freq=200)
    # model.evaluate(test_data, log_freq=20, batch_size=BATCH_SIZE)
    misc.logger_setlevel(logger, args.verbose)
    mi_model = misc.Struct()
    mi_model.args = args
    mi_model.net = get_net(args, MiNets=MiNets, quiet=quiet)
    if args.fleet:
        mi_model.net = fleet.distributed_model(mi_model.net)
    mi_model.optim, mi_model.lr_scheduler = get_optimizer(mi_model.net, args)
    mi_model.loss_fn = get_loss_fn(args)
    mi_model.metric_fn = get_metric_fn(args)

    # make sure args.objective is correct
    if mi_model.metric_fn is not None and args.objective != 'loss' and \
            args.objective not in args.metric_labels:
        logger.error(f'args.objective: {args.objective} is not loss or in metrics, use loss!')
        args.objective = 'loss'
        args.objective_direction = 'minimize'

    return mi_model


# @mi.no_grad()
def evaluate_in_train(model=None, miset=None, trail=misc.Struct(), **kwargs):
    """ miset should be either MyDataSet or DataLoader """
    eval_trail = misc.Struct(
        epoch = 0, # the following three are from the train()
        batch = 0, # so as to align record with the training curve
        batch_size = 32, # for early stop only
        shuffle = False,
        drop_last = False,
        times_called = 0,
        valid_loss = [],
        valid_loss_by_call = [], # pd.DataFrame(), # pd.DataFrame(columns=['ibatch', 'loss', 'loss_std', 'epoch', 'batch']),
        save_dir = None,
        saved_idx = [], # this points to valid_loss
        saved_dirs = [],
        objective = 'loss',
        objective_direction = 'minimize',
        objective_best = None,
        chkpt_save_limit = 7,
        verbose = 1,
    )
    eval_trail.update(vars(trail))
    eval_trail.update(kwargs)
    eval_trail.times_called += 1

    # this is different from args.save_dir
    if eval_trail.save_dir and isinstance(eval_trail.save_dir, str):
        eval_trail.save_dir = Path(eval_trail.save_dir)

    misc.logger_setlevel(logger, 0)
    loss_df = evaluate(model, miset, save_dir=None, return_guess=False, post_process=None,
            shuffle=False, batch_size=eval_trail.batch_size, drop_last=eval_trail.drop_last)
    misc.logger_setlevel(logger, eval_trail.verbose)

    mean_ds = loss_df.mean()
    if model.metric_fn is None:
        metric_recap = ''
    else:
        metric_recap = ''.join([f', {_s}: {mean_ds[_s]:6.4f}' for _s in args.metric_labels])

    logger.info(f'evaluate loss: \033[0;32m{mean_ds.loss:6.4f}\033[0m, ' + \
                f'std: {mean_ds.loss_std:6.4f}{metric_recap}')

    for _col in ['ibatch', 'epoch', 'batch', 'lr', 'step_stride', 'batch_size', 'stage']:
        mean_ds[_col] = eval_trail[_col]
        loss_df[_col] = eval_trail[_col]

    # loss_this_epoch = loss_this_epoch.astype({'epoch':np.int32, 'batch':np.int32, 'recap':np.int32, 'idx':np.int32,
    #     'len':np.int32, 'loss':np.float32, 'loss_std': np.float32, 'lr': np.float32, 'step_stride': np.int32,
    #     'batch_size': np.int32, 'stage': np.int32})

    eval_trail.valid_loss.append(loss_df)
    eval_trail.valid_loss_by_call.append(mean_ds)
    # eval_trail.valid_loss_by_call = eval_trail.valid_loss_by_call.append(mean_ds, ignore_index=True)
    # eval_trail.valid_loss_by_call.loc[len(eval_trail.valid_loss_by_call)] = mean_ds

    if eval_trail.objective not in mean_ds:
        logger.info(f'objective: {eval_trail.objective} not in loss_df, use [loss] column!')
        eval_trail.objective = 'loss'
        eval_trail.objective_direction = 'minimize'

    save_chkpt = False
    # do not do anything with the first call (usually with initialized parameters)
    if eval_trail.times_called > 1:
        if eval_trail.objective_best is None:
            eval_trail.objective_best = mean_ds[eval_trail.objective]
        elif eval_trail.objective_direction in ['up', 'max','maximize']:
            save_chkpt = mean_ds[eval_trail.objective] > eval_trail.objective_best
        elif eval_trail.objective_direction in ['down', 'min', 'minimize']:
            save_chkpt = mean_ds[eval_trail.objective] < eval_trail.objective_best
        else:
            logger.critical(f'Unrecognized configs.objective_direction: {eval_trail.objective_direction}!!!')

    if save_chkpt: # store the best objective
        eval_trail.objective_best = mean_ds[eval_trail.objective]

    if save_chkpt and eval_trail.save_dir and eval_trail.times_called > 1:

        # call "next_backup_path" just in case it exists
        new_save_dir = gwio.next_backup_path(eval_trail.save_dir / \
            f'chkpt_epo{eval_trail.epoch:03d}_{eval_trail.objective}_{mean_ds[eval_trail.objective]:8.6f}')

        logger.info(f'Saved chkpt with {misc.str_color(eval_trail.objective)}=' + \
                    misc.str_color(f'{eval_trail.objective_best:8.6f}') + \
                    f' to {misc.str_color(new_save_dir)}')

        save_model_optim(model, save_dir=new_save_dir)
        # mitas_utils.save_loss_csv(new_save_dir / f'valid_bat{configs.batch}_{mean_ds.loss:6.4f}.csv', loss_df)

        eval_trail.saved_dirs.append(new_save_dir)
        eval_trail.saved_idx.append(len(eval_trail.valid_loss_by_call) - 1)

        for old_save_dir in eval_trail.saved_dirs[:-eval_trail.chkpt_save_limit]:
            if not old_save_dir.exists() or new_save_dir.samefile(old_save_dir): continue
            logger.info(f'Removing chkpt: {old_save_dir} due to save_limit: {eval_trail.chkpt_save_limit}')
            shutil.rmtree(old_save_dir)

    return eval_trail


def load_train_valid_dataset(args):

    # ======= read in data files
    midat = mitas_utils.load_chop_midat(args)

    if args.eval_name is None:
        logger.info(f'Splitting train and test with fraction: {args.split_test}, seed: {args.split_seed}...')
        # stratify = None
        # if args.split_stratify:
        #     if args.split_stratify in midat:
        #         stratify = midat[args.split_stratify]
        #         logger.info(f'Use data column: {misc.str_color(args.split_stratify, style="reverse")} for stratification...')
        #     else:
        #         logger.warning(f'Cannot find split_stratify: {args.split_stratify} in midata!!!')
        # train_data, valid_data = train_test_split(midat, test_size=args.split_test,
        #         random_state=args.split_seed, stratify=stratify)

        train_data, valid_data = brew_midat.split_midat(midat, fraction=1.-args.split_test, stratify=args.split_stratify,
            bucket_key=args.split_bucket_key, bucket_num=args.split_bucket_num, shuffle=args.split_shuffle,
            random_state=args.split_seed, save_pkl=False, save_json=False, save_names=False, quiet=args.verbose)

    else:
        train_data = midat
        data_name = args.data_name # this is overwritten in the load_midata()
        valid_data = mitas_utils.load_chop_midat(args, data_name=args.eval_name)
        args.data_name = data_name

    logger.info('Getting train dataset ...')
    train_set = get_dataset(train_data, args)

    if len(valid_data) > 0:
        logger.info('Getting validation dataset...')
        # jit_loader = args.jit_loader # do this if you want to preload all evaluation dataset
        valid_set = get_dataset(valid_data, args)
        # args.jit_loader = jit_loader
        # valid_loader = get_dataloader(valid_set, shuffle=False,
        #             batch_size=min([args.batch_size, len(valid_set)]),
        #             drop_last=False)
        args.evaluate_callback = func_partial(evaluate_in_train, miset=valid_set, shuffle=False,
                    save_dir=args.save_dir, verbose=args.verbose, batch_size=args.batch_size,
                    objective=args.objective, objective_direction=args.objective_direction)
    else:
        valid_set = None
        args.evaluate_callback = None

    return train_set, valid_set


def scan_data_args(args, miset, data_sizes=None, batch_sizes=None, **kwargs):
    """ data/batch_sizes='auto'|None|int|list/array """
    args.update(kwargs)

    if args.save_dir and isinstance(args.save_dir, str):
        args.save_dir = Path(args.save_dir)
    if args.save_dir: args.save_dir.mkdir(parents=True, exist_ok=True)

    num_data = len(miset)

    # get data grids, the default goes up from 1 by a factor 2
    def get_data_grid(val_in, default=1):
        if val_in is None:
            val_out = [default]
        elif isinstance(val_in, str):
            if val_in.lower() == 'auto':
                num_grids = int(np.log2(num_data)) + 1
                val_out = np.logspace(0, num_grids - 1, num=num_grids, base=2, dtype=int)
            elif val_in.lower() == 'all': # all
                val_out = [num_data]
            else:
                val_out = [1, num_data]
        elif isinstance(val_in, int):
            val_out = [val_in]
        else:
            val_out = np.array(val_in, dtype=int)
        return val_out

    data_sizes = get_data_grid(data_sizes, default=num_data)
    batch_sizes = get_data_grid(batch_sizes, default=args.batch_size)

    scan_best_loss = pd.DataFrame() # np.zeros((len(data_sizes)*len(batch_sizes), 6), dtype=float)
    # the lists here store the train_loss, etc. from each scan
    scan_train_loss = []
    scan_valid_loss = [] # this is the callback return

    data_indices = np.linspace(0, num_data-1, num=num_data, dtype=int)
    for i, (data_size, batch_size) in enumerate(itertools.product(data_sizes, batch_sizes)):
        logger.info(f'scan #{i}, data_size: {data_size}, batch_size: {batch_size}')
        batch_size = int(batch_size) # somehow needed for paddle

        # get train data
        if 0 < data_size < num_data:
            data_indices = np.random.permutation(data_indices)
            train_data = [miset[data_indices[_i]] for _i in range(data_size)]
        else:
            train_data = miset

        # train with chosen data and batch_size
        logger.info('Creating a new model...') # or re-initialize the model
        model = get_model(args)

        # both returns are pd.DataFrames of
        save_dir = args.save_dir
        objective = train(model, train_data, batch_size=batch_size, save_dir=None)
        args.save_dir = save_dir

        # should consider to reduce the level to batch, at least
        scan_train_loss.append(model.train_loss.assign(data_size=data_size, batch_size=batch_size))
        scan_valid_loss.append(model.valid_loss.assign(data_size=data_size, batch_size=batch_size))

        # np.tile([data_size, batch_size], (train_loss.shape[0], 1))
        # train_rmsd[-1,-2]: the last epoch
        # last_train_epoch = train_loss[train_loss.epoch == train_loss.epoch.iat[-1]]
        # last_estop_epoch = estop_loss[estop_loss.epoch == train_loss.epoch.iat[-1]]

        scan_best_loss = scan_best_loss.append(dict(
                    data_size = data_size,
                    batch_size = batch_size,
                    train_loss = model.train_loss_per_epoch.loss.min(),
                    valid_loss = model.valid_loss_per_epoch.loss.min(),
                    ), ignore_index=True)

        if args.save_dir: # save all the train and valid curves from each call
            save_prefix = f'scan_data_size{data_sizes[0]}-{data_sizes[-1]}' + \
                          f'_batch{batch_sizes[0]}-{batch_sizes[-1]}'

            save_file = args.save_dir / (save_prefix + '_train.csv')
            mitas_utils.save_loss_csv(save_file, pd.concat(scan_train_loss), groupby=['data_size', 'batch_size', 'ibatch'])
            logger.info(f'Saved train curve: {save_file}')

            save_file = args.save_dir / (save_prefix + '_valid.csv')
            mitas_utils.save_loss_csv(save_file, pd.concat(scan_valid_loss), groupby=['data_size', 'batch_size', 'ibatch'])
            logger.info(f'Saved valid curve: {save_file}')

            save_file = args.save_dir / (save_prefix + '_best.csv')
            scan_best_loss.to_csv(save_file, index=False, float_format='%.4g')
            logger.info(f'Saved scan summary: {save_file}')

        # valid_loss = evaluate(model, valid_data, args=args, save_dir=None, shuffle=False, batch_size=512)
        # scan_best_loss[i,[4,5]] = valid_loss.loc[:,['loss', 'loss_std']].mean(axis=0)
        # scan_valid_loss.append(valid_loss.assign(
        #     data_size = [data_size] * valid_loss.shape[0],
        #     batch_size = [batch_size] * valid_loss.shape[0]))

        # np.append(np.tile([data_size, batch_size], (valid_rmsd.shape[0], 1)),
        #     valid_rmsd, axis=1)

    return scan_best_loss


def scout_args(args, train_set, valid_set=None, arg_names=None, arg_values=None,
            grid_search=False, **kwargs):
    """ Both arg_names and arg_values are lists of MATCHING names/values """
    # take care of args
    args.update(kwargs)

    if args.save_dir:
        args.save_dir = Path(args.save_dir)
        args.save_dir.mkdir(parents=True, exist_ok=True)

    # scan_best_loss = np.zeros((np.prod(arg_lens), 6), dtype=float)
    scout_best_loss = pd.DataFrame(columns=arg_names + ['train_loss', 'valid_loss'], dtype=float)

    scout_train_loss = []
    scout_valid_loss = []

    if grid_search:
        value_set_list = list(itertools.product(*arg_values))
    else:
        value_set_list = list(zip(*arg_values))

    objectives_info = ((args.objective, args.objective_direction),)
    if args.objective == 'loss':
        objectives_info += ('loss', 'minimize')

    for i, value_set in enumerate(value_set_list):
        scan_args = dict(zip(arg_names, value_set))
        scout_best_loss.loc[i, arg_names] = value_set
        logger.info(misc.str_color(f'args set #: {i + 1}/{len(value_set_list)}, {scan_args}'))

        args.update(scan_args)
        args = mitas_utils.autoconfig_args(args)
        args.update(scan_args)
        args.batch_size = int(args.batch_size)
        args.train_size = int(args.train_size) if args.train_size else None
        args.valid_size = int(args.valid_size) if args.valid_size else None

        if args.rebake_midat: # midata should be the pkldata!!!
            train_data = mitas_utils.bake_midata(train_set, args)
            if valid_set is not None:
                valid_data = mitas_utils.bake_midata(valid_set, args)
                args.evaluate_callback = func_partial(evaluate_in_train, miset=valid_data,
                            save_dir=args.save_dir, verbose=args.verbose, batch_size=args.batch_size,
                            objective=args.objective, objective_direction=args.objective_direction)
        else:
            train_data = train_set

        model = get_model(args)

        save_dir = args.save_dir # train() overwrites save_dir
        objective = train(model, train_data, save_dir=None)
        args.save_dir = save_dir

        # collect information about this run
        # 1) best objectives (and loss)
        if args.objective_direction in ['maximize', 'max', 'up']:
            best_epoch_train = model.train_loss_per_epoch[args.objective].argmax()
            best_epoch_valid = model.valid_loss_per_epoch[args.objective].argmax()
        elif args.objective_direction in ['minimize', 'min','down']:
            best_epoch_train = model.train_loss_per_epoch[args.objective].argmin()
            best_epoch_valid = model.valid_loss_per_epoch[args.objective].argmin()
        else:
            logger.critical(f'Unrecognized args.objective_direction: {args.objective_direction}')
            best_epoch_train = len(model.train_loss_per_epoch) - 1
            best_epoch_valid = len(model.valid_loss_per_epoch) - 1

        scout_best_loss.loc[i, f'train_{args.objective}'] = \
            model.train_loss_per_epoch.iloc[best_epoch_train][args.objective]
        scout_best_loss.loc[i, f'valid_{args.objective}'] = \
            model.valid_loss_per_epoch.iloc[best_epoch_valid][args.objective]

        if args.objective != 'loss':
            scout_best_loss.loc[i, 'train_loss'] = model.train_loss_per_epoch.loss.min()
            scout_best_loss.loc[i, 'valid_loss'] = model.valid_loss_per_epoch.loss.min()

        # 2) best epoch train and valid loss
        scout_train_loss.append(model.train_loss[best_epoch_train].assign(**scan_args)) # append scan_args
        # one epoch could have multiple valid_loss dataframes
        valid_epochs = [_df.iloc[0].epoch for _df in model.valid_loss]
        ivalid = misc.locate_num(valid_epochs, best_epoch_train)
        scout_valid_loss.append(model.valid_loss[ivalid].assign(**scan_args))

        # Saving results
        if args.save_dir: # save all the train and valid curves from each call
            save_prefix = 'scout_args_' + '-'.join(arg_names)

            save_file = args.save_dir / (save_prefix + '_train.csv')
            mitas_utils.save_loss_csv(save_file, pd.concat(scout_train_loss), groupby=arg_names + ['ibatch'])
            logger.info(f'Saved train curve: {save_file}')

            save_file = args.save_dir / (save_prefix + '_valid.csv')
            mitas_utils.save_loss_csv(save_file, pd.concat(scout_valid_loss), groupby=arg_names + ['ibatch'])
            logger.info(f'Saved valid curve: {save_file}')

            save_file = args.save_dir / (save_prefix + '_best.csv')
            scout_best_loss.to_csv(save_file, index=False, float_format='%.4g')
            logger.info(f'Saved scan summary: {save_file}')
        else:
            logger.info(f'scout_args results are not saved with args.save_dir: {args.save_dir}')

    return scout_best_loss


def launch_train(args):
    """  """

    if isinstance(args.scheduler, str) and os.path.isfile(args.scheduler):
        with open(args.scheduler, 'r') as iofile:
            args_scheduler = json.load(iofile)
        if not isinstance(args_scheduler, list):
            logger.error(f'train scheduler: {args.scheduler} must be a list of dicts!!!')
            return
        args.scheduler = args_scheduler    ### save the list of dict, intead of json file which may have changed!
    elif isinstance(args.scheduler, list):
        args_scheduler = args.scheduler
    else:
        args_scheduler = [{}]

    for istage, args_stage in enumerate(args_scheduler):

        logger.info(f'========== Starting training stage #{istage + 1}/{len(args_scheduler)} with stage-specific args: {args_stage} ...')
        args.update(args_stage)

        if istage == 0: # only do this for the first stage
            # ======= Get & save model
            model = get_model(args)
            if args.resume and args.load_dir:
                load_state_dict(model, load_dir=args.load_dir)

            args.net_src_file = gwio.copy_text_file_to_dir(args.net_src_file, args.save_dir)
            args.run_src_file = gwio.copy_text_file_to_dir(__file__, args.save_dir)
            gwio.dict2json(vars(args), fname='args.json', fdir=args.save_dir)
            # save_model_optim(model, save_dir=args.save_dir 'model_train')

            train_set, valid_set = load_train_valid_dataset(args)
        else:
            # if 'loss_fn' in args_stage:
            model.loss_fn = get_loss_fn(args)
            # if 'metric_fn' in args_stage:
            model.metric_fn = get_metric_fn(args)
            # if
            model.optim_fn = get_optimizer(model.net, args)

        if args.spawn:
            mi_distro.spawn(train, args=(model, train_set), )
        else:
            train(model, train_set)


def launch_evaluate(args):
    model = get_model(args)
    if args.load_dir:
        load_state_dict(model, load_dir=args.load_dir)

    midat = mitas_utils.load_chop_midat(args, data_name=args.valid_files,
        # avoid using the data_dir in args.json
        data_dir=args.data_dir if "data_dir" in args.kwargs else None)

    # midat['idx'] = np.arange(len(midat)) + 1

    if args.split_eval:
        midats = brew_midat.split_midat(midat, fraction=1.-args.split_test, stratify=args.split_stratify,
            bucket_key=args.split_bucket_key, bucket_num=args.split_bucket_num, shuffle=args.split_shuffle,
            random_state=args.split_seed, save_pkl=False, save_json=False, save_names=False, quiet=args.verbose)        
        suffixes = ['TR', 'VL']
    else:
        midats = [midat]
        suffixes = ['']
    
    for idat, midat in enumerate(midats):
        midat.reset_index(inplace=True, drop=True)

        miset = get_dataset(midat, args)

        # get save path
        save_stem = Path(args.valid_files[0]).resolve()
        if len(args.valid_files) == 1:
            save_stem = f'{save_stem.parent.stem}.{save_stem.stem}{suffixes[idat]}.evaluate'
        else:
            save_stem = f'{save_stem.parent.stem}{suffixes[idat]}.evaluate'

        # if args.save_dir.samefile(args.load_dir):
        if 'save_dir' in args.kwargs:
            eval_save_path = args.save_dir / save_stem
            gwio.dict2json(vars(args), fname='args.json', fdir=eval_save_path)
        else:
            eval_save_path = args.load_dir / save_stem
        eval_save_path.mkdir(parents=True, exist_ok=True)

        # this for for backward compatibility
        if isinstance(args.label_genre, str):
            args.label_genre = [args.label_genre]

        # all returns are numpys or pandas frames
        loss_df, y_guess_all = evaluate(model, miset, shuffle=False, drop_last=False,
            save_dir=eval_save_path, return_guess=True, return_numpy=True)

        # save these cols into loss csv file
        cols_from_midat = [_col for _col in ['idx', 'moltype', 'file', 'db', 'id'] if _col in midat]
        loss_df = loss_df.merge(midat[cols_from_midat], left_index=True, right_index=True, how='left', copy=True, indicator=False, suffixes=[None, '_midat'])

        # add predicted f1 to the loss csv
        if 'f1' in args.label_genre:
            # find where f1 is in guess and label
            if isinstance(args.label_genre, str):
                if1 = 0
            else:
                if1 = args.label_genre.index('f1') - len(args.label_genre)
            loss_df['f1_guess'] = np.concatenate(y_guess_all[if1], axis=0)

        loss_df.to_csv(eval_save_path / 'valid_loss_meta.csv', index=False)

        # do post processing here (should move to post_analyze_output if growing too long)
        if 'ct' in args.label_genre:
            # find where ct is in guess and label (ctmat in fact)
            if isinstance(args.label_genre, str):
                ict = 0
            else:
                # use negative so that it also works for the label data
                ict = args.label_genre.index('ct') - len(args.label_genre)
            y_guess_ppm = y_guess_all[ict]
            y_label_ppm = [_data[ict] for _data in miset]

            # get the variances from the ppmat
            pfarm_var_fn = functools.partial(mitas_utils.pfarm_variance, batch=False, beta=1.0, threshold=None)
            pfarm_var_mat = misc.mpi_map(pfarm_var_fn, list(zip(y_guess_ppm, y_label_ppm)), starmap=True,
                        desc=f'run pfarm_variance')
            pfarm_var_mat = np.stack(pfarm_var_mat, axis=0)
            pfarm_labels = ['pre', 'f1', 'acc', 'rec', 'mcc']

            for _i, metric in enumerate(pfarm_labels):
                midat[f'{metric}_var'] = pfarm_var_mat[:, _i]
                loss_df[f'{metric}_var'] = pfarm_var_mat[:, _i]

            def pfarm_metrics_with_threshold(threshold):
                pfarm_fn = functools.partial(mitas_utils.pfarm_metric, batch=False, beta=1.0, threshold=threshold)
                pfarm_mat = misc.mpi_map(pfarm_fn, list(zip(y_guess_ppm, y_label_ppm)), starmap=True,
                            desc=f'run pfarm_metric with threshold={threshold:.4f}')
                pfarm_mat = np.stack(pfarm_mat, axis=0)
                pfarm_mean = pfarm_mat.mean(axis=0)
                logger.info('New metrics:: ' + ','.join([f'{pfarm_labels[_i]}: {pfarm_mean[_i]:.4f}' for _i in range(len(pfarm_labels))]))

                loss_df_copy = loss_df.copy(deep=True) 
                for _i, metric in enumerate(pfarm_labels):
                    midat[metric] = pfarm_mat[:, _i]
                    if metric in loss_df_copy and f'{metric}_eval' not in loss_df_copy:
                        loss_df_copy.rename(columns={metric:f'{metric}_eval'}, inplace=True)
                    loss_df_copy[f'{metric}'] = pfarm_mat[:, _i]

                return loss_df_copy

            if args.output_finetune is not None and \
                    set(['grid_search', 'gridsearch', 'grid-search']) & set(args.output_finetune):
                import eval_mitas
                logger.info('Running grid search for guess vs label ...')
                threshold_by_grid_search = eval_mitas.pfarm_grid_search_threshold(y_guess_ppm, y_label_ppm,
                    grid_min=0.0, grid_max=1.0, grid_num=20, tolerance=0.001, max_iter=100, metric='f1',)

                # calculate new metrics with threshold
                logger.info(f'Grid search found best threshold: {threshold_by_grid_search} ...')

                loss_df_finetuned = pfarm_metrics_with_threshold(threshold_by_grid_search)
                loss_df_finetuned.to_csv(eval_save_path / 'valid_loss_finetuned.csv', index=False)

                # get loss_df_final
                if args.output_threshold is None: # the same finetuned and final csv
                    args.output_threshold = threshold_by_grid_search
                    loss_df_final = loss_df_finetuned
                else:
                    logger.warning(f'Using args.output_threshold: {args.output_threshold} instead of threshold found by grid search!!!')
                    loss_df_final = pfarm_metrics_with_threshold(args.output_threshold)
            else:
                loss_df_final = pfarm_metrics_with_threshold(0.5 if args.output_threshold is None else args.output_threshold)

            # save loss summary file
            loss_df_final.to_csv(eval_save_path / 'valid_loss_final.csv', index=False)

        # save all evaluate results below

        # add loss_df and y_guess_all to the dataframe
        for i, label_genre in enumerate(args.label_genre):
            loss_df[f'{label_genre}_guess'] = y_guess_all[i]

        # loss_df.drop(columns=['len'])
        # midat = midat.join(loss_df, on='idx', how='inner', sort=False, rsuffix='_loss')
        midat = pd.merge(midat, loss_df, on='idx', copy=False, how="inner",
            indicator=False, suffixes=[None, '_eval']) # Do not move this up!!! Need to be after grid_search

        mitas_utils.save_all_results(midat, save_stem, save_dir=eval_save_path, args=args)


def launch_predict(args):
    model = get_model(args)
    if args.load_dir:
        load_state_dict(model, load_dir=args.load_dir)

    midat = mitas_utils.load_chop_midat(args, data_name=args.predict_files,
        # avoid using the data_dir in args.json
        data_dir=args.data_dir if "data_dir" in args.kwargs else None)

    # this for for backward compatibility
    if isinstance(args.label_genre, str):
        args.label_genre = [args.label_genre]

    miset = get_dataset(midat, args)

    # get save_path
    save_stem = Path(args.predict_files[0]).resolve()
    if len(args.predict_files) == 1:
        save_stem = f'{save_stem.parent.stem}.{save_stem.stem}.predict'
    else:
        save_stem = f'{save_stem.parent.stem}.predict'

    # if args.save_dir.samefile(args.load_dir):
    if 'save_dir' in args.kwargs:
        save_path = args.save_dir / save_stem
        gwio.dict2json(vars(args), fname='args.json', fdir=save_path)
    else:
        save_path = args.load_dir / save_stem
    save_path.mkdir(parents=True, exist_ok=True)

    y_guess_all = predict(model, miset, shuffle=False, drop_last=False,
        save_dir=None, return_numpy=True)

    # perhaps should do output_finetuning here

    # save a pickle file for now (should save individual files in the future)
    for i, label_genre in enumerate(args.label_genre):
        midat[f'{label_genre}_guess'] = y_guess_all[i]

    mitas_utils.save_all_results(midat, 'predict_all', save_dir=save_path, args=args)


def launch_cross_validate(args):
    model = get_model(args)
    if args.resume and args.load_dir:
        load_state_dict(model, load_dir=args.load_dir)

    args.net_src_file = gwio.copy_text_file_to_dir(args.net_src_file, args.save_dir)
    args.run_src_file = gwio.copy_text_file_to_dir(__file__, args.save_dir)
    gwio.dict2json(vars(args), fname='args.json', fdir=args.save_dir)

    midat = mitas_utils.load_chop_midat(args)
    train_data, valid_data = train_test_split(midat, test_size=args.split_test,
                random_state=args.split_seed)

    train_set = get_dataset(train_data, args)
    valid_set = get_dataset(valid_data, args)

    # train as usual first
    callback_func = func_partial(evaluate_in_train, miset=valid_set,
        save_dir=args.save_dir, verbose=args.verbose, batch_size=args.batch_size,
        objective=args.objective, objective_direction=args.objective_direction)

    train(model, train_set, evaluate_callback=callback_func)

    # cross-validation training
    num_seqs = len(train_set)
    rand_indices = np.linspace(0, num_seqs-1, num=num_seqs, dtype=int)
    rand_indices = np.random.permutation(rand_indices)

    top_save_dir = args.save_dir
    xvalid_size = num_seqs // args.num_cvs
    xvalid_dirs = []
    for i in range(args.num_cvs):
        valid_data_xv = [train_set[rand_indices[j]] for j in
            range(i*xvalid_size, (i+1)*xvalid_size)]
        train_data_xv = [train_set[rand_indices[j]] for j in
            itertools.chain(range(0, i*xvalid_size), range((i+1)*xvalid_size, num_seqs))]

        save_dir_xv = top_save_dir / f'xvalid_{i}'
        xvalid_dirs.append(save_dir_xv)

        callback_func = func_partial(evaluate_in_train, miset=get_dataset(valid_data_xv, args, data_genre=None),
            save_dir=save_dir_xv, batch_size=args.batch_size, verbose=args.verbose,
            objective=args.objective, objective_direction=args.objective_direction)

        # a new model is created every time (probably should try to reset the model states)
        model = get_model(args)

        train(model, get_dataset(train_data_xv, args, data_genre=None),
            batch_size=args.batch_size, save_dir=save_dir_xv, evaluate_callback=callback_func)


def launch_scan_data(args):
    # a new model is created every time
    # model = get_model(args)
    # if args.resume and args.load_dir:
    #     state_dict_load(model, fdir=args.load_dir)

    args.net_src_file = gwio.copy_text_file_to_dir(args.net_src_file, args.save_dir)
    args.run_src_file = gwio.copy_text_file_to_dir(__file__, args.save_dir)
    gwio.dict2json(vars(args), fname='args.json', fdir=args.save_dir)

    train_set, valid_set = load_train_valid_dataset(args)

    scan_report = scan_data_args(args, train_set,
        valid_data = None,
        data_sizes = args.data_sizes, # [1,2,4], # 'auto',
        batch_sizes = args.batch_sizes, #[1,2,4], # 'auto',
    )


def launch_scout_args(args):
    """  """
    args.net_src_file = gwio.copy_text_file_to_dir(args.net_src_file, args.save_dir)
    args.run_src_file = gwio.copy_text_file_to_dir(__file__, args.save_dir)
    gwio.dict2json(vars(args), fname='args.json', fdir=args.save_dir)

    num_args = len(args.args_info)
    args.arg_names, args.arg_types, args.arg_values = [], [], []

    # parse args_info: arg_name,type,values...
    for _i in range(num_args):
        arg_tokens = args.args_info[_i].split(',')
        if len(arg_tokens) < 4:
            logger.critical(f'Each arg spec must have at least 4 parts: {arg_tokens}!!!')

        args.arg_names.append(arg_tokens[0])
        args.arg_types.append(arg_tokens[1].lower())
        if 'int' in args.arg_types[-1]:
            args.arg_values.append([int(_v) for _v in arg_tokens[2:]])
        elif 'float' in args.arg_types[-1]:
            args.arg_values.append([float(_v) for _v in arg_tokens[2:]])
        else:
            logger.critical(f'Unknown arg_type: {args.arg_types[-1]} for: {args.arg_names[-1]}')

    # generate the list of arg_values except for bayes_search
    if args.random_polling or args.uniform_polling:
        for _i, min_max in enumerate(args.arg_values):
            num_points = 2
            if len(min_max) < 3:
                logger.warning(f'Default num=2 will be used for {args.arg_names[_i]}: {min_max}')
            else:
                num_points = max([2, int(min_max[-1])])

            dtype = 'int' if 'int' in args.arg_types[_i] else 'float'

            if 'log' in args.arg_types[_i]:
                min_max = np.log10(min_max)

            if args.random_polling:
                args.arg_values[_i] = (np.sort(np.random.random_sample(num_points)) * (min_max[-2] - min_max[0]) + \
                    min_max[0]).astype(dtype)
            else:
                args.arg_values[_i] = np.linspace(min_max[0], min_max[-2], num_points, dtype=dtype)

            if 'log' in args.arg_types[_i]:
                args.arg_values[_i] = 10 ** args.arg_values[_i]

        # if args.arg_types[_i] in ['int', 'integer']:
        #     args.arg_values[_i] = np.linspace(min_max[0], min_max[-1], num_points, dtype=int)
        #     # min_max[0] + np.sort(np.random.random_sample(args.num_scouts)) * (min_max[-1] - min_max[0])
        # elif args.arg_types[_i] in ['float', 'float32', 'float64']:
        #     args.arg_values[_i] = np.linspace(min_max[0], min_max[-1], num_points, dtype=float)
        # elif args.arg_types[_i] in ['logint', 'log_int', 'loginteger', 'log_integer']:
        #     args.arg_values[_i] = np.logspace(math.log10(min_max[0]), math.log10(min_max[-1]), num_points, dtype=int)
        #     # np.exp(np.log(min_max[0]) + np.sort(np.random.random_sample(args.num_scouts)) * np.log(min_max[-1] / min_max[0])
        # elif args.arg_types[_i] in ['logfloat', 'log_float', 'logfloat', 'log_float']:
        #     args.arg_values[_i] = np.logspace(math.log10(min_max[0]), math.log10(min_max[-1]), num_points, dtype=float)
        # else:
        #     logger.critical(f'Unknown arg_type: {args.arg_types[_i]} for: {args.arg_names[_i]}')
    else:
        logger.info(f'Make sure that no NUM is included in the arg_values!!!')

    if args.grid_search:
        args.arg_values = [np.sort(_v) for _v in args.arg_values]
        # arg_sets = list(itertools.product(*arg_values))
    else:
        # check all arg_values should be of the same length
        if any([len(_v) - len(args.arg_values[0]) for _v in args.arg_values[1:]]):
            logger.critical(f'All arg values must be of the same length except for grid_search')

    # print(args.arg_names, args.arg_values, args.arg_types)

    train_set, valid_set = load_train_valid_dataset(args)
    # model = get_model(args)

    if args.bayes_search:
        import optuna, joblib
        logger.info(f'Only the first two values in arg_values will be used: {args.arg_values}')
        verbose = args.verbose
        args.verbose = 0
        def get_optuna_params(trial, args=args):
            params = {}
            for _i, min_max in enumerate(args.arg_values):
                _arg_name = args.arg_names[_i]
                if args.arg_types[_i] in ['int', 'integer']:
                    params[_arg_name] = trial.suggest_int(_arg_name, min_max[0], min_max[1],
                            log=False, step=min_max[2] if len(min_max) > 2 else 1)
                elif args.arg_types[_i] in ['float', 'float32', 'float64']:
                    params[_arg_name] = trial.suggest_uniform(_arg_name, min_max[0], min_max[1])
                elif args.arg_types[_i] in ['logint', 'log_int', 'loginteger', 'log_integer']:
                    params[_arg_name] = trial.suggest_int(_arg_name, min_max[0], min_max[1],
                            log=True, step=min_max[2] if len(min_max) > 2 else 1)
                elif args.arg_types[_i] in ['logfloat', 'log_float', 'logfloat', 'log_float']:
                    params[_arg_name] = trial.suggest_loguniform(_arg_name, min_max[0], min_max[1])
                else:
                    logger.critical(f'Unknown arg_type: {args.arg_types[_i]} for: {args.arg_names[_i]}')
            return params

        def optuna_objective(trial, args=args):
            params = get_optuna_params(trial)
            args_copy = misc.Struct(vars(args))
            args_copy.update(params)
            args_copy = mitas_utils.autoconfig_args(args_copy)
            args_copy.update(params)
            model = get_model(args_copy)
            objective = train(model, train_set)
            return objective

        def optuna_save_study(study, save_prefix, args=args):
            joblib.dump(study, args.save_dir / f'{save_prefix}.pkl')
            study.trials_dataframe().to_csv(args.save_dir / f'{save_prefix}.csv', index=False)
            best_trial = study.best_params
            best_trial.update({args.objective:study.best_value})
            gwio.dict2json(best_trial, args.save_dir / f'{save_prefix}.json')

        def optuna_callback(study, trial, args=args):
            print(f'current objective: {trial.value}')
            print(f'       parameters: {trial.params}')
            # neptune.log_metric('run_score', trial.value)
            # neptune.log_text('run_parameters', str(trial.params))
            # joblib.dump(study, args.save_dir / 'optuna_study_tmp.pkl')
            # study.trials_dataframe().to_csv(args.save_dir / 'optuna_study_tmp.csv', index=False)
            optuna_save_study(study, 'optuna_study')

        study = optuna.create_study(direction=args.objective_direction)
        study.optimize(optuna_objective, n_trials=args.bayes_search, callbacks=[optuna_callback])
        print(f'Best parameters: {study.best_params}')
        print(f'Best objective: {study.best_value}')
        optuna_save_study(study, 'optuna_study')
        args.verbose = verbose
        return

    best_valid_loss = np.inf
    old_arg_values = args.arg_values
    new_arg_values = args.arg_values
    master_save_dir = args.save_dir.as_posix()
    ispawn = 0
    while True:
        if ispawn == 0:
            save_dir = Path(master_save_dir)
        else:
            save_dir = Path(master_save_dir) / f'spawn_{ispawn}'

        scout_best_loss = scout_args(args, train_set, valid_set,
            arg_names = args.arg_names,
            arg_values = new_arg_values,
            grid_search = args.grid_search,
            save_dir = save_dir,
            #     arg_names = ['learning_rate', 'l2decay'],
            #     arg_values = [[1e-5, 1e-4, 1e-3, 1e-2], [1e-4, 1e-2]],
        )

        if not args.spawn_search: break #!!!! stop here uness spawn_search
        ispawn += 1

        # get arg_values giving the best valid_loss
        imin = scout_best_loss.valid_loss.argmin()
        if best_valid_loss < scout_best_loss.valid_loss[imin]:
            logger.info(f'Best spawned args found with valid_loss: {best_valid_loss}')
            break
        else:
            best_valid_loss = scout_best_loss.valid_loss[imin]

        # get the arg values giving the best loss
        arg_values_best = scout_best_loss.loc[imin, args.arg_names].to_numpy()
        # mutate from the best arg_values
        new_arg_values = []
        for _i in range(args.num_spawns):
            new_arg_values.append(arg_values_best.copy())

            iarg = np.random.randint(low=0, high=num_args)
            # make a random change, better stay within the confines of the input
            if args.arg_types[iarg] == 0: # linear
                new_arg_values[-1][iarg] = old_arg_values[iarg][0] + \
                    np.random.rand() * (old_arg_values[iarg][-1] - old_arg_values[iarg][0])
            else:
                new_arg_values[-1][iarg] = np.exp(np.log(old_arg_values[iarg][0]) + \
                    np.random.rand() * np.log(old_arg_values[iarg][-1] / old_arg_values[iarg][0]))

        # get ready for the next scout run
        new_arg_values = list(zip(*new_arg_values))


def launch_average_model(args):
    """ call evaluate() """
    # first get the load_dirs and parse valid_loss from dir. name if possible
    load_dirs = []
    valid_loss_models = [] # np.ones((num_models,), dtype=np.float32)
    for imodel, load_dir in enumerate(args.model_dirs):
        logger.debug(f'Check model_dir #{imodel}: {load_dir} ...')

        load_dir = Path(load_dir)
        if not load_dir.exists():
            logger.error(f'Cannot find model_dir #{imodel}: {load_dir} !!!')
            continue

        if args.best_chkpt or args.best_save:
            net_state_dirs = []
            if args.best_chkpt: net_state_dirs +=  list(load_dir.glob('chkpt_epo*'))
            if args.best_save: net_state_dirs +=  list(load_dir.glob('save_epo*'))
            if len(net_state_dirs) == 0:
                logger.error(f'Failed to find any chkpt_* or save_* directories in: {load_dir} !!!')
                continue
            # directory name contains the validation loss
            epo_values = np.array([int(_dir.name.split('_')[-2][3:]) for _dir in net_state_dirs], dtype=np.int)
            loss_values = np.array([float(_dir.name.split('_')[-1]) for _dir in net_state_dirs], dtype=np.float32)
            if args.by_epoch:
                idx_best = epo_values.argmax()
            else:
                idx_best = loss_values.argmin()
            load_dirs.append(net_state_dirs[idx_best])
            valid_loss_models.append(loss_values[idx_best])
        else:
            load_dirs.append(load_dir)
            valid_loss_models.append(misc.str2float(load_dir.name.split('_')[-1], default=1.0))

        # check for args.json
        if not (load_dirs[-1] / 'args.json').exists():
            logger.error(f'Cannot find args.json in load_dir: {load_dirs[-1]}!!!')
            load_dirs.pop(-1)
            valid_loss_models.pop(-1)
        else:
            logger.info(f'Found load_dir: {load_dirs[-1]} with valid loss: {valid_loss_models[-1]}')

    if len(load_dirs) == 0:
        logger.critical('Load_dirs are empty!!!')
        return

    valid_loss_models = np.array(valid_loss_models, dtype=np.float32)

    # load data using the current command line args
    midat = mitas_utils.load_chop_midat(args, data_name=args.data_files,
            data_dir=args.data_dir if "data_dir" in args.kwargs else None)
    miset = get_dataset(midat, args)

    num_seqs = len(miset)
    num_models = len(load_dirs)

    y_guess_models = []
    loss_df_models = []

    if 'save_dir' not in args.kwargs:
        args.save_dir = Path('model_averages')
        args.save_dir.mkdir(parents=True, exist_ok=True)

    logger_level = logger.root.level
    logger.debug(f'Averaging models in directories:{load_dirs} ...')
    for imodel, load_dir in enumerate(load_dirs):
        logger.info(f'Loading model #{imodel} from: {str(load_dir)} ...')

        logger.setLevel(logging.WARNING)
        # create and restore the model
        model_args, _ = mitas_utils.parse_args(['train', '-v', '0'])
        model_args.update(gwio.json2dict(fname='args.json', fdir=load_dir))
        model_args.update(args.kwargs) # kwargs overwrite the model args!!!
        model_args.load_dir = Path(load_dir)
        model = get_model(model_args)
        load_state_dict(model, load_dir=load_dir)

        loss_df, y_guess = evaluate(model, miset, shuffle=False, batch_size=args.batch_size,
                    save_dir=None, drop_last=False, return_guess=True, return_numpy=True)
        y_guess_models.append(y_guess)
        loss_df_models.append(loss_df)

        logger.setLevel(logger_level)

        # show results of the loss_df?
        logger.info(f'Model #{imodel} loss:')
        print(loss_df.mean())
        # as_label_models.append(model.loss_fn[0].as_label(y_output).numpy())
        # loss_vs_seq_models[imodel] = loss_model['loss'].to_numpy()
        # std_vs_seq_models[imodel] = loss_model['loss_std'].to_numpy()
        # compute loss
        # if np.array_equal(as_label_models[-1][0].shape, y_truth[0].shape):
        #     loss_vs_seq, std_vs_seq = compute_loss(model.loss_fn, y_output, y_truth,
        #                 batch_size=args.batch_size, seqs_len=seqs_len, shuffle=False,
        #                 loss_padding=args.loss_padding, loss_sqrt=args.loss_sqrt)
        #     logger.info(f'Model: {imodel} loss: \033[0;46m{loss_vs_seq.mean():6.4f}\033[0m' +
        #                 f' std: {std_vs_seq.mean():6.4f}')
        #     loss_vs_seq_models[imodel] = loss_vs_seq
        #     std_vs_seq_models[imodel] = std_vs_seq
        # else:
        #     loss_vs_seq_models[imodel] = 1
        #     std_vs_seq_models[imodel] = 1


    seqs_len = midat['len'].to_numpy()
    y_truth = [miset[i][-1] for i in range(num_seqs)]

    if args.model_weights in ['loss']:
        model_weights = 1.0 / valid_loss_models
        model_weights = model_weights / model_weights.sum()
    elif args.model_weights in ['none', 'no', 'const', 'constant']:
        model_weights = np.ones((num_models)) / num_models
    else:
        logger.error(f'Do not yet support model weights: {model_weights} !!!')
        model_weights = None

    logger.info(f'Averaging models with weights: {model_weights}...')

    # zip to get each item to be a list of model predictions for one sequence
    y_guess_models = zip(*y_guess_models)
    # y_output_models = zip(*as_label_models)
    # stack to get one numpy array for each seq, then get the averaged model predictions!
    if num_seqs > 23:
        y_guess_models = misc.mpi_map(np.stack, y_guess_models)
        y_guess_aver = misc.mpi_map(func_partial(np.average, axis=0, weights=model_weights),
                                y_guess_models)
    else:
        y_guess_models = list(map(np.stack, y_guess_models))
        y_guess_aver = list(map(func_partial(np.average, axis=0, weights=model_weights),
                                y_guess_models))

    if model.metric_fn is not None and np.array_equal(y_guess_aver[0].shape, y_truth[0].shape):
        logger.info(f'Computing metrics for the averaged prediction...')
        metrics_aver = [model.metric_fn[0](y_guess_aver[i], y_truth[i], batch=False) for i in range(num_seqs)]
        metrics_aver = pd.DataFrame(dict(zip(model.metric_fn[0].labels, list(zip(*metrics_aver)))))
        print(metrics_aver.mean())
        # loss_model_aver, std_model_aver = compute_loss( #model.loss_fn
        #             SeqLossFn_P2P(F.mse_loss, name='mse', reduction='none'),
        #             y_model_aver, y_truth, batch_size=args.batch_size, seqs_len=seqs_len,
        #             shuffle=False, loss_mask=args.loss_mask, loss_sqrt=args.loss_sqrt)
        # logger.info(f'Model losses: {loss_vs_seq_models.mean(axis=1)}')
        # logger.info(f'Model averaged loss: \033[0;46m{loss_vs_seq_models.mean():6.4f}\033[0m' +
        #             f' std: {std_vs_seq_models.mean():6.4f}')
        # logger.info(f'Averaged model loss: \033[0;46m{loss_model_aver.mean():6.4f}\033[0m' +
        #             f' std: {std_model_aver.mean():6.4f}')

    # save the average model results
    logger.info(f'Saving the averaged model estimates in: {str(args.save_dir)}')
    midat['y_model'] = y_guess_aver
    mitas_utils.save_all_results(midat, f'averaged{num_models}', save_dir=args.save_dir, args=args)

    mitas_utils.save_predict_matrix(y_guess_aver, args.save_dir, seqs_len=seqs_len, names=None)


def launch_view(args):
    model = get_model(args)

    args.net_src_file = gwio.copy_text_file_to_dir(args.net_src_file, args.save_dir)
    args.run_src_file = gwio.copy_text_file_to_dir(__file__, args.save_dir)
    if not (args.save_dir / 'args.json').exists():
        gwio.dict2json(vars(args), fname='args.json', fdir=args.save_dir)

    model.net.train()
    logger.info(f'Saving model diagram to {args.save_dir / "model_train"} ...')
    mi.jit.save(model.net, (args.save_dir / 'model_train').as_posix(),
                input_spec=model.net.Embed.in_shapes)
    model.net.eval()
    logger.info(f'Saving model diagram to {args.save_dir / "model_eval"} ...')
    mi.jit.save(model.net, (args.save_dir / 'model_eval').as_posix(),
                input_spec=model.net.Embed.in_shapes)
        # input_spec=[InputSpec(shape=[2,512, 15], dtype='float32', name='x'),
                    # InputSpec(shape=[2, 1], dtype='int32', name='seqs_len')])
    if not args.to_static:
        mi.disable_static()


def launch(args):
    """  """
    sys.setrecursionlimit(int(1e4))

    if args.fleet:
        strategy = fleet.DistributedStrategy()
        fleet.init(is_collective=True, strategy=strategy)

    if args.to_static:
        mi.enable_static()
    elif not mi.in_dynamic_mode():
        mi.disable_static()

    logger.info(f'Launching missions: {args.mission}...')
    logger.info(f'Setting terminal title ... \033]0;{args.mission}@{os.uname().nodename}::{Path(args.home_dir).stem}\a')

    if 'summary' in args.mission or 'summarize' in args.mission or 'view' in args.mission:
        launch_view(args)

    if 'train' in args.mission:
        launch_train(args)

    if 'cross_validate' in args.mission:
        launch_cross_validate(args)

    if 'rename' in args.mission:
        # simply rename the load_dir to save_dir
        if args.load_dir is None or not args.load_dir.exists():
            logger.critical(f'Unable to rename load_dir: {args.load_dir}, cannot find it!!!!')
            return
        if args.load_dir.samefile(args.save_dir):
            logger.warning(f'Same load_dir: {args.load_dir} and save_dir: {args.save_dir}')

        logger.info(f'Renaming load_dir: {args.load_dir} to: {args.save_dir}')
        shutil.rmtree(args.save_dir)
        args.load_dir.replace(args.save_dir)

    if 'evaluate' in args.mission:
        launch_evaluate(args)

    if 'predict' in args.mission:
        launch_predict(args)

    if 'scan_data' in args.mission:
        launch_scan_data(args)

    if 'scout_args' in args.mission:
        launch_scout_args(args)

    if 'average_model' in args.mission:
        launch_average_model(args)

    logger.info(f'All missions accomplished! (^_^)')


def get_args(argv):
    """ return default, user-provided (via argv), and various config args """
    # The order of args determination is the following:
    #   1) misc.parse_args() gives defaults and kwargs (the commandline args)
    #   2) config.json (if exists in current directory)
    #   3) args.config file (if passed),
    #   4) parsed kwargs again
    #   5) misc.autoconfig_args() will auto-set values
    #      for any args with unset values not yet set.
    #   6) kwargs again
    #   7) args.json in args.load_dir will overwrite all values
    #   8) kwargs will overwrite everything at the end
    #   9) args.save_dir

    # step #1: initialize the args structure
    args, kwargs = mitas_utils.parse_args(argv)
    misc.logging_config(logging, logfile=args.log, lineno=True, level=args.verbose)

    logger.info(f"====================== NEW RUN =======================")
    logger.info(f'argv: {args.argv}')
    args.net_src_file = MiNets.__file__

    # step #2: check local configuration json
    default_config = Path('config.json') # path.cwd() / 'config.json'
    if default_config.exists():
        logger.info(f'Loading local kwargs: {default_config}')
        config_dict = gwio.json2dict(default_config)
        # print(json.dumps(args_local, indent=4))
        if logger.root.level <= logging.DEBUG:
            logger.debug(gwio.json_str(config_dict))
        args.update(config_dict)

    # step #3: check whether args.config is passed
    if args.config is not None and Path(args.config).exists():
        logger.info(f'Loading config json: {args.config}')
        args.update(gwio.json2dict(args.config))

    # step #4:
    logger.info(f'Applying argv kwargs:')
    args.update(kwargs)
    logger.info(gwio.json_str(kwargs))

    # step #5 and #6:
    args = mitas_utils.autoconfig_args(args)
    args.update(kwargs) # reapply...

    # step #7 and (#8)
    # load args.load_dir/args.json if exists
    if args.load_dir: # load args.json
        logger.info(f'Loading model args from directory: {args.load_dir}')
        args.load_dir = Path(args.load_dir)
        if not args.load_dir.exists():
            logger.critical(f'Model directory: {args.load_dir} does not exist, fail to load!')
            logger.critical(f'Use --save_dir {args.load_dir} if intended for saving...')
            sys.exit(1)

        if (args.load_dir / 'args.json').exists():
            logger.info('Updating args with json: ' + \
                        misc.str_color(args.load_dir / "args.json", style='reverse'))
            args.update(gwio.json2dict(fname='args.json', fdir=args.load_dir))
            args.update(kwargs) # kwargs overwrite the model args!!!
        else:
            logger.warning(f'args.json not found in {args.load_dir}, default args used!!')

    # step #9: set up save_dir if unset
    if args.save_dir in [None, '', 'auto']:
        if args.load_dir and 'save_dir' not in kwargs:
            args.save_dir = args.load_dir
        else:
            max_layers = max([
                # misc.get_1st_value([args.init_num, 0]), # init_num default: 2
                misc.get_1st_value([args.linear_num, 0]),
                misc.get_1st_value([args.conv1d_num, 0]),
                misc.get_1st_value([args.conv2d_num, 0]),
                misc.get_1st_value([args.attn_num, 0]),
                misc.get_1st_value([args.lstm_num, 0]),
                misc.get_1st_value([args.return_num, 0]),
                ])
            max_channels = max(misc.unpack_list_tuple([
                misc.get_1st_value([args.embed_dim, 0]),
                misc.get_1st_value([args.init_dim, 0]),
                misc.get_1st_value([args.linear_dim, 0]),
                misc.get_1st_value([args.conv1d_dim, 0]),
                misc.get_1st_value([args.conv2d_dim, 0]),
                misc.get_1st_value([args.lstm_dim, 0]),
                misc.get_1st_value([args.return_dim, 0]),
                ]))

            net_label = args.net # .replace('_', '-')
            loss_fn_label = '_'.join(args.loss_fn).replace('+', '-') if args.__dict__.get('scheduler', None) is None else 'scheduler'

            # args.json is re-loaded later in some missions
            if 'average_model' in args.mission:
                args.save_dir = '.'.join([
                    Path(args.data_dir).stem,
                    Path(args.data_name).stem,
                    f'model-average-{len(args.model_dirs)}p',
                    ])
            elif 'scout_args' in args.mission:
                args.save_dir = '.'.join([
                    Path(args.data_dir).stem,
                    Path(args.data_name).stem,
                    net_label, # + f'_l{max_layers}c{max_channels}',
                    # args.input_genre,
                    loss_fn_label,
                    f'scout-args-{len(args.args_info)}p',
                    ])
            elif args.job_genre in ['net', 'model']:
                args.save_dir = '.'.join([
                    Path(args.data_dir).stem,
                    Path(args.data_name).stem,
                    f'{net_label}_l{max_layers}c{max_channels}',
                    # args.input_genre,
                    loss_fn_label,
                    ])
            elif args.job_genre in ['data']:
                args.save_dir = '.'.join([
                    Path(args.data_dir).stem,
                    Path(args.data_name).stem,
                    # args.net,
                    f'l{max_layers}c{max_channels}',
                    # args.input_genre,
                    loss_fn_label,
                    ])
            else: # args.job_genre in [None, 'all']:
                args.save_dir = '.'.join([
                    Path(args.data_dir).stem,
                    Path(args.data_name).stem,
                    f'{net_label}_l{max_layers}c{max_channels}',
                    # args.input_genre,
                    loss_fn_label,
                    ])

        if args.save_dir_prefix:
            args.save_dir = f'{args.save_dir_prefix}_' + str(args.save_dir)
            # args.save_dir_suffix = ''
        if args.save_dir_suffix:
            args.save_dir = str(args.save_dir) + f'_{args.save_dir_suffix}'
            # args.save_dir_suffix = ''

        args.save_dir = gwio.next_backup_path(args.save_dir)

    logger.info(f'Results will be saved in: {misc.str_color(args.save_dir, style="reverse")}')

    # final clean-up
    if isinstance(args.data_dir, str): args.data_dir = Path(args.data_dir)
    if isinstance(args.load_dir, str): args.load_dir = Path(args.load_dir)
    if isinstance(args.save_dir, str): args.save_dir = Path(args.save_dir)
    if not args.save_dir.exists(): args.save_dir.mkdir(parents=True, exist_ok=True)

    # generate new logger file
    args.kwargs = kwargs
    if args.net_id is None: args.net_id = f'{args.net}_{datetime.now().strftime("%Y-%m-%d-%H-%M-%S")}'

    args.log = args.save_dir / ((args.mission if isinstance(args.mission, str) else '-'.join(args.mission)) + \
                f'_{datetime.now().strftime("%Y-%m-%d-%H-%M-%S")}.log')
    misc.logger_setlogfile(logger, logfile=args.log)

    # deal with the random seed
    if args.random_seed is None:
        # np.random.seed(None)
        random_state = np.random.get_state()
        args.random_seed = random_state[1][random_state[2] % len(random_state[1])]
        logger.info(f'Storing new random generator seed: {args.random_seed} ...')

    # maybe only do this for evaluate and predict?
    # seed_mitas(args.random_seed)

    # gwio.dict2json(vars(args), fname='last.json', fdir=Path.cwd())
    return args

def seed_mitas(seed=1234567):
    logger.info(f'Seeding random generators with: {seed} ...')
    mi.seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    # torch.manual_seed(seed)
    # torch.cuda.manual_seed(seed)
    # torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.
    # torch.backends.cudnn.benchmark = False
    # torch.backends.cudnn.deterministic = True


if __name__ == '__main__' :
    """  """
    # if sys.stdin.isatty():     # if running from terminal
    # else: # run from vscode # os.chdir()
        # args, argv_dict = parse_args(['-h'])
    args = get_args(sys.argv[1:])

    if 'device' in args.kwargs and args.device is not None:
        mi.device.set_device(args.device)

    if args.visual_dl:
        from visualdl import LogWriter

    # distributed training
    if args.spawn:
        import paddle.distributed as mi_distro
        mi_distro.init_parallel_env()
    elif args.fleet:
        from paddle.distributed import fleet

    if mi.get_device() == 'cpu':
        mi.set_flags({
            'FLAGS_paddle_num_threads': int(os.cpu_count() * 0.8),
        })

    if os.uname().nodename in ['udesk-dna'] or os.uname().nodename.startswith('gpu'):
        mi.set_flags({
            'FLAGS_eager_delete_tensor_gb': 0.0,
            'FLAGS_fast_eager_deletion_mode': True,
            'FLAGS_eager_delete_scope': True,
            'FLAGS_allocator_strategy': 'naive_best_fit',
            # 'FLAGS_fraction_of_gpu_memory_to_use': 0,
        })
    else:
        mi.set_flags({
            'FLAGS_eager_delete_tensor_gb': 1.0,
            'FLAGS_fast_eager_deletion_mode': True,
            'FLAGS_eager_delete_scope': True,
            'FLAGS_allocator_strategy': 'naive_best_fit',
            # 'FLAGS_fraction_of_gpu_memory_to_use': 0,
        })

    launch(args)
