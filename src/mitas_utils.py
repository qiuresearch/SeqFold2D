#!/usr/bin/env python
import os
import argparse
import itertools
import functools
import math
import numpy as np
from datetime import datetime
from pathlib import Path
from matplotlib import pyplot as plt
import optuna

import logging
logger = logging.getLogger(__name__)

# homebrew
import misc
import molstru
import brew_midat

def parse_args(*argv):
    r"""Parse arguments (in the form of sys.argv[1:]) and/or default args

    Arguments:
        argv (str/list, required): the command line options after fly_paddle.py.
            A subparser must be given!
    """
    if isinstance(argv, str): argv = [argv]
    argv = misc.unpack_list_tuple(argv)

    flex_var = functools.partial(misc.parser_flex_var, argv=argv)
    formatter = misc.parser_formatter()

    parser = argparse.ArgumentParser(description='''
        Portal for paddling to the future''',
        formatter_class=formatter)#, argparse.RawTextHelpFormatter)

    # ======= if its value will be set by autoconfig_args(), it shoud be set to None as default =======

    # the parent parser inherited by all subparsers
    paparser = argparse.ArgumentParser(add_help=False, formatter_class=formatter)
    paparser.add_argument(*flex_var('-argv'), metavar='', type=str, default='-h', help='string of argv from command line (auto-defined)')
    paparser.add_argument(*flex_var('-kwargs'), metavar='', type=dict, default=dict(), help='dict of argv from command line (auto_defined)')
    paparser.add_argument(*flex_var('-verbose'), metavar='', choices=[0,1,2], default=1, type=int, help="('_')")
    paparser.add_argument(*flex_var('-job_genre'), metavar='', type=str, default=None, help='only used for auto-generating directory names')
    paparser.add_argument(*flex_var('-objective'), metavar='', type=str, default='loss', help='metric name for saving checkpoints (loss will be used if no metrics)')
    paparser.add_argument(*flex_var('-objective_direction'), metavar='', type=str, default='minimize', choices=['minimize', 'maximize'], help='whether the metric should go up or down)')

    paparser.add_argument(*flex_var('-to_static'), metavar='', type=misc.str2bool, nargs='?', const=True, default=False, help='turn on static mode')
    paparser.add_argument(*flex_var('-device'), metavar='', type=str, nargs='?', const=None, default=None, help='set device: cpu, gpu, gpu:x')
    paparser.add_argument(*flex_var('-spawn'), metavar='', type=misc.str2bool, nargs='?', const=True, default=False, help='distributed training')
    paparser.add_argument(*flex_var('-fleet'), metavar='', type=misc.str2bool, nargs='?', const=True, default=False, help='distributed training (multiple save_dirs!)')

    paparser.add_argument(*flex_var('-host'), metavar='', type=str,  default=f'{os.uname().nodename}({os.uname().sysname})', help='auto generated worker info')
    paparser.add_argument(*flex_var('-home_dir'), metavar='', type=str, default=os.getcwd(), help='auto-generated current directory')
    paparser.add_argument(*flex_var('-run_src_file'), metavar='', type=str, default=None, help="the main run code")
    paparser.add_argument(*flex_var('-net_src_file'), type=str, default=None, metavar='', help="python code for net definitions")
    paparser.add_argument(*flex_var('-params'), type=dict, default=None, metavar='', help="the number of parameters (auto-generated)")
    paparser.add_argument(*flex_var('-log'), type=str, default='fly_paddle.log', metavar='log_file', help="the log file")

    # two ways to load args: 1) args file via -args, 2) args.json in data_dir
    paparser.add_argument(*flex_var('-config'), metavar='json_file', type=str, default=None, help='args config file to load')
    paparser.add_argument(*flex_var('-random_seed'), type=int, default=None, metavar='', help='random seed for torch/paddle/np')
    paparser.add_argument(*flex_var('-resume'), metavar='', type=misc.str2bool, nargs='?', const=True, default=True, help='set to load model state dicts')
    paparser.add_argument(*flex_var('-load_dir'), metavar='', type=str, default=None,  help="for loading model args, src file, and states")
    paparser.add_argument(*flex_var('-save_dir'), metavar='', type=str, nargs='?', const=None, default=None, help="for saving model and results")
    paparser.add_argument(*flex_var('-save_dir_prefix'), metavar='', type=str, nargs='?', const='', default=None, help="prefix to add to save_dir")
    paparser.add_argument(*flex_var('-save_dir_suffix'), metavar='', type=str, nargs='?', const='', default=None, help="suffix to add to save_dir")
    paparser.add_argument(*flex_var('-save_level'), type=int, default=2, metavar='', help="0: no save, 1: final only, 2: all interim")
    paparser.add_argument(*flex_var('-save_groupby'), metavar='', type=str, nargs='+', default=['epoch', 'batch'], help="groupby columns, e.g., batch, epoch")
    # paparser.add_argument(*flex_var('-log_sep'), type=str, default="="*7, metavar='', help='divider string for logger')
    # -{datetime.now().strftime("%b%d-%H-%M-%S")}
    # data
    paparser.add_argument(*flex_var('-data_args'), type=str, default='======= data args =======', metavar='', help="======= data args =======")
    paparser.add_argument(*flex_var('-data_dir'), type=str, default='./', metavar='', help="data directory")
    paparser.add_argument(*flex_var('-data_name'), type=str, default=None, metavar='', help="data file name (split into train/eval if training and no eval_name)")
    paparser.add_argument(*flex_var('-eval_name'), type=str, default=None, metavar='', help="eval file name")
    # paparser.add_argument(*flex_var('-data_suffix'), type=str, default='.pkl', metavar='', help="data file suffix")
    # paparser.add_argument(*flex_var('-eval_dir'), type=str, default=None, metavar='', help="data file name")

    # for data loading and baking
    paparser.add_argument(*flex_var('-data_genre'), type=str, default='contarna', metavar='', help="predefined data type: contarna, ab2021, etc.")
    paparser.add_argument(*flex_var('-data_len'), type=int, nargs='+', default=[0], metavar='', help="[[min, max], pad_flag(0: nothing; -1: max_len; max_len)]")
    paparser.add_argument(*flex_var('-data_include'), type=str, nargs='+', default=None, metavar='', help="col_name col_values...")
    paparser.add_argument(*flex_var('-data_exclude'), type=str, nargs='+', default=None, metavar='', help="col_name col_values...")
    paparser.add_argument(*flex_var('-data_range'), type=str, nargs='+', default=None, metavar='', help="lenCT 4 1000")
    paparser.add_argument(*flex_var('-data_ratio'), type=str, nargs='+', default=None, metavar='', help="lenCT len 0.04 1")
    paparser.add_argument(*flex_var('-data_size'), type=int, default=None, metavar='', help="the number of data to use if > 0")

    paparser.add_argument(*flex_var('-split_test'), type=float, default=0.15, metavar='', help="for train_test_split()")
    paparser.add_argument(*flex_var('-split_seed'), type=int, default=1001, metavar='', help="for train_test_split()")
    paparser.add_argument(*flex_var('-split_shuffle'), type=misc.str2bool, metavar='', nargs='?', const=True, default=False, help='shuffle for train_test_split')
    paparser.add_argument(*flex_var('-split_stratify'), type=str, default=None, metavar='', help='dataframe column or dict key for stratified split')
    paparser.add_argument(*flex_var('-split_bucket_key'), type=str, default=None, metavar='', help='bucketize column for stratified split')
    paparser.add_argument(*flex_var('-split_bucket_num'), type=int, default=11, metavar='', help='bucketize column for stratified split')

    # for input data
    paparser.add_argument(*flex_var('-input_genre'), type=str, nargs='+', default=['seq2onehot'], metavar='', help="seq2onehot+bpmat+quant+...")
    paparser.add_argument(*flex_var('-input_fmt'), type=str, default=None, metavar='', help="for the first input only! N: batch_size, L: seq length, C: # of channels/features")
    paparser.add_argument(*flex_var('-input_dim'), type=int, default=None, metavar='', help="for the first input only! the dimension of features")
    paparser.add_argument(*flex_var('-feats_nn'), type=int, default=0, metavar='', help="# of nearest neighbors to use")
    paparser.add_argument(*flex_var('-feats_dbn'), type=misc.str2bool, metavar='', nargs='?', const=True, default=False, help="use feature dbn in data")
    paparser.add_argument(*flex_var('-feats_attr'), type=misc.str2bool, metavar='', nargs='?', const=True, default=False, help="use feature attribute in data")
    paparser.add_argument(*flex_var('-feats_extra'), type=misc.str2bool, metavar='', nargs='?', const=True, default=False, help="use feature extra features in data")

    # for label data
    paparser.add_argument(*flex_var('-label_genre'), metavar='', type=str, nargs='+', default=['ct'], help="label type: upp/ct/dist/...")
    paparser.add_argument(*flex_var('-label_fmt'), metavar='', type=str, default='NL', help="label fmt: NL/NLL/... (not used yet)")
    paparser.add_argument(*flex_var('-label_tone'), metavar='', type=str, default='none', help="label tone: none/hard/soft")
    paparser.add_argument(*flex_var('-label_ntype'), metavar='', type=int, default=None, help="number of label types (not used yet)")
    paparser.add_argument(*flex_var('-label_min_delta_ij'), metavar='', type=int, default=1, help="minimum delta_ij for ct")
    paparser.add_argument(*flex_var('-label_min_stem_len'), metavar='', type=int, default=1, help="minimum stem length for ct")
    paparser.add_argument(*flex_var('-label_soft2hard'), metavar='', type=misc.str2bool, nargs='?', const=True, default=False, help='convert label from soft to hard')
    paparser.add_argument(*flex_var('-label_hard2soft'), metavar='', type=misc.str2bool, nargs='?', const=True, default=False, help='convert label from hard to soft')
    paparser.add_argument(*flex_var('-label_smooth'), metavar='', type=float, default=0.0, help="coefficient of label smoothing")
    # additional labels -- None at this point

    # for output used by evaluate and predict
    paparser.add_argument(*flex_var('-post_process'), type=str, nargs='+', default=None, metavar='', help="for evaluate and predict only (canonical/noncanonical)")
    paparser.add_argument(*flex_var('-output_genre'), type=str, nargs='+', default=None, metavar='', help="output genre (upp/ct/dist/ppm/bpseq), defaulted to label_genre")
    paparser.add_argument(*flex_var('-output_finetune'), type=str, nargs='+', default=None, metavar='', help="grid_search/etc")
    # paparser.add_argument(*flex_var('-output_analyze'), type=str, default=None, metavar='', help="TURNED OFF!!!")
    paparser.add_argument(*flex_var('-output_threshold'), type=float, default=None, metavar='', help="threshold for converting ppm to bpmat")
    # paparser.add_argument(*flex_var('-output_lumpsum'), type=misc.str2bool, nargs='?', const=True, default=True, metavar='', help="whether to save lumpsum pkl/csv/fasta")
    # paparser.add_argument(*flex_var('-output_individual'), type=misc.str2bool, nargs='?', const=True, default=False, metavar='', help="whether to save individual fasta/bpseq")

    # net
    paparser.add_argument(*flex_var('-net_args'), type=str, default='======= net args =======', metavar='', help="======= net args =======")
    paparser.add_argument(*flex_var('-net_summary'), type=misc.str2bool, metavar='', nargs='?', const=True, default=True, help="whether to show summary")
    paparser.add_argument(*flex_var('-net'), type=str, default='linear', metavar='', help="the name of the net class")
    paparser.add_argument(*flex_var('-net_id'), type=str, default=None, metavar='', help="auto-generated net id (unique?)")
    paparser.add_argument(*flex_var('-seq2mat_method'), metavar='', type=str, default='concat', help='how to convert sequence to matrix')

    # global parameters that can be overwritten by bloack-specific settings if applicable
    paparser.add_argument(*flex_var('-net_globals'), type=str, default='------- net global args (overwritable by locals)', metavar='', help="------- net gloabls")
    paparser.add_argument(*flex_var('-param_init'), type=str, default='xavieruniform', metavar='', help='global parameter initializer')
    paparser.add_argument(*flex_var('-depth'), type=int, default=0, metavar='', help="global depth (number of layers)")
    paparser.add_argument(*flex_var('-width'), type=int, default=0, metavar='', help="global width (number of channels)")
    paparser.add_argument(*flex_var('-norm_in'), type=str, metavar='', default=None, help='norm function for the input to each module')
    paparser.add_argument(*flex_var('-act_fn'), type=str, nargs='+', default=['relu', 'swish'], metavar='', help="global activation: relu/sigmoid/...")
    paparser.add_argument(*flex_var('-norm_fn'), type=str, default='none', metavar='', help="global normalization: batch/instance/layer/none")
    paparser.add_argument(*flex_var('-norm_axis'), type=int, nargs='+', default=[-1], metavar='', help="global normal axis")
    paparser.add_argument(*flex_var('-norm_mask'), type=misc.str2bool, metavar='', nargs='?', const=True, default=False, help='whether to apply mask for norm_fn')
    paparser.add_argument(*flex_var('-norm_trainable'), type=misc.str2bool, metavar='', nargs='?', const=True, default=True, help='turn on weight/bias for norm_fn')
    paparser.add_argument(*flex_var('-act_after_norm'), type=misc.str2bool, metavar='', nargs='?', const=True, default=False, help='globally do activation after norm')
    paparser.add_argument(*flex_var('-pre_act_norm'), type=misc.str2bool, metavar='', nargs='?', const=True, default=False, help='globally apply act/norm before weight/bias')
    paparser.add_argument(*flex_var('-dropout'), type=float, default=0.42, metavar='', help="global dropout (can be overwritten locally)")
    paparser.add_argument(*flex_var('-norm_out'), type=str, metavar='', default=None, help='norm function for the output from each module')

    #!!!!!!!!!!!!!!!! embedding layer parameters are NOT affected by the global parameters above !!!!!!!!!!!!!!!!
    paparser.add_argument(*flex_var('-embed_args'), type=str, default='------- embedding args (unaffected by net_globals, no dropout)', metavar='', help="------- embed args")
    # paparser.add_argument(*flex_var('-embed_num'), type=int, default=1, metavar='', help="unused")
    paparser.add_argument(*flex_var('-embed_fn'), type=str, default=None, metavar='', help="embed/linear/none")
    paparser.add_argument(*flex_var('-embed_dim'), type=int, default=None, metavar='', help="the output dim of embed")
    paparser.add_argument(*flex_var('-embed_padding_idx'), type=int, default=0, metavar='', help="the padding idx of embed")
    paparser.add_argument(*flex_var('-embed_act_fn'), type=str, default=None, metavar='', help="only when embed_fn is not embed")
    paparser.add_argument(*flex_var('-embed_norm_fn'), type=str, default=None, metavar='', help="only when embed_fn is not embed")
    paparser.add_argument(*flex_var('-embed_norm_axis'), type=int, nargs='+', default=[-1], metavar='', help="only when embed_fn is not embed")
    paparser.add_argument(*flex_var('-embed_norm_mask'), type=misc.str2bool, metavar='', nargs='?', const=True, default=False, help='masked normalization')
    paparser.add_argument(*flex_var('-embed_norm_trainable'), type=misc.str2bool, metavar='', nargs='?', const=True, default=False, help='disable affine transform for norm_fn')
    paparser.add_argument(*flex_var('-embed_pre_act_norm'), type=misc.str2bool, metavar='', nargs='?', const=True, default=False, help='apply act/norm before weight/bias')
    paparser.add_argument(*flex_var('-embed_act_after_norm'), type=misc.str2bool, metavar='', nargs='?', const=True, default=True, help='activation after norm')

    paparser.add_argument(*flex_var('-init_args'), type=str, default='------- initialization module args', metavar='', help="------- initialization args")
    paparser.add_argument(*flex_var('-init_norm_in'), type=str, default=None, metavar='', help="normalize input for the input to each module")
    paparser.add_argument(*flex_var('-init_net'), type=str, default='linear', metavar='', help="not used yet, linear is used for now")
    paparser.add_argument(*flex_var('-init_num'), type=int, default=1, metavar='', help="# of initialization blocks (default: 1)")
    paparser.add_argument(*flex_var('-init_dim'), type=int, nargs='+', default=None, metavar='', help="input hidden dimensions (default: [width, width]")
    paparser.add_argument(*flex_var('-init_act_fn'), type=str, default=None, metavar='', help="activation for input layers")
    paparser.add_argument(*flex_var('-init_norm_fn'), type=str, default=None, metavar='', help="normalization for input layers")
    paparser.add_argument(*flex_var('-init_norm_axis'), type=int, nargs='+', default=None, metavar='', help='norm. axis for input layers')
    paparser.add_argument(*flex_var('-init_norm_mask'), type=misc.str2bool, metavar='', nargs='?', const=True, default=None, help='masked normalization?')
    paparser.add_argument(*flex_var('-init_norm_trainable'), type=misc.str2bool, metavar='', nargs='?', const=True, default=None, help='disable affine transform for norm_fn')
    paparser.add_argument(*flex_var('-init_act_after_norm'), type=misc.str2bool, metavar='', nargs='?', const=True, default=None, help='activation after norm')
    paparser.add_argument(*flex_var('-init_pre_act_norm'), type=misc.str2bool, metavar='', nargs='?', const=True, default=None, help='apply act/norm before weight/bias')
    paparser.add_argument(*flex_var('-init_dropout'), type=float, default=None, metavar='', help="dropout for input layers [args.dropout/3])")
    paparser.add_argument(*flex_var('-init_resnet'), type=misc.str2bool, metavar='', nargs='?', const=True, default=False, help="whether to use residual net (should never set it!)")
    paparser.add_argument(*flex_var('-init_resnet_beta'), type=float, default=1.0, metavar='', help='beta for resnet')
    paparser.add_argument(*flex_var('-init_norm_out'), type=str, default=None, metavar='', help="normalize output for the output to the module")

    paparser.add_argument(*flex_var('-linear_args'), type=str, default='------- linear tower/module args', metavar='', help="------- linear tower/module args")
    paparser.add_argument(*flex_var('-linear_norm_in'), type=str, default=None, metavar='', help="normalize input for the input to the module")
    paparser.add_argument(*flex_var('-linear_num'), type=int, default=None, metavar='', help="# of linear blocks")
    paparser.add_argument(*flex_var('-linear_dim'), type=int, nargs='+', default=None, metavar='', help="dims of linear layers")
    paparser.add_argument(*flex_var('-linear_act_fn'), type=str, default=None, metavar='', help="activation for linear layers")
    paparser.add_argument(*flex_var('-linear_norm_fn'), type=str, default=None, metavar='', help="normalization for linear layers")
    paparser.add_argument(*flex_var('-linear_norm_axis'), type=int, nargs='+', default=None, metavar='', help='norm. axis for linear layers')
    paparser.add_argument(*flex_var('-linear_norm_mask'), type=misc.str2bool, metavar='', nargs='?', const=True, default=None, help='masked normalization?')
    paparser.add_argument(*flex_var('-linear_norm_trainable'), type=misc.str2bool, metavar='', nargs='?', const=True, default=None, help='disable affine transform for norm_fn')
    paparser.add_argument(*flex_var('-linear_act_after_norm'), type=misc.str2bool, metavar='', nargs='?', const=True, default=None, help='activation after norm')
    paparser.add_argument(*flex_var('-linear_pre_act_norm'), type=misc.str2bool, metavar='', nargs='?', const=True, default=None, help='apply act/norm before weight/bias')
    paparser.add_argument(*flex_var('-linear_dropout'), type=float, default=None, metavar='', help="dropout for linear layers [args.dropout]")
    paparser.add_argument(*flex_var('-linear_resnet'), type=misc.str2bool, metavar='', nargs='?', const=True, default=False, help="whether to use residual net")
    paparser.add_argument(*flex_var('-linear_resnet_beta'), type=float, default=1.0, metavar='', help='beta for resnet')
    paparser.add_argument(*flex_var('-linear_norm_out'), type=str, default=None, metavar='', help="normalize output for the output to the module")

    paparser.add_argument(*flex_var('-lstm_args'), type=str, default='------- lstm tower/module args', metavar='', help="------- lstm tower/module args")
    paparser.add_argument(*flex_var('-lstm_norm_in'), type=str, default=None, metavar='', help="normalize input for the input to the module")
    paparser.add_argument(*flex_var('-lstm_num'), type=int, default=None, metavar='', help="# of LSTM blocks")
    paparser.add_argument(*flex_var('-lstm_dim'), type=int, nargs='+', default=None, metavar='', help="dims of LSTM layers")
    paparser.add_argument(*flex_var('-lstm_train_initial'), type=misc.str2bool, metavar='', nargs='?', const=True, default=False, help='whether to train initial states')
    paparser.add_argument(*flex_var('-lstm_direction'), type=str, default='bidirectional', metavar='', help="direction of LSTM layers")
    paparser.add_argument(*flex_var('-lstm_act_fn'), type=str, default=None, metavar='', help="activation in between LSTM layers")
    paparser.add_argument(*flex_var('-lstm_norm_fn'), type=str, default=None, metavar='', help="normalization for LSTM layers")
    paparser.add_argument(*flex_var('-lstm_norm_axis'), type=int, nargs='+', default=None, metavar='', help='norm. axis for LSTM layers')
    paparser.add_argument(*flex_var('-lstm_norm_mask'), type=int, nargs='+', default=None, metavar='', help='norm. mask for LSTM layers')
    paparser.add_argument(*flex_var('-lstm_norm_trainable'), type=misc.str2bool, metavar='', nargs='?', const=True, default=None, help='disable affine transform for norm_fn')
    paparser.add_argument(*flex_var('-lstm_act_after_norm'), type=misc.str2bool, metavar='', nargs='?', const=True, default=None, help='activation after norm')
    paparser.add_argument(*flex_var('-lstm_pre_act_norm'), type=misc.str2bool, metavar='', nargs='?', const=True, default=None, help='apply act/norm before weight/bias')
    paparser.add_argument(*flex_var('-lstm_dropout'), type=float, default=None, metavar='', help="dropout for LSTM layers [args.dropout/2]")
    paparser.add_argument(*flex_var('-lstm_resnet'), type=misc.str2bool, metavar='', nargs='?', const=True, default=False, help="whether to use residual net")
    paparser.add_argument(*flex_var('-lstm_resnet_beta'), type=float, default=1.0, metavar='', help='beta for resnet')
    paparser.add_argument(*flex_var('-lstm_norm_out'), type=str, default=None, metavar='', help="normalize output for the output to the module")

    paparser.add_argument(*flex_var('-conv1d_args'), type=str, default='------- conv1d tower/module args', metavar='', help="------- conv1d tower/module args")
    paparser.add_argument(*flex_var('-conv1d_norm_in'), type=str, default=None, metavar='', help="normalize input for the input to the module")
    paparser.add_argument(*flex_var('-conv1d_num'), type=int, default=None, metavar='', help="# of Conv1D blocks")
    paparser.add_argument(*flex_var('-conv1d_dim'), type=int, nargs='+', default=None, metavar='', help="channels of Conv1D layers")
    paparser.add_argument(*flex_var('-conv1d_kernel'), type=int, nargs='+', default=[5, 3], metavar='', help="kernal size in 1D convolution")
    paparser.add_argument(*flex_var('-conv1d_stride'), type=int, nargs='+', default=[1], metavar='', help="stride in 1D convolution")
    paparser.add_argument(*flex_var('-conv1d_dilation'), type=int, nargs='+', default=[1], metavar='', help="dilation in 1D convolution")
    paparser.add_argument(*flex_var('-conv1d_padding'), type=int, nargs='+', default=None, metavar='', help="padding in 1D convolution")
    paparser.add_argument(*flex_var('-conv1d_act_fn'), type=str, default=None, metavar='', help="activation for conv1d layers")
    paparser.add_argument(*flex_var('-conv1d_norm_fn'), type=str, default=None, metavar='', help="normalization for conv1d layers")
    paparser.add_argument(*flex_var('-conv1d_norm_axis'), type=int, nargs='+', default=None, metavar='', help='norm. axis for conv1d layers')
    paparser.add_argument(*flex_var('-conv1d_norm_mask'), type=misc.str2bool, metavar='', nargs='?', const=True, default=None, help='masked normalization?')
    paparser.add_argument(*flex_var('-conv1d_norm_trainable'), type=misc.str2bool, metavar='', nargs='?', const=True, default=None, help='disable affine transform for norm_fn')
    paparser.add_argument(*flex_var('-conv1d_act_after_norm'), type=misc.str2bool, metavar='', nargs='?', const=True, default=None, help='activation after norm')
    paparser.add_argument(*flex_var('-conv1d_pre_act_norm'), type=misc.str2bool, metavar='', nargs='?', const=True, default=None, help='apply act/norm before weight/bias')
    paparser.add_argument(*flex_var('-conv1d_dropout'), type=float, default=None, metavar='', help="dropout for conv1d layers [args.dropout/2]")
    paparser.add_argument(*flex_var('-conv1d_resnet'), type=misc.str2bool, metavar='', nargs='?', const=True, default=False, help="whether to use residual net")
    paparser.add_argument(*flex_var('-conv1d_resnet_beta'), type=float, default=1.0, metavar='', help='beta for resnet')
    paparser.add_argument(*flex_var('-conv1d_norm_out'), type=str, default=None, metavar='', help="normalize output for the output to the module")

    paparser.add_argument(*flex_var('-conv2d_args'), type=str, default='------- conv2d tower/module args', metavar='', help="------- conv2d tower/module args")
    paparser.add_argument(*flex_var('-conv2d_norm_in'), type=str, default=None, metavar='', help="normalize input for the input to the module")
    paparser.add_argument(*flex_var('-conv2d_num'), type=int, default=None, metavar='', help="# of Conv2D blocks")
    paparser.add_argument(*flex_var('-conv2d_dim'), type=int, nargs='+', default=None, metavar='', help="channels of Conv2D layers ")
    paparser.add_argument(*flex_var('-conv2d_kernel'), type=int, nargs='+', default=[5, 3], metavar='', help="kernal size in 2D convolution")
    paparser.add_argument(*flex_var('-conv2d_stride'), type=int, nargs='+', default=[1], metavar='', help="stride in 2D convolution")
    paparser.add_argument(*flex_var('-conv2d_dilation'), type=int, nargs='+', default=[1], metavar='', help="dilation in 2D convolution")
    paparser.add_argument(*flex_var('-conv2d_padding'), type=int, nargs='+', default=None, metavar='', help="padding in 2D convolution")
    paparser.add_argument(*flex_var('-conv2d_act_fn'), type=str, default=None, metavar='', help="activation for conv2d layers")
    paparser.add_argument(*flex_var('-conv2d_norm_fn'), type=str, default=None, metavar='', help="normalization for conv2d layers")
    paparser.add_argument(*flex_var('-conv2d_norm_axis'), type=int, nargs='+', default=None, metavar='', help='norm. axis for conv2d layers')
    paparser.add_argument(*flex_var('-conv2d_norm_mask'), type=misc.str2bool, metavar='', nargs='?', const=True, default=None, help='masked normalization?')
    paparser.add_argument(*flex_var('-conv2d_norm_trainable'), type=misc.str2bool, metavar='', nargs='?', const=True, default=None, help='disable affine transform for norm_fn')
    paparser.add_argument(*flex_var('-conv2d_act_after_norm'), type=misc.str2bool, metavar='', nargs='?', const=True, default=None, help='activation after norm')
    paparser.add_argument(*flex_var('-conv2d_pre_act_norm'), type=misc.str2bool, metavar='', nargs='?', const=True, default=None, help='apply act/norm before weight/bias')
    paparser.add_argument(*flex_var('-conv2d_dropout'), type=float, default=None, metavar='', help="dropout for conv2d layers [args.dropout/2]")
    paparser.add_argument(*flex_var('-conv2d_resnet'), type=misc.str2bool, metavar='', nargs='?', const=True, default=False, help="whether to use residual net")
    paparser.add_argument(*flex_var('-conv2d_resnet_beta'), type=float, default=1.0, metavar='', help='beta for resnet')
    paparser.add_argument(*flex_var('-conv2d_norm_out'), type=str, default=None, metavar='', help="normalize output for the output to the module")

    paparser.add_argument(*flex_var('-attn_args'), type=str, default='------- attnxd tower/module args', metavar='', help="------- attnxd tower/module args")
    paparser.add_argument(*flex_var('-attn_norm_in'), type=str, default=None, metavar='', help="normalize input for the input to the module")
    paparser.add_argument(*flex_var('-attn_posenc'), type=str, default='trig', metavar='', help='attention position encoder')
    paparser.add_argument(*flex_var('-attn_posenc_join'), type=str, default='add', metavar='', help='join method for position embedding')
    paparser.add_argument(*flex_var('-attn_posenc_dim'), type=int, default=None, metavar='', help='dimension of position embedding')
    paparser.add_argument(*flex_var('-attn_posenc_mlp_num'), type=int, default=1, metavar='', help='MLP layers for position embedding')
    paparser.add_argument(*flex_var('-attn_posenc_mlp_dim'), type=int, default=None, metavar='', help='MLP layers for position embedding')
    paparser.add_argument(*flex_var('-attn_num'), type=int, default=None, metavar='', help="# of Attention Encoder blocks")
    paparser.add_argument(*flex_var('-attn_method'), type=str, default='paddle', metavar='', help="method: paddle/dotproduct/efficient")
    paparser.add_argument(*flex_var('-attn_force_nlc'), type=misc.str2bool, metavar='', nargs='?', const=True, default=False, help='whether to force NLC shape first')
    paparser.add_argument(*flex_var('-attn_axis'), metavar='', type=int, default=[1], nargs='+', help="the axis for AxialAttention Encoders")
    paparser.add_argument(*flex_var('-attn_nhead'), type=int, default=None, metavar='', help="# of attention heads (defaut: max(2, width//16)")
    paparser.add_argument(*flex_var('-attn_temperature'), metavar='', type=str, default=None, help="cool/warm/hot/anneal (default: sqrt(dim))")
    paparser.add_argument(*flex_var('-attn_dropout'), type=float, default=None, metavar='', help="dropout before residual layer at the end of each encoder layer [args.dropout/2]")
    paparser.add_argument(*flex_var('-attn_attn_dropout'), type=float, default=None, metavar='', help="dropout of attention weights after softmax (default: args.attn_dropout/2)")
    paparser.add_argument(*flex_var('-attn_join'), type=str, default='add', metavar='', help='how to join input and attn_max: add/concat (default: add)')
    paparser.add_argument(*flex_var('-attn_ffdim'), type=int, default=None, metavar='', help="feedforward dims of Attention Encoders")
    paparser.add_argument(*flex_var('-attn_ffact_fn'), type=str, default='swish', metavar='', help="feedforward activation of each attention layer (default: swish)")
    paparser.add_argument(*flex_var('-attn_ffdropout'), type=float, default=None, metavar='', help="feedforward dropout of each attention layer  [args.dropout]")
    paparser.add_argument(*flex_var('-attn_norm_before'), type=misc.str2bool, metavar='', nargs='?', const=True, default=False, help="apply layer_norm before dropout/resnet")
    paparser.add_argument(*flex_var('-attn_act_fn'), type=str, default='softmax', metavar='', help="activation of attention weights (default: softmax)")
    paparser.add_argument(*flex_var('-attn_pre_act_norm'), type=misc.str2bool, metavar='', nargs='?', const=True, default=None, help='whether to apply act/norm beforehand')
    paparser.add_argument(*flex_var('-attn_norm_out'), type=str, default=None, metavar='', help="normalize output for the output to the module")

    paparser.add_argument(*flex_var('-return_args'), type=str, default='------- return module args', metavar='', help="------- return module args")
    paparser.add_argument(*flex_var('-return_norm_in'), type=str, default=None, metavar='', help="normalize input for the input to the module")
    paparser.add_argument(*flex_var('-return_net'), type=str, default='linear', metavar='', help="not used yet")
    paparser.add_argument(*flex_var('-return_num'), type=int, default=1, metavar='', help="# of layers in return block")
    paparser.add_argument(*flex_var('-return_dim'), type=int, nargs='+', default=None, metavar='', help="return block hidden dimensions")
    paparser.add_argument(*flex_var('-return_act_fn'), type=str, default=None, metavar='', help="activation for return layers")
    paparser.add_argument(*flex_var('-return_norm_fn'), type=str, default=None, metavar='', help="normalization for return layers")
    paparser.add_argument(*flex_var('-return_norm_axis'), type=int, nargs='+', default=None, metavar='', help='norm. axis for return layers')
    paparser.add_argument(*flex_var('-return_norm_mask'), type=misc.str2bool, metavar='', nargs='?', const=True, default=None, help='masked normalization?')
    paparser.add_argument(*flex_var('-return_norm_trainable'), type=misc.str2bool, metavar='', nargs='?', const=True, default=None, help='disable affine transform for norm_fn')
    paparser.add_argument(*flex_var('-return_act_after_norm'), type=misc.str2bool, metavar='', nargs='?', const=True, default=None, help='activation after norm')
    paparser.add_argument(*flex_var('-return_pre_act_norm'), type=misc.str2bool, metavar='', nargs='?', const=True, default=None, help='apply act/norm before weight/bias')
    paparser.add_argument(*flex_var('-return_dropout'), type=float, default=None, metavar='', help="dropout for return layers [args.dropout]")
    paparser.add_argument(*flex_var('-return_resnet'), type=misc.str2bool, metavar='', nargs='?', const=True, default=False, help="whether to use residual net (should never set it!)")
    paparser.add_argument(*flex_var('-return_resnet_beta'), type=float, default=1.0, metavar='', help='beta for resnet')
    paparser.add_argument(*flex_var('-return_norm_out'), type=str, default=None, metavar='', help="normalize output for the output to the module")

    # Conditional random field is used to refine the as-label outputs
    paparser.add_argument(*flex_var('-crf_args'), type=str, default='------- crf tower/module args', metavar='', help="------- crf tower/module args")
    paparser.add_argument(*flex_var('-crf_net'), type=str, default='linear', metavar='', help="not used yet")
    paparser.add_argument(*flex_var('-crf_num'), type=int, default=1, metavar='', help="# of CRF iterations")
    paparser.add_argument(*flex_var('-crf_dim'), type=int, nargs='+', default=None, metavar='', help="CRF layer dimensions in one iteration")

    paparser.add_argument(*flex_var('-loss_args'), type=str, default='======= loss/metric args =======', metavar='', help="======= loss/metric args =======")
    paparser.add_argument(*flex_var('-loss_fn'), type=str, nargs='+', default=['softmax+mse'], metavar='', help="loss function type: mse/bce/...")
    paparser.add_argument(*flex_var('-loss_fn_scale'), type=float, nargs='+', default=[1.0], metavar='', help="the scale for each loss_fn")
    # the loss_args below may only apply to some loss_fns
    paparser.add_argument(*flex_var('-loss_bpp_scale'), type=float, default=1.0, metavar='', help='bpp scale for loss_fn:fscore+upp, etc')
    paparser.add_argument(*flex_var('-loss_l2_scale'), type=float, default=1.0, metavar='', help='l2 scale for loss_fn:fscore+upp+l2, etc')
    # paparser.add_argument(*flex_var('-loss_masked_l1decay'), type=misc.str2bool, metavar='', nargs='?', const=True, default=False, help="add masked_l1decay to loss")
    paparser.add_argument(*flex_var('-loss_mask'), type=misc.str2bool, metavar='', nargs='?', const=True, default=False, help="apply mask when calculating loss")
    paparser.add_argument(*flex_var('-loss_sqrt'), type=misc.str2bool, metavar='', nargs='?', const=True, default=False, help="take sqrt before summing losses in a batch")
    paparser.add_argument(*flex_var('-loss_symmetric'), type=misc.str2bool, metavar='', nargs='?', const=True, default=False, help="inlcude f1 score for both positive and negative lables")
    # paparser.add_argument(*flex_var('-loss_cooldn_end'), type=float, default=0, help="cool down end point (function specific...)")
    paparser.add_argument(*flex_var('-loss_cooldn_steps'), metavar='', type=int, default=0, help="cool down loss function continuousely (function specific...)")
    paparser.add_argument(*flex_var('-loss_alpha'), type=float, nargs='+', default=[1.0,1.0], metavar='', help='the percentage of negative labels')
    paparser.add_argument(*flex_var('-loss_auto_alpha'), type=misc.str2bool, metavar='', nargs='?', const=True, default=False, help='whether to determine loss_alpha automatically')
    paparser.add_argument(*flex_var('-loss_auto_alpha_pow'), type=float, nargs='+', default=[1.0,0.0], metavar='', help='the power for auto_alpha, [begin, end (if cool down)]')
    paparser.add_argument(*flex_var('-loss_auto_alpha_mode'), type=str, default='npratio', metavar='', help='choose from npratio/length')
    paparser.add_argument(*flex_var('-loss_beta'), type=float, nargs='+', default=[1.0,1.0], metavar='', help='weight of recall wrt precision in f-score')
    paparser.add_argument(*flex_var('-loss_auto_beta'), type=misc.str2bool, metavar='', nargs='?', const=True, default=False, help='whether to determine loss_beta automatically')
    paparser.add_argument(*flex_var('-loss_auto_beta_pow'), metavar='', type=float, nargs='+', default=[1.0,0.0], help='the power for auto_beta, [begin, end (if cool down)]')
    paparser.add_argument(*flex_var('-loss_auto_beta_mode'), type=str, default='npratio', metavar='', help='choose from npratio/length')
    paparser.add_argument(*flex_var('-loss_gamma'), metavar='', type=float, nargs='+', default=[0.0,0.0], help='focal loss exponent [beign, end (if cool down)]')
    paparser.add_argument(*flex_var('-loss_twargs'), type=dict, default=dict(), metavar='', help='auto-stored kwargs dict from user tweaking')
    paparser.add_argument(*flex_var('-metric_fn'), type=str, nargs='+', default=None, metavar='', help='names of metric fns')
    paparser.add_argument(*flex_var('-metric_labels'), type=str, nargs='+', default=None, metavar='', help='names of metrics (auto-defined)')
    paparser.add_argument(*flex_var('-metric_threshold'), type=float, default=None, metavar='', help='threshold for metric_fn')
    paparser.add_argument(*flex_var('-metric_beta'), type=float, default=1.0, metavar='', help='beta for metric_fn')

    ## optimization
    paparser.add_argument(*flex_var('-optim_args'), type=str, default='======= optim args =======', metavar='', help="======= optim args =======")
    paparser.add_argument(*flex_var('-optim_fn'), type=str, default='adamw', metavar='', help="optimizer type: adamw/adam/sgd/...")
    paparser.add_argument(*flex_var('-optim_max_stride'), type=int, default=512, metavar='', help='max number of backprops per optim.step(), used for auto-step-stride increase only!')
    paparser.add_argument(*flex_var('-optim_step_stride'), type=int, default=1, metavar='', help='the number of backprops per optim.step()')

    paparser.add_argument(*flex_var('-learning_rate'), type=float, default=0.001, metavar='', help="change to 3e-4???")
    paparser.add_argument(*flex_var('-beta1'), type=float, default=0.9, metavar='', help="beta1 for Adam")
    paparser.add_argument(*flex_var('-beta2'), type=float, default=0.999, metavar='', help="beta2 for Adam")
    paparser.add_argument(*flex_var('-epsilon'), type=float, default=1e-7, metavar='', help="epsilon for Adam (10xdefault)")
    paparser.add_argument(*flex_var('-grad_clip'), type=str, default=None, metavar='', help='choose from globalnorm, norm, value')

    paparser.add_argument(*flex_var('-lr_warmup_steps'), type=int, default=7, metavar='', help='# of lr warmup epochs at the beginning (from args.learning_rate/steps)')
    paparser.add_argument(*flex_var('-lr_cooldn_steps'), type=int, default=7, metavar='', help='# of lr cooldn epochs at the end (to args.learning_rate/steps)')
    paparser.add_argument(*flex_var('-lr_scheduler'),  type=str, default='reduced', metavar='', help="learning rate scheduler")
    paparser.add_argument(*flex_var('-lr_factor'), type=float, default=0.7, metavar='', help="learning rate relative change factor")
    paparser.add_argument(*flex_var('-lr_patience'), type=int, default=7, metavar='', help="learning rate patience (checked every epoch)")

    paparser.add_argument(*flex_var('-weight_decay'), type=float, default=0.01, metavar='', help="weight decay for AdamW [default: 0.01]")
    paparser.add_argument(*flex_var('-l1decay'), type=float, default=None, metavar='', help="L1Decay rate for Adam [default: None]")
    paparser.add_argument(*flex_var('-l2decay'), type=float, default=None, metavar='', help="L2Decay rate for Adam [default: None]")

    # training
    paparser.add_argument(*flex_var('-train_args'), type=str, default='======= train args =======', metavar='', help="======= train args =======")
    paparser.add_argument(*flex_var('-train_size'), type=int, default=None, metavar='', help="for")
    paparser.add_argument(*flex_var('-valid_size'), type=int, default=None, metavar='', help="for train/evaluate/predict")
    paparser.add_argument(*flex_var('-predict_size'),  type=int, default=None, metavar='', help="for train/evaluate/predict")
    paparser.add_argument(*flex_var('-batch_size'),  type=int, default=1, metavar='', help="for train/evaluate/predict")
    paparser.add_argument(*flex_var('-drop_last'), type=misc.str2bool, metavar='', nargs='?', const=True, default=False, help='whether to drop the last incomplete batch')
    paparser.add_argument(*flex_var('-jit_loader'), type=misc.str2bool, metavar='', nargs='?', const=True, default=False, help="turn on jit data loading")
    paparser.add_argument(*flex_var('-num_epochs'), type=int, default=777, metavar='', help="# of maximum epochs (may be terminated early)")
    paparser.add_argument(*flex_var('-num_recaps_per_epoch'), type=int, default=10, metavar='', help="# of recaps/summaries per epoch")
    paparser.add_argument(*flex_var('-num_evals_per_epoch'), type=int, default=1, metavar='', help="# of eval. checkpoints per epoch")
    paparser.add_argument(*flex_var('-visual_dl'), type=misc.str2bool, metavar='', nargs='?', const=True, default=False, help='save one input and model data per recap')

    paparser.add_argument(*flex_var('-evaluate_callback'), type=str, default=None, metavar='', help="not used yet")
    paparser.add_argument(*flex_var('-chkpt_save_limit'), type=int, default=7, metavar='', help='maximum of saved checkpoint states')
    paparser.add_argument(*flex_var('-trainloss_rdiff'), type=float, default=1e-3, metavar='', help="relative difference")
    paparser.add_argument(*flex_var('-validloss_rdiff'), type=float, default=1e-3, metavar='', help="relative difference")
    paparser.add_argument(*flex_var('-trainloss_patience'), type=int, default=11, metavar='', help="train loss change patience")
    paparser.add_argument(*flex_var('-validloss_patience'), type=int, default=11, metavar='', help="valid loss change patience")
    paparser.add_argument(*flex_var('-marathon'), type=misc.str2bool, metavar='', nargs='?', const=True, default=False, help="a very long run with stringent stop criteria")
    paparser.add_argument(*flex_var('-nonstop'), type=misc.str2bool, metavar='', nargs='?', const=True, default=False, help="do not check train or valid loss")

    # pre-defined settings
    # paparser.add_argument(type=str, default='======= mood args =======', metavar='', help="======= mood args =======")
    # paparser.add_argument(*flex_var('-small'), type=misc.str2bool, metavar='', nargs='?', const=True, default=False, help="a small net to get a feel")
    # paparser.add_argument(*flex_var('-medium'), type=misc.str2bool, metavar='', nargs='?', const=True, default=False, help="hidden sizes of 32, 64 and 128")
    # paparser.add_argument(*flex_var('-large'), type=misc.str2bool, metavar='', nargs='?', const=True, default=False, help="a wide but shallow net")
    # paparser.add_argument(*flex_var('-xlarge'), type=misc.str2bool, metavar='', nargs='?', const=True, default=False, help="a very large net")
    # paparser.add_argument(*flex_var('-xxlarge'), type=misc.str2bool, metavar='', nargs='?', const=True, default=False, help="a very large net")
    # paparser.add_argument(*flex_var('-xxxlarge'), type=misc.str2bool, metavar='', nargs='?', const=True, default=False, help="a very large net")
    # paparser.add_argument(*flex_var('-shallow'), type=misc.str2bool, metavar='', nargs='?', const=True, default=False, help="a shallow net with three layers")
    # paparser.add_argument(*flex_var('-deep'), type=misc.str2bool, metavar='', nargs='?', const=True, default=False, help="a deep net with seven layers")
    # paparser.add_argument(*flex_var('-abysmal'), type=misc.str2bool, metavar='', nargs='?', const=True, default=False, help="a deep net with seven layers")

    paparser.add_argument(*flex_var('-action_args'), type=str, default='======= action args =======', metavar='', help="======= action args =======")

    # action as subparsers
    subparsers = parser.add_subparsers(dest='mission', required=True) #, formatter_class=formatter) # argparse.RawTextHelpFormatter)

    subparser = subparsers.add_parser('summary', parents=[paparser], description='', help='View the net/loss only', formatter_class=formatter)
    subparser = subparsers.add_parser('summarize', parents=[paparser], description='', help='alias for summary', formatter_class=formatter)
    subparser = subparsers.add_parser('view', parents=[paparser], description='', help='alias for summary', formatter_class=formatter)

    subparser = subparsers.add_parser('train', parents=[paparser], description='', help='just do it', formatter_class=formatter)
    subparser.set_defaults() # it will be overwritten by later set_defaults!
    subparser.add_argument(*flex_var("-scheduler"), type=str, nargs='?', const=None, default=None, metavar='', help='a file containing a list of dicts')

    subparser = subparsers.add_parser('dynamic_train', parents=[paparser], description='', help='not implemented', formatter_class=formatter)
    subparser.set_defaults() # it will be overwritten by later set_defaults!
    subparser.add_argument(*flex_var("-data_lens"), nargs='+', type=int, default=[300, 800, 2000, 5000], metavar='', help='a list of integers')
    subparser.add_argument(*flex_var("-batch_sizes"), nargs='+', type=int, default=[8, 4, 2, 1], metavar='', help='a list of integers')

    subparser = subparsers.add_parser('rename', parents=[paparser], description='', help='rename load_dir to save_dir (auto-generated if not set)', formatter_class=formatter)
    subparser.set_defaults()

    subparser = subparsers.add_parser('cross_validate', parents=[paparser], description='', help='cross evaluate the model', formatter_class=formatter)
    subparser.set_defaults()
    subparser.add_argument(*flex_var("-num_cvs"), type=int, default=5, metavar='', help='# of cross validations')

    subparser = subparsers.add_parser('evaluate', parents=[paparser], description='', help='predict and calculate loss', formatter_class=formatter)
    subparser.add_argument("valid_files", type=str, nargs='+', metavar='', help="input fasta files for validation")
    subparser.add_argument(*flex_var("-split_eval"), type=misc.str2bool, metavar='', nargs='?', const=True, default=False, help="split evaluation data like train/valid splits")
    subparser.add_argument(*flex_var("-save_lumpsum"), type=misc.str2bool, metavar='', nargs='?', const=True, default=False, help="save lumpsum files")
    subparser.add_argument(*flex_var("-save_individual"), type=misc.str2bool, metavar='', nargs='?', const=True, default=False, help="save individual files")
    subparser.add_argument(*flex_var("-save_genre"), type=str, nargs='+', default=['seq'], help="file types to save")
    subparser.add_argument(*flex_var("-named_after"), type=str, default='file', choices=['idx', 'id', 'file'], help="how to name saved files")

    subparser = subparsers.add_parser('predict', parents=[paparser], description='', help='predict and save', formatter_class=formatter)
    subparser.add_argument("predict_files", type=str, nargs='+', metavar='', help="input fasta files for prediction")
    subparser.add_argument(*flex_var("-save_lumpsum"), type=misc.str2bool, metavar='', nargs='?', const=True, default=True, help="save lumpsum files")
    subparser.add_argument(*flex_var("-save_individual"), type=misc.str2bool, metavar='', nargs='?', const=True, default=False, help="save individual files")
    subparser.add_argument(*flex_var("-save_genre"), type=str, nargs='+', default=['seq'], help="file types to save")
    subparser.add_argument(*flex_var("-named_after"), type=str, default='file', choices=['idx', 'id', 'file'], help="how to name saved files")

    subparser = subparsers.add_parser('average_model', parents=[paparser], description='average multiple models', help='average multiple models', formatter_class=formatter)
    subparser.add_argument("data_files", type=str, nargs='+', metavar='', help="input fasta files for prediction")
    subparser.add_argument(*flex_var("-model_dirs"), type=str, nargs='+', default=['./'], metavar='', help='directories of the models to average')
    subparser.add_argument(*flex_var("-model_weights"), type=str, nargs='?', const=None, default='loss', metavar='', help='directories of the models to average')
    subparser.add_argument(*flex_var("-best_chkpt"), type=misc.str2bool, metavar='', nargs='?', const=True, default=False, help='use the best check point in each directory')
    subparser.add_argument(*flex_var("-best_save"), type=misc.str2bool, metavar='', nargs='?', const=True, default=False, help='use the best save in each directory')
    subparser.add_argument(*flex_var("-by_epoch"), type=misc.str2bool, metavar='', nargs='?', const=True, default=True, help='use the best save in each directory')
    subparser.add_argument(*flex_var("-by_loss"), type=misc.str2bool, metavar='', nargs='?', const=True, default=False, help='use the best save in each directory')
    # subparser.set_defaults(save_dir='model_averages', metric_fn=['pfarm'])

    subparser.add_argument(*flex_var("-save_lumpsum"), type=misc.str2bool, metavar='', nargs='?', const=True, default=False, help="save lumpsum files")
    subparser.add_argument(*flex_var("-save_individual"), type=misc.str2bool, metavar='', nargs='?', const=True, default=False, help="save individual files")
    subparser.add_argument(*flex_var("-save_genre"), type=str, nargs='+', default=['seq'], help="file types to save")
    subparser.add_argument(*flex_var("-named_after"), type=str, default='file', choices=['idx', 'id', 'file'], help="how to name saved files")

    subparser = subparsers.add_parser('scan_data', parents=[paparser], description='', help='scan data_size and batch_size', formatter_class=formatter)
    subparser.add_argument(*flex_var("-data_sizes"), nargs='+', type=int, default=[0], metavar='', help='a list of integers')
    subparser.add_argument(*flex_var("-batch_sizes"), nargs='+', type=int, default=[1,2,4,8], metavar='', help='a list of integers')

    subparser = subparsers.add_parser('scout_args', parents=[paparser], description='', help='scout model args', formatter_class=formatter)#argparse.RawTextHelpFormatter)
    subparser.add_argument(*flex_var("args_info"), type=str, nargs='+', metavar='', help=
    '''arg_name,type,values for each var, USE COMA, NO SPACE!!!
    type   - one of int, float, logint, logfloat
    values - v1,v2,..vn (n>=2) for serial search (default) or grid_search
             min,max,num for random_polling or uniform_polling
             min,max,step for bayes_search
    ''')
    subparser.add_argument(*flex_var("-random_polling"),  type=misc.str2bool, metavar='T/F', nargs='?', const=True, default=False, help='flag for random_searches')
    subparser.add_argument(*flex_var("-uniform_polling"), type=misc.str2bool, metavar='T/F', nargs='?', const=True, default=False, help='flag for uniform searches')
    subparser.add_argument(*flex_var("-grid_search"),    type=misc.str2bool, metavar='T/F', nargs='?', const=True, default=False, help='flag for grid searches')
    subparser.add_argument(*flex_var("-bayes_search"),   type=int, default=0, metavar='N', help='number of Bayes searches')
    subparser.add_argument(*flex_var("-spawn_search"),   type=int, default=0, metavar='N', help='number of spawn searches')
    subparser.add_argument(*flex_var("-rebake_midat"), type=misc.str2bool, metavar='T/F', nargs='?', const=True, default=False, help='whether new midata need to be obtained for each iter')

    # subparser.add_argument(*flex_var("--arg_names"), nargs='+', type=str, default=['learning_rate', 'dropout'], metavar='', help='a list of strings')
    # subparser.add_argument(*flex_var("--arg_values"), nargs='+', type=str, default=['0.0001,0.001,0.01', '0.1,0.3,0.5'],
    #             metavar='', help='a list of STRINGs with values separated by "," for each arg\n' +
    #             'if grid_search is true, each string contains all values for the arg\n' +
    #             'if not grid_search, the string format is "min,max"')
    # subparser.add_argument(*flex_var("--arg_types"), nargs='+', type=int, default=[0], metavar='',
    #             help='the type for each arg, 0 for linear uniform, 1 for float uniform, 2 for int log uniform, 3 for float log uniform')

    # Note: ONLY ONE set of args is maintained for each model
    args = misc.Struct(vars(parser.parse_args(argv)))
    args.argv = " ".join(argv)
    argv_dict = dict(argv=args.argv, mission=args.mission)
    argv_dict.update(vars(misc.argv_lookup_values(argv, args))) # remember command line args
    # save "argv" and "mission" fields which will be overwritten when loading args.json
    return args, argv_dict


def autoconfig_args(args):
    """Set/reset default arg values not easily done with argparser """
    # directories should be path
    if args.data_dir: args.data_dir = Path(args.data_dir)
    if args.load_dir: args.load_dir = Path(args.load_dir)
    if args.save_dir: args.save_dir = Path(args.save_dir)

    # deal with input and feature
    args.data_genre = args.data_genre.lower()

    if isinstance(args.input_genre, str):
        args.input_genre = [args.input_genre]
    args.input_genre = [_s.lower().replace('-','_') for _s in args.input_genre]

    if args.data_genre in ['ab2021']:
        # only deals with the first input, which should be the primary input...
        if args.input_genre[0] in ['esm_mean', 'mean_esm']:
            if args.input_dim is None:
                args.input_dim = 3
        elif args.input_genre[0] in ['domain_feat', 'domain_prop']:
            if args.input_dim is None:
                args.input_dim = 141
        elif args.input_genre[0] in ['aa_feat', 'aa_prop']:
            if args.input_dim is None:
                args.input_dim = 28
        elif args.input_genre[0] in ['esm', 'esm_file', 'esm_mat', 'esm_esm']:
            if args.input_dim is None:
                args.input_dim = 1280
        else:
            logger.error(f'Unrecognized input_genre: {args.input_genre}!!!')
    elif args.data_genre in ['rna2d', 'contarna']:
        # input_genre determine input_dim
        if args.input_genre[0] in ['seq2onehot', 'onehot', 'one-hot']:
            if args.input_fmt is None:
                args.input_fmt = 'NLC'
            # args.input_ndim = 1
            if args.embed_fn == 'embed':
                logger.critical(f'Cannot have embed_fn: {args.embed_fn} for input_genre: {args.input_genre}!!!')
                args.embed_fn = None

            if args.input_dim is None:
                args.input_dim = 4 * (1 + 2 * args.feats_nn)
                if args.feats_dbn: args.input_dim += 3 * (1 + 2 * args.feats_nn)
                if args.feats_attr: args.input_dim += 8
                if args.feats_extra: args.input_dim += 2
                logger.info(f'Setting seq2onehot input_dim: {args.input_dim}')
        elif args.input_genre[0] in ['seq2quant', 'quant']:
            if args.input_fmt is None:
                args.input_fmt = 'NL'
            if args.embed_fn != 'embed':
                logger.warning(f'embed_fn: {args.embed_fn} is not embed for input_genre: {args.input_genre}!')

            if args.input_dim is None:
                args.input_dim = 6 ** (1 + 2 * args.feats_nn) # the number of possible values
                if args.feats_dbn:
                    args.input_dim *= 4 ** (1 + 2 * args.feats_nn)
                logger.info(f'Setting scalar input_dim: {args.input_dim}')
        elif args.input_genre[0] in ['bert']:
            if args.input_fmt is None:
                args.input_fmt = 'NLC'
            if args.input_dim is None:
                args.input_dim = 768
        elif args.input_genre[0] in ['attn']:
            if args.input_fmt is None:
                args.input_fmt = 'NCHW'
            if args.input_dim is None:
                args.input_dim = 12
        else:
            logger.critical(f'Unrecognized input_genre: {args.input_genre}!!!')
    else:
        logger.critical(f'Unknown data_genre: {args.data_genre}!!!')

    # deal with label type
    if isinstance(args.label_genre, str):
        args.label_genre = [args.label_genre]

    # below is now useless as args.label_genre is a list (was string)
    for i, label_genre in enumerate(args.label_genre):
        args.label_genre[i] = label_genre.lower()
        if args.label_genre[i] in ['upp']:
            if args.label_fmt is None: args.label_fmt = 'NL'
            if args.label_ntype is None: args.label_ntype = 2

        elif args.label_genre[i] in ['ct']:
            if args.label_fmt is None: args.label_fmt = 'NLL'
            if args.label_ntype is None: args.label_ntype = 2

        elif args.label_genre[i] in ['delta_g', 'deltag']:
            if args.label_fmt is None: args.label_fmt = 'N'

        elif args.label_genre[i] in ['tangle', 'f1']:
            if args.label_fmt is None: args.label_fmt = 'NL'

        else:
            logger.critical(f'Unrecognized label_genre: {args.label_genre}!!!')

    # >1 batch_size requires the same sequence length
    if args.batch_size > 1 and hasattr(args.data_len, '__len__') and args.data_len[-1] == 0:
        logger.critical(f'NO sequence length padding with batch_size:{args.batch_size} > 1!!!')
        # args.data_len[-1] = -1

    # net parameters
    if args.init_dropout is None: args.init_dropout = args.dropout / 3
    if args.linear_dropout is None: args.linear_dropout = args.dropout
    if args.conv1d_dropout is None: args.conv1d_dropout = args.dropout / 2
    if args.conv2d_dropout is None: args.conv2d_dropout = args.dropout / 2
    if args.attn_dropout is None: args.attn_dropout = args.dropout / 2
    if args.attn_attn_dropout is None: args.attn_attn_dropout = args.attn_dropout / 2
    if args.attn_ffdropout is None: args.attn_ffdropout = args.dropout
    if args.lstm_dropout is None: args.lstm_dropout = args.dropout / 2
    if args.return_dropout is None: args.return_dropout = args.dropout / 2

    if args.depth > 0:
        logger.info(f'Setting number of layers = {args.depth}')
        if args.linear_num is None: args.linear_num = args.depth
        if args.conv1d_num is None: args.conv1d_num = args.depth
        if args.conv2d_num is None: args.conv2d_num = args.depth
        if args.lstm_num is None:
            if args.lstm_direction in ['bidirect', 'bidirectional', 'bi'] :
                args.lstm_num = args.depth // 2
            else:
                args.lstm_num = args.depth
        if args.attn_num is None: args.attn_num = args.depth

    if args.width > 0:
        args.width = (args.width // 8) * 8
        logger.info(f'Setting channel sizes = {args.width}')
        if args.embed_dim is None: args.embed_dim = args.width
        if args.init_dim is None: args.init_dim = [args.width, args.width]
        if args.linear_dim is None: args.linear_dim = [args.width, args.width]
        if args.conv1d_dim is None: args.conv1d_dim = [args.width, args.width]
        if args.conv2d_dim is None: args.conv2d_dim = [args.width, args.width]
        if args.lstm_dim is None:
            if args.lstm_resnet and args.lstm_direction in ['bidirect', 'bidirectional', 'bi']:
                args.lstm_dim = [args.width // 2, args.width //2]
            else:
                args.lstm_dim = [args.width, args.width]

        if args.attn_nhead is None:
            if args.width % 16 == 0:
                args.attn_nhead = max([2, args.width // 16])
            elif args.width % 12 == 0:
                args.attn_nhead = max([2, args.width // 12])
            elif args.width % 8 == 0:
                args.attn_nhead = max([2, args.width // 8])
            else:
                args.attn_nhead = 1 if args.width % 2 else 2
        if args.return_dim is None: args.return_dim = [args.width] * 3 + [2]

    # mood args
    if args.marathon:
        args.num_epochs = 77777
        args.trainloss_rdiff = 1e-4
        args.trainloss_patience = 21
        args.validloss_rdiff = 1e-4
        args.validloss_patience = 21

    return args


def vocab2onehot_dict(vocab, dtype=np.float32):
    """ return one-hot dict """
    identity_mat = np.identity(len(vocab), dtype=dtype)

    onehot_dict = dict()
    for i, token in enumerate(vocab):
        onehot_dict[token] = identity_mat[i]

    return onehot_dict


def calc_padding(kernel_size, stride=1, dilation=1):
    """ the goal is to keep num_channels constant """
    return ((dilation * (kernel_size - 1) + 1) - stride) / 2


def random_sample(midata, size=1, replace=False):
    """ midata can be a list/tuple/np.ndarray, replace=True will yield repeated elements """
    return [midata[i] for i in np.random.choice(len(midata), size, replace=replace)]


def fix_length(data, length=1, skip_dims=None, **kwargs):
    """ cut or pad to the given length, use """
    if not hasattr(data, '__len__'): data = np.array([data])
    if skip_dims is None:
        skip_dims = []
    elif not hasattr(skip_dims, '__len__'):
        skip_dims = [skip_dims]

    pad_width = ()
    pad_noyes = False
    for i, dim in enumerate(data.shape):
        if i in skip_dims:
            pad_width += (0, 0),
            continue
        if dim > length:
            data = data.take(indices=range(length), axis=i)
            pad_width += (0, 0),
        else:
            pad_width += (0, length - dim),
            pad_noyes = True

    if pad_noyes:
        return np.pad(data, pad_width, **kwargs)
    else:
        return data


def fix_length1d(data, length, **kwargs):
    """ if needed, np.pad is used for padding with the same kwargs """
    if not hasattr(data, '__len__'): data = np.aray([data])
    data_len = len(data)

    if data_len >= length:
        return data[:length]
    else:
        return np.pad(data, (0, length - data_len), **kwargs)


def fix_length2d(data, length, **kwargs):
    """ data is 2D matrix without batch dim
    np.pad is used for padding with **kwargs"""
    data_len = data.shape

    if isinstance(length, int) or isinstance(length, np.integer):
        length = [length]

    len2pad = [0, 0]

    if data_len[0] >= length[0]: # check 1st dimension
        data = data[:length[0], :]
    else:
        len2pad[0] = length[0] - data_len[0]

    if data_len[1] >= length[-1]: # check 2nd dimension
        data = data[:, :length[-1]]
    else:
        len2pad[1] = length[-1] - data_len[1]

    if any(len2pad): # pad if needed
        return np.pad(data, ((0, len2pad[0]), (0, len2pad[1])), **kwargs)
    else:
        return data


def cut_padding(data, seq_len, ):
    """ The 1st dim is taken as batch dim if ndim > 1
        The last dim is taken as feature if ndim > 2 (not yet so)
    """

    if hasattr(seq_len, '__len__'): seq_len = max(seq_len)

    if data.ndim == 1:
        data = data[:seq_len]
    elif data.ndim == 2:
        data = data[:, :seq_len]
    elif data.ndim == 3:
        data = data[:, :seq_len, :seq_len]
    elif data.ndim == 4:
        data = data[:, :seq_len, :seq_len, :seq_len]
    elif data.ndim == 5:
        data = data[:, :seq_len, :seq_len, :seq_len, :seq_len]

    return data


def soft2hard_label(data, keep_dim=False, mi=np):
    """ convert soft to hard labels via argmax() """
    hard_data = data.argmax(axis=-1)
    if keep_dim:
        return mi.expand_dims(hard_data, -1)
    else:
        return hard_data


def hard2soft_label(data, ntype=2, discrete=False, mi=np):
    """ true label starts from zero
    accept non-integers for hard labels, in which weights are assigned to
    two neighboring classes depending on the distance
    """

    soft_data = mi.zeros(list(data.shape) + [ntype], dtype='float32')

    for i in range(ntype): # there must be a better way...
        soft_data[..., i] = mi.clip(mi.abs(data - i), 0.0, 1.0)

    soft_data = 1.0 - soft_data

    return soft_data


def load_chop_midat(args=misc.Struct(), **kwargs):
    r"""Load midata (default a dict or dataframe)
    Update/add columns including "idx", "len"
    Filter/select data according to args.data_len and args.data_size

    Return:
        pkldata: a dataframe

    Arguments:
        kwargs > args > def_args
    """
    def_args = misc.Struct(dict(
        data_dir = '',
        data_name = 'train',
        # data_suffix = '.pkl',
        data_len = [-1],
        data_size = None,
        verbose = 1,
    ))
    def_args.update(vars(args))
    def_args.update(kwargs)
    args.update(vars(def_args))

    # Future version:
    # should check whether fname contains wildcard or is empty
    # in this case, only the file names are saved to pkldata


    # if args.data_dir is None:
    #     pkldata_file = Path(args.data_name)
    # else:
    #     pkldata_file = Path(args.data_dir) / args.data_name

    # if not pkldata_file.is_file():
    #     pkldata_file = pkldata_file.with_suffix('.pkl')
    # # try without the data_dir
    # if not pkldata_file.is_file() and args.data_dir is not None:
    #     pkldata_file = Path(args.data_name)
    # if not pkldata_file.is_file():
    #     pkldata_file = pkldata_file.with_suffix('.pkl')

    # read the pkl file
    # logger.info(f'======= Loading Data =======')
    # logger.info(f'      file name: {pkldata_file}')
    # try:
    #     with pkldata_file.open('rb') as iofile:
    #         df = pickle.load(iofile) # it is a dictionary of id, seq, dbn, ct... when available
    # except TypeError:
    #     df = pd.read_pickle(pkldata_file)

    logger.info(f'======= Loading Data =======')
    logger.info(f'      file name: {args.data_name}, dir: {args.data_dir}')
    df = brew_midat.get_midat(args.data_name, data_dir=args.data_dir, return_save_prefix=False)

    # go with pd.DataFrame (for backward-compatibility, no longer needed)
    # if isinstance(df, dict): df = pd.DataFrame(df)

    args.num_samples = df.shape[0]
    if 'idx' not in df.columns: df['idx'] = list(range(1, args.num_samples + 1))

    # ct data sometimes have wrong 'len' in the header,
    if 'seq' in df: df['len'] = df['seq'].str.len()
    args.max_len = df['len'].max()

    logger.info(f' Num of samples: {args.num_samples}, min len: {df["len"].min()}, max len: {args.max_len}, user len: {args.data_len}')

    # apply sequence length
    #    1) input_len as [min, max, pad_flag]
    #    2) input_len as [pad_flag] (all sequences used)
    # pad_flag is always the last element: >0: cut/pad to the length, 0: keep original, <0: use max_len
    if type(args.data_len) not in (list, tuple):
        args.data_len = [args.data_len]
    if len(args.data_len) == 3: # [min, max, pad_flag]
        len_min = args.data_len[0]
        len_max = args.data_len[1]
        df = df[(df.len >= len_min) & (df.len <= len_max)]
        logger.info(f'Num of selected: {df.shape[0]}, length range: [{len_min}, {len_max}]')

    # apply data_include and data_exclude
    if args.data_include is not None and len(args.data_include) > 1:
        logger.info(f'Applying data_include: {args.data_include}...')
        df = df[df[args.data_include[0]].isin(args.data_include[1:])]
        logger.info(f'{df.shape[0]} samples left after data_include')

    if args.data_exclude is not None and len(args.data_exclude) > 1:
        logger.info(f'Applying data_exclude: {args.data_exclude}...')
        df = df[~ df[args.data_exclude[0]].isin(args.data_exclude[1:])]
        logger.info(f'{df.shape[0]} samples left after data_exclude')

    # apply data_range: [colname, min, max]
    if args.data_range is not None and len(args.data_range) > 1:
        logger.info(f'Applying data_range: {args.data_range}...')
        colname = args.data_range[0]
        if colname in df.columns:
            df = df[df[colname] >= float(args.data_range[1])]
            if len(args.data_range) > 2:
                df = df[df[colname] <= float(args.data_range[2])]
            logger.info(f'{df.shape[0]} samples left after data_range')
        else:
            logger.warning(f'dataframe does not have column: {colname}! (can be safely ignored)')

    # apply data_ratio: [colname1, colname2, min, max]
    if args.data_ratio is not None and len(args.data_ratio) > 3:
        logger.info(f'Applying data_ratio: {args.data_ratio}...')

        if args.data_ratio[0] in df.columns and args.data_ratio[1] in df.columns:
            df = df[df[args.data_ratio[0]] >= float(args.data_ratio[2]) * df[args.data_ratio[1]]]
            if len(args.data_ratio) > 3:
                df = df[df[args.data_ratio[0]] <= float(args.data_ratio[3]) * df[args.data_ratio[1]]]
            logger.info(f'{df.shape[0]} samples left after data_ratio')
        else:
            logger.warning(f'Dataframe does not have all cols: {args.data_ratio[0:2]}! (can be safely ignored)')

    # apply data_size
    if args.data_size is not None and args.data_size > 0:
        if df.shape[0] >= args.data_size:
            logger.info(f'Selecting data size: {args.data_size} out of size: {df.shape[0]}...')
            df = df.sample(args.data_size)
        else:
            logger.info(f'data size: {args.data_size} >= num data: {df.shape[0]}, not applied!')

    # get the embed_seqlen (0 means to keep current length!)
    if args.data_len[-1] < 0: args.data_len[-1] = df.len.max()

    # logger.info(f'Columns: {df.columns.to_list()}')

    return df


def bake_midat_ab2021(df, args=misc.Struct(), **kwargs):
    r"""Process midata (dataframe/dict) to a list of samples data

    Returns:
        midata: a list of data items for each sample

    Arguments:
        df: must be a dataframe, not a data series, even if just one row
        kwargs > args > def_args
    """
    def_args = misc.Struct(dict(
        data_genre = 'ab2021',
        input_fmt = 'NLC',
        input_genre = 'esm',
        feats_attr = False,
        feats_extra = False,
        label_genre = 'delta_g',
        jit_loader = False,
        verbose = 1,
    ))

    def_args.update(vars(args))
    def_args.update(kwargs)
    args.update(vars(def_args))

    # collect basic info about data
    num_samples = df.shape[0]

    if args.jit_loader:
        logger.debug(f'input genre: {args.input_genre}, fmt: {args.input_fmt}\n' + \
                 f'attr: {args.feats_attr}, label_genre: {args.label_genre}')
    else:
        logger.info(f'input genre: {args.input_genre}, fmt: {args.input_fmt}\n' + \
                 f'attr: {args.feats_attr}, label_genre: {args.label_genre}')

    if 'train' in args.mission:
        df = df[df.num_domains == 2]

    # only use multiprocessing when args.jit_loader == False
    if args.jit_loader:
        map_func = lambda func, *iterable: list(map(func, *iterable))
        starmap_func = itertools.starmap
    else:
        map_func = misc.mpi_map
        starmap_func = misc.mpi_starmap

    # get "input": sequence data
    if args.input_genre in ['mean_esm', 'esm_mean']:
        input_data = np.stack((
                        np.stack(df['anarci_seq_esm_mean'].values, 0),
                        np.stack(df['antigen_seq_esm_mean'].values, 0),
                        np.stack(df['seq_esm_mean'].values, 0),
                        ), axis=-1).astype(np.float32)
    elif args.input_genre in ['domain_prop', 'domain_feat', 'aa_feat', 'aa_prop',
                            'esm_mat', 'aa_esm']:
        input_data = df[args.input_genre].to_list()
    elif args.input_genre in ['esm_esm', 'two_esm', 'esm_two']:
        input_data = df['two_esm'].to_list()
    elif args.input_genre in ['esm_file']:
        lib_dir = args.data_dir / args.data_name
        input_data = []
        for idx in df.idx:
            logger.info(f'Loading esm data for idx: {idx}...')
            input_data.append(np.concatenate((
                        np.loadtxt(lib_dir / f'{idx}.anarci_seq.esm'),
                        np.loadtxt(lib_dir / f'{idx}.antigen_seq.esm'),
                        np.loadtxt(lib_dir / f'{idx}.seq.esm'),
                        ), axis=0).astype(np.float32))
    else:
        logger.error(f'Unrecognized input_genre: {args.input_genre}!!!')
    # seq_input = seq_input.astype(np.float32)

    lendata = df[['len', 'idx']].to_numpy(dtype=int)

    # add feats_extra to the seqdata (aka input)
    if args.batch_size > 1:
        _ncols = input_data[0].shape[-1]
        input_data = [fix_length2d(_v, [args.max_len, _ncols]) for _v in input_data]

    # get "y": upp/ct/...
    deltag_data = None
    if 'delta_g' in df:
        logger.debug('Processing delta_g data...')

        deltag_data = np.stack(df['delta_g'].values, 0).astype(np.float32)

    # return
    midata = None
    args.label_genre = args.label_genre.lower()
    if args.label_genre == 'delta_g':
        if deltag_data is None:
            midata = list(zip(input_data, lendata))
        else:
            midata = list(zip(input_data, lendata, deltag_data))
    else:
        logger.error(f'label genre: {args.label_genre} is not supported!!!')

    if args.verbose > 1:
        shapes = [data.shape for data in midata[0]]
        print(f'Number of datasets: {len(midata)}')
        print(f'Number of items in each set: {len(midata[0])}, with shapes: {shapes}')
        print('Checking for consistent dimensions...')
        for i, data in enumerate(midata):
            for j, item in enumerate(data):
                if shapes[j] != item.shape:
                    print(f'The shape of dataset #{i} item #{j}: {item.shape} differs from the first: {shapes[j]}')
        print('Done!')

    return midata


def seq2bpmat_quad(seq, min_delta_ij=3, min_stem_len=1, dtype='float32'):
    bpmat_quad = np.stack([
        molstru.seq2bpmat_just_pairs(seq, min_delta_ij=min_delta_ij, min_stem_len=min_stem_len,
            return_energy=False, dtype='float32'),
        molstru.seq2bpmat_just_pairs(seq, min_delta_ij=min_delta_ij, min_stem_len=min_stem_len,
            return_energy=True, dtype='float32'),
        molstru.seq2bpmat_gauss_neighbors(seq, min_delta_ij=min_delta_ij, min_stem_len=min_stem_len,
            nn=12, dtype='float32'),
        molstru.seq2bpmat_turner_neighbors(seq, min_delta_ij=min_delta_ij,
            nn=1, dtype='float32'),
        ],
        axis=-1,
        )
    return bpmat_quad


def smoothen_label(label, smooth=0.01, ntype=2):
    return (1.0 - smooth) * label + smooth / ntype


def bake_midat_contarna(df, args=misc.Struct(), **kwargs):
    r"""Process midata (dataframe/dict) to a list of samples as inputs

    Returns:
        midat: a list of data items for each sample. The order of data items are:
               midat[0]     :    [idx, len]
               midat[1]     :    input_feats (onehot/quant/extrat/etc.)
               ......       :    additional inputs (bpmat/etc.)
               onehot_vec   :    mainly for post-processing
               ......       :    additional labels
               midat[-1]    :    labels

    Arguments:
        df:     Must be a dataframe, not a data series, even if just one row.

                len and seq are required column/keys

        kwargs > args > def_args
    """
    def_args = dict(
        input_genre = ['seq2onehot'], # onehot or quant
        input_fmt = 'NLC',
        feats_nn = 0, # do not include nearest neighbor
        feats_dbn = False, 
        feats_attr = False,
        feats_extra = False,
        label_genre = ['upp'],
        label_fmt = 'NL',
        label_tone = 'none',
        label_ntype = 2,
        label_smooth = 0.0,
        label_min_delta_ij = 0,
        label_min_stem_len = 1,
        data_len = [0],
        jit_loader = False,
        verbose = 1,
    )

    args.update(def_args, skip_existing=True)
    if len(kwargs): args.update(kwargs)

    midat = [] # the list of inputs to return
    # always the first item: idx, len
    midat.append(df[['idx', 'len']].to_numpy(dtype=int))

    # collect basic info about data
    num_seqs = len(df['seq'])
    input_fix2len = args.data_len[-1]

    if isinstance(args.input_genre, str):
        args.input_genre = [args.input_genre]

    if hasattr(args, 'input_bpmat') and args.input_bpmat and 'bpmat' not in args.input_genre:
        args.input_genre += ['bpmat']

    if args.jit_loader:
        logger.debug(f'======= Bake Data (size: {len(df)}) =======')
        logger.debug(f'input genre: {args.input_genre}, fmt: {args.input_fmt}, ' + \
                f'nn: {args.feats_nn}, dbn: {args.feats_dbn}, ' + \
                f'attr: {args.feats_attr}, extra: {args.feats_extra}, ' + \
                f'fix2len: {input_fix2len}'
                )
        logger.debug(f'label genre: {args.label_genre}, fmt: {args.label_fmt}, ' + \
                f'tone: {args.label_tone}, ntype: {args.label_ntype}, ' + \
                f'min_delta_ij: {args.label_min_delta_ij}, smooth: {args.label_smooth}')
    else:
        logger.info(f'======= Bake Data (size: {len(df)}) =======')
        logger.info(f'input genre: {args.input_genre}, fmt: {args.input_fmt}, ' + \
                f'nn: {args.feats_nn}, dbn: {args.feats_dbn}, ' + \
                f'attr: {args.feats_attr}, extra: {args.feats_extra}, ' +
                f'fix2len: {input_fix2len}'
                )
        logger.info(f'label genre: {args.label_genre}, fmt: {args.label_fmt}, ' + \
                f'tone: {args.label_tone}, ntype: {args.label_ntype}, ' + \
                f'min_delta_ij: {args.label_min_delta_ij}, smooth: {args.label_smooth}')

    # only use multiprocessing when args.jit_loader == False
    if args.jit_loader:
        map_fn = lambda func, *iterable, **dummy: list(map(func, *iterable))
        starmap_fn = lambda func, *iterable, **dummy: itertools.starmap(func, *iterable)
        # starmap_func = itertools.starmap
    else:
        map_fn = misc.mpi_map
        starmap_fn = misc.mpi_starmap

    # get "input" to minet: infeatures

    for input_genre in args.input_genre:
        if input_genre.startswith(('seq2onehot', 'onehot', 'seq2quant', 'quant')):

            if input_genre.startswith(('seq2onehot', 'onehot', 'one-hot')):
                input_feats_fn = functools.partial(molstru.seq2onehot,
                            use_nn=args.feats_nn,
                            use_attr=args.feats_attr,
                            use_dbn=args.feats_dbn,
                            length=input_fix2len)
            elif input_genre.startswith(('seq2quant', 'quant')):
                input_feats_fn = functools.partial(molstru.seq2quant,
                            use_nn=args.feats_nn,
                            use_dbn=args.feats_dbn,
                            length=input_fix2len)

            if not args.jit_loader:
                logger.info(f'Getting input features, fmt: {input_genre} ...')

            if args.feats_dbn:
                input_feats = starmap_fn(input_feats_fn, zip(df['seq'], df['dbn']),
                            desc='Getting infeatures from seq&dbn')
            else:
                input_feats = map_fn(input_feats_fn, df['seq'], desc='Getting infeatures from seq')

            # add feats_extra to the seqdata (aka input)
            if args.feats_extra:
                if not args.jit_loader: logger.info(f'Concatenating input extra features ...')
                input_feats = [np.concatenate((_v, fix_length(df.iloc[_i]['extra'], len(_v), skip_dims=1)),
                        axis=1) for _i, _v in enumerate(input_feats)]

        elif input_genre.startswith(('bert')):
            if not args.jit_loader: logger.info('Collecting input bert representations...')
            input_feats = df['bert'].tolist()
            if input_fix2len > 0:
                pad_func = functools.partial(fix_length, length=input_fix2len, skip_dims=[1])
                input_feats = map_fn(pad_func, input_feats, desc='Applying fix_length')

        elif input_genre.startswith(('attn')):
            if not args.jit_loader: logger.info('Collecting input attention maxtrices...')
            input_feats = df['attn'].tolist()
            # input_x = [_x.transpose((1,2,0)) for _x in input_x]
            if input_fix2len > 0:
                # the first dimension is the number of layers
                pad_func = functools.partial(fix_length, length=input_fix2len, skip_dims=[0])
                input_feats = map_fn(pad_func, input_feats, desc='Applying fix_length')

        elif input_genre in ['bpmat']: # the base pair matrix for padding
            if not args.jit_loader: logger.info(f'Generating input bp constraint matrix...')
            # meta_seq = map_func(functools.partial(molstru.seq2vector, length=input_len), df['seq'])
            bpmat = map_fn(#functools.partial(
                # molstru.seq2bpmat_just_pairs, return_energy=False, min_delta_ij=3,
                #     min_stem_len=1, dtype='float32'),
                functools.partial(seq2bpmat_quad, min_delta_ij=args.label_min_delta_ij, min_stem_len=args.label_min_stem_len),
                df['seq'],
                desc=f'Calculating seq2bpmat, min_delta_ij: {args.label_min_delta_ij}, min_stem_len: {args.label_min_stem_len}')

            if input_fix2len > 0:
                pad_func = functools.partial(fix_length2d, mode='constant', constant_values=((0, 0), (0, 0)))
                bpmat = starmap_fn(
                    pad_func,
                    zip(bpmat, [input_fix2len] * num_seqs),
                    desc='Fixing bpmat length(2D)')

            input_feats = bpmat
        else:
            logger.critical(f'Unknown input genre: {input_genre}')
            continue
        midat.append(input_feats)

    # decided to add the raw x in onehot format (always before the labels)
    rawx_fn = functools.partial(molstru.seq2onehot,
                use_nn=0,
                use_attr=False,
                use_dbn=False,
                length=input_fix2len)
    # rawx_fn = functools.partial(molstru.seq2quant,
    #             use_nn=0,
    #             use_dbn=False,
    #             length=input_fix2len)

    rawx = map_fn(rawx_fn, df['seq'], desc='Getting raw_x from seq')
    midat.append(rawx)

    # get "label": upp/ct/...
    for label_genre in args.label_genre:
        if label_genre in ['upp']:
            if not args.jit_loader: logger.info('Processing label upp data...')

            if 'upp' in df.columns:
                label_y = df['upp'].to_list()
            elif 'ct' in df.columns:
                logger.info('Converting label ct to upp...')
                if args.label_min_delta_ij:
                    ct = map_fn(functools.partial(molstru.ct_clip_delta_ij,
                                min_delta_ij=args.label_min_delta_ij), df['ct'],
                                desc=f'Applying label_min_delta_ij: {args.label_min_delta_ij}')
                else:
                    ct = df['ct']

                if args.label_min_stem_len > 1:
                    logger.error(f'Not yet implemented for min_tem_len: {args.label_min_stem_len}!!!!')

                label_y = starmap_fn(molstru.ct2upp, zip(ct, df['len']), desc='Applying ct2upp')
            else:
                logger.warning(f'Cannot find either upp or ct in data! (ignore if predicting)')
                continue
                raise ValueError('Neither upp or ct is found in dataset!!!')

            if input_fix2len > 0:
                logger.info(f'Padding length to {input_fix2len} ...')
                pad_func = functools.partial(fix_length1d, mode='constant', constant_values=(0.0,0.0))
                label_y = starmap_fn(pad_func, zip(label_y, [input_fix2len] * num_seqs),
                                        desc='Fixing 1D length')

            midat.append(label_y)

        elif label_genre in ['ct']:
            if not args.jit_loader: logger.debug('Processing label ct data...')

            if 'ct' in df.columns:
                label_y = starmap_fn(functools.partial(molstru.ct2ctmat, dtype=np.float32),
                                zip(df['ct'], df['len']), desc=f'Applying ct2mat')
            else:
                logger.warning(f'Cannot find ct in data! (ignore if predicting)')
                continue

            if args.label_smooth > 0:
                label_y =  map_fn(functools.partial(
                    smoothen_label, smooth=args.label_smooth, ntype=args.label_ntype),
                    label_y,
                    desc=f'Smoothening label by: {args.label_smooth} with ntype: {args.label_ntype}',
                )

            if args.label_min_delta_ij:
                label_y = map_fn(functools.partial(molstru.bpmat_clip_delta_ij,
                                min_delta_ij=args.label_min_delta_ij), label_y,
                                desc=f'Applying label_min_delta_ij: {args.label_min_delta_ij}',
                                )
            if args.label_min_stem_len > 1:
                logger.error(f'Not yet implemented for min_tem_len: {args.label_min_stem_len}!!!!')

            if input_fix2len > 0:
                pad_func = functools.partial(fix_length2d, mode='constant', constant_values=((0, 0), (0, 0)))
                label_y = starmap_fn(pad_func, zip(label_y, [input_fix2len] * num_seqs),
                                    desc='Fixing 2D length')

            # expand the last dimension
            # ctdata = [np.expand_dims(_ct, -1) for _ct in ctdata]

            midat.append(label_y)

        elif label_genre in ['tangle']:
            if 'tangle' not in df.columns:
                logger.warning(f'Cannot find tangle in data! (ignore if predicting)')
                continue                

            angles = df['tangle'].to_list()
            masks = [(_angle != 1000.0).astype(_angle.dtype) for _angle in angles] # 1 for selected
            rad_angles = [np.pi / 180.0 * _angle for _angle in angles]
            sin_angles = map_fn(np.sin, rad_angles)
            cos_angles = map_fn(np.cos, rad_angles)

            label_y = zip(masks, angles, sin_angles, cos_angles)
            label_y = [np.stack(_label, -1)  for _label in label_y] # [L, num_angles, [mask,angle,sin,cos]] for one sample

            if input_fix2len > 0:
                # only pad the seq dimension (0), not the angle dimension (1), or the last dim for
                pad_func = functools.partial(fix_length, length=input_fix2len, skip_dims=[1,2], constant_values=0.0)
                label_y = map_fn(pad_func, label_y, desc='Applying fix_length to tangle')

            midat.append(label_y)

        elif label_genre in ['f1']:
            # just use len as space keeper
            label_y = df['len'].to_list()
            midat.append(label_y)

        else:
            logger.critical(f'Unrecognized label_genre: {label_genre}!!!')
            
    # re-order to get one list item for one sample
    midat = list(zip(*midat))

    if args.verbose > 1:
        shapes = [data.shape for data in midat[0]]
        print(f'Number of datasets: {len(midat)}')
        print(f'Number of items in each set: {len(midat[0])}, with shapes: {shapes}')
        print('Checking for consistent dimensions...')
        for i, data in enumerate(midat):
            for j, item in enumerate(data):
                if shapes[j] != item.shape:
                    print(f'The shape of dataset #{i} item #{j}: {item.shape} differs from the first: {shapes[j]}')
        print('Done!')

    return midat


def posencoder_bert_1d(x_shape, seqs_len=None, mi=np, debug=False):
    """ only first two dims of x_shape are used: [batch_size, data_len, ...]
    adopted from e2efold source code
    """

    assert len(x_shape) >= 2, "input ndim must be at least 2"

    batch_size = x_shape[0]
    data_len = x_shape[1]

    if seqs_len is None:
        logger.warning(f'No sequence length is passed, use data_len!!!')
        seqs_len = mi.full((batch_size), data_len, dtype='float32')

    pos_i_abs = mi.linspace(1, data_len, num=data_len, dtype='float32').reshape(
                (1, data_len, 1)).expand((batch_size, -1, -1))

    pos_i_rel =  mi.linspace(1, data_len, num=data_len, dtype='float32').reshape(
                (1, data_len, 1)).expand((batch_size, -1, -1))

    pos_i_rel = pos_i_rel / seqs_len.reshape((-1, 1, 1))

    pos = mi.concat([pos_i_abs, pos_i_rel], -1)

    pe = list()

    # 1/x, 1/x^2
    pe.append(pos)
    pe.append(1.0 / pos_i_abs)
    pe.append(1.0 / mi.pow(pos_i_abs, 2))

    # sin(nx)
    for n in range(1, 50):
        pe.append(mi.sin(n * pos))

    # poly
    for i in range(2, 5):
        pe.append(mi.pow(pos_i_rel, i))

    for i in range(3):
        gaussian_base = mi.exp(-mi.pow(pos, 2)) * math.sqrt(math.pow(2, i) / math.factorial(i)) * mi.pow(pos, i)
        pe.append(gaussian_base)

    pe = mi.concat(pe, -1)

    for i in range(batch_size):
        pe[i, seqs_len[i]:, :] = 0.0

    logger.debug(f'Generating [BERT] position encoder {x_shape} with dim: {pe.shape}')

    if debug:
        from mpl_toolkits.axes_grid1 import make_axes_locatable
        plt.ion()
        ax = plt.subplot()
        im = ax.matshow(pe[0], cmap='RdBu') # RdBu is another cmap choice
        plt.xlabel(f'Feature Dim C={pe.shape[2]}'), plt.ylabel(f'Data Length L={pe.shape[1]}')
        plt.title(f'Position Encoder 1D: {x_shape} for seq_len={int(seqs_len[0])}')
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="8%", pad=0.05)
        plt.colorbar(im, cax=cax)
        plt.show()
        # plt.savefig('position_encoder1d_debug.png')
    return pe


def posencoder_trig_1d(instance_size, base=None, curve='trig', mi=np, debug=False):
    """ x dim: [i=batch_size, j=seq_len, k=input_dim]
    As the longest wavelength (for the last two dimensions) is 2pi*base,

    The rationale is that the rate of positional variation is proportional to 1/wavelength,
    so the positional encoding is stronger at shorter wavelengths (i.e., earlier features).
    Larger wavelengths at later dimensions allow the semantic information in the input to
    take over.

    So the choice of 10000 may be optimal in the sense that the wavelength of 20000*pi
    is large enough that the semantics will dominate at the higher feature dimensions.
    """

    assert len(instance_size) >= 2, "input ndim must be at least 2"
    assert instance_size[-1] % 2 == 0, "feature dim must be even"

    if base is None: base = 10000.0

    jlen = instance_size[-2]
    klen = instance_size[-1]
    # base = mi.to_tensor(base, dtype='float32')

    j = mi.arange(0, jlen, 1, dtype='float32').reshape((jlen, 1))
    k = mi.arange(0, klen // 2, 1, dtype='float32').reshape((1, klen // 2))

    # omega_k = 1 / 10000 ** (2 * k / klen)
    omega_k = mi.exp(-math.log(base) * 2.0 / klen * k)

    omega_jk = mi.matmul(j, omega_k)

    pe = mi.empty((jlen, klen), dtype='float32')
    pe[:, 0::2] = mi.sin(omega_jk)
    pe[:, 1::2] = mi.cos(omega_jk)

    logger.info(f'Generating [{curve}] position encoder of [{jlen}, {klen}] with base {base}')

    if debug:
        from mpl_toolkits.axes_grid1 import make_axes_locatable
        plt.ion()
        ax = plt.subplot()
        im = ax.matshow(pe, cmap='Jet') # RdBu is another cmap choice
        plt.xlabel(f'Feature Dim C={klen}'), plt.ylabel(f'Seq. Length L={jlen}')
        plt.title(f'Position Encoder 1D: {instance_size}')
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="8%", pad=0.05)
        plt.colorbar(im, cax=cax)
        plt.show()
        # plt.savefig('position_encoder1d_debug.png')
    return pe


def posencoder_trig_2d(instance_size, base=None, curve='trig', mi=np, debug=False):
    """ x dim: [i=height, j=width, k=input_dim]
    Adopted from https://github.com/wzlxjtu/PositionalEncoding2D
    The key idea is to encode row/height in the first half of input_dim,
    and column/width in the second half of input_dim
    """
    # x = mi.empty((args.batch_size, 50, 128), # args.input_dim), dtype='float32')
    assert len(instance_size) >= 3, "input dim must be at least 3"
    assert instance_size[-1] % 4 == 0, "feature dim must be a multiple of 4"

    ilen = instance_size[-3]
    jlen = instance_size[-2]
    klen = instance_size[-1]

    if base is None:
        base = mi.full([1], 10000.0, dtype='float32')

    i = mi.arange(0, ilen, dtype='float32').reshape((ilen, 1))
    j = mi.arange(0, jlen, dtype='float32').reshape((jlen, 1))

    # half of input_dim is used for height or width separately
    k = mi.arange(0, klen // 4, 1, dtype='float32').reshape((1, klen // 4))

    # omega_k = 1 / 10000 ** (2 * k / klen)
    omega_k = mi.exp(-mi.log(base) * 4.0 / klen * k) # 4.0 here because klen is

    omega_ik = mi.matmul(i, omega_k).unsqueeze(1) # [i, 1, k]
    omega_jk = mi.matmul(j, omega_k).unsqueeze(0) # [1, j, k]

    pe = mi.empty((ilen, jlen, klen), dtype='float32')

    midk = klen // 2
    pe[:, :, 0:midk:2] = mi.sin(omega_ik)
    pe[:, :, 1:midk:2] = mi.cos(omega_ik)
    pe[:, :, midk::2] = mi.sin(omega_jk)
    pe[:, :, midk + 1::2] = mi.cos(omega_jk)

    logger.info(f'Generating [{curve}] position encoder of [{ilen}, {jlen}, {klen}] with base {int(base)}')

    if debug:
        plt.ion()
        plt.figure(figsize=(12,15))
        plt.subplot(221), plt.imshow(pe[:,:,0])
        plt.xlabel(f'Width={jlen}'), plt.ylabel(f'Height={ilen}'), plt.title(f'featuer_dim: 0')
        plt.subplot(222), plt.imshow(pe[:,:,klen // 2 - 1])
        plt.xlabel(f'Width={jlen}'), plt.ylabel(f'Height={ilen}'), plt.title(f'featuer_dim: {klen // 2 - 1}')
        plt.subplot(223), plt.imshow(pe[:,:,klen // 2])
        plt.xlabel(f'Width={jlen}'), plt.ylabel(f'Height={ilen}'), plt.title(f'featuer_dim: {klen // 2}')
        plt.subplot(224), plt.imshow(pe[:,:,-1])
        plt.xlabel(f'Width={jlen}'), plt.ylabel(f'Height={ilen}'), plt.title(f'featuer_dim: {klen - 1}')
        plt.suptitle(f'Position Encoder 2D: {instance_size}')
        # plt.colorbar(im, cax=cax)
        plt.show()
    return pe


def posencoder_rotary_1d(instance_size, base=None, curve='trig', mi=np, debug=False):
    """ x dim: [i=batch_size, j=seq_len, k=input_dim]
    As the longest wavelength (for the last two dimensions) is 2pi*base,

    The rationale is that the rate of positional variation is proportional to 1/wavelength,
    so the positional encoding is stronger at shorter wavelengths (i.e., earlier features).
    Larger wavelengths at later dimensions allow the semantic information in the input to
    take over.

    So the choice of 10000 may be optimal in the sense that the wavelength of 20000*pi
    is large enough that the semantics will dominate at the higher feature dimensions.
    """

    assert len(instance_size) >= 2, "input ndim must be at least 2"
    assert instance_size[-1] % 2 == 0, "feature dim must be even"

    if base is None: base = 10000.0

    jlen = instance_size[-2]
    klen = instance_size[-1]
    # base = mi.to_tensor(base, dtype='float32')

    j = mi.arange(0, jlen, 1, dtype='float32').reshape((jlen, 1))
    k = mi.arange(0, klen // 2, 1, dtype='float32').reshape((1, klen // 2))

    # omega_k = 1 / 10000 ** (2 * k / klen)
    omega_k = mi.exp(-math.log(base) * 2.0 / klen * k)

    omega_jk = mi.matmul(j, omega_k)

    pe = mi.empty((jlen, klen), dtype='float32')
    pe[:, 0::2] = mi.sin(omega_jk)
    pe[:, 1::2] = mi.cos(omega_jk)

    logger.info(f'Generating [{curve}] position encoder of [{jlen}, {klen}] with base {base}')

    if debug:
        from mpl_toolkits.axes_grid1 import make_axes_locatable
        plt.ion()
        ax = plt.subplot()
        im = ax.matshow(pe, cmap='Jet') # RdBu is another cmap choice
        plt.xlabel(f'Feature Dim C={klen}'), plt.ylabel(f'Seq. Length L={jlen}')
        plt.title(f'Position Encoder 1D: {instance_size}')
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="8%", pad=0.05)
        plt.colorbar(im, cax=cax)
        plt.show()
        # plt.savefig('position_encoder1d_debug.png')
    return pe


def get_attn_mask_efficient_1d(data_len, seqs_len, attn_mask=None, tile_mask=None, mi=np):
    """ This is specifically for efficient attention masking
        attn_mask is the old mask to be re-used if possible
        mask dimension: [batch_size, nhead=1, data_len, input_dim=1]
    """
    if seqs_len is None:
        return None
    elif not hasattr(seqs_len, '__len__'):
        seqs_len = mi.to_tensor(seqs_len)

    # no mask if all lengths are greater than or equal to data_len
    if mi.all(seqs_len >= data_len):
        return None

    batch_size = len(seqs_len)
    tile_size = 1
    if tile_mask:
        if not hasattr(tile_mask, '__len__'): tile_mask = [tile_mask]
        for _i in tile_mask: tile_size *= _i

    # mask is added to the product before softmax in paddlepaddle
    if attn_mask is None or attn_mask.ndim != 4 or batch_size * tile_size > attn_mask.shape[0] or \
            attn_mask.shape[-2] != data_len:
        mask_shape = (batch_size * tile_size, 1, data_len, 1)
        logger.info(f'Generate new mask with shape: {mask_shape}, x_shape: {data_len}')
        attn_mask = mi.full(mask_shape, 0.0) # -np.inf)
    else:
        attn_mask[:] = 0.0

    for i in range(batch_size):
        # attn_mask[i, 0, :seqs_len[i], :seqs_len[i]] = 0
        # assign both just in case (Only the first of the two below is needed)
        attn_mask[i * tile_size : (i + 1) * tile_size, :, seqs_len[i]:, :] = -np.inf

    return attn_mask


def get_attn_mask_efficient_2d(data_len, seqs_len, attn_mask=None, mi=np):
    """ This is specifically for efficient attention masking
        attn_mask is the old mask to be re-used if possible (not yet implemented)
        mask dimension: [batch_size, nhead=1, height*width, feature_dim=1]
    """
    if seqs_len is None:
        return None
    elif not hasattr(seqs_len, '__len__'):
        seqs_len = mi.to_tensor(seqs_len)

    x_height = data_len[0]
    x_width = data_len[1]

    # no mask needed if seqs_len are greater than or equal to data lengths
    if all(seqs_len.numpy() >= x_height) and all(seqs_len.numpy() >= x_width):
        return None

    batch_size = len(seqs_len)

    # mask is added to the product before softmax in paddlepaddle
    # if attn_mask is None or attn_mask.ndim !=4 or \
                # attn_mask.shape[-2] < x_height or attn_mask.shape[-1] < x_width:
    # yet to find a way to reuse the matrix
    attn_mask = mi.full((batch_size, x_width), 0.0) # -np.inf)
    # else:
    #     attn_mask[:] = 0.0
    #     attn_mask = attn_mask.reshape(batch_size, )

    for i in range(batch_size):
        # attn_mask[i, 0, :seqs_len[i], :seqs_len[i]] = 0
        # assign both just in case (Oonly the first of the two below is needed)
        attn_mask[i, seqs_len[i]:] = -np.inf

    attn_mask = mi.tile(attn_mask, [1, x_height])

    # 2nd dim == 1 is the attn_nhead
    return attn_mask.reshape([batch_size, 1, x_height * x_width, 1])


def get_attn_mask_1d(data_len, seqs_len, attn_mask=None, tile_mask=None, mi=np):
    """ data_len is an integer referring to the L value in NLC
        seqs_len can be either numpy.ndarry or mi.Tensor
        attn_mask is the old mask to be re-used if possible
        tile_mask is for axial attention only, referring to the total size of other axes
            merged to the batch_size axis
        mask dimension: [batch_size*tile_size, nhead=1, data_len=1, data_len]
    """
    if seqs_len is None:
        return None
    if not hasattr(seqs_len, '__len__'):
        seqs_len = [seqs_len]
    # if hasattr(seqs_len, 'numpy'):
    #     seqs_len = seqs_len.numpy()
    # else:
        # seqs_len = np.array(seqs_len, dtype='int32')

    # no mask if all lengths are greater than or equal to data_len
    if mi is not np and isinstance(seqs_len, np.ndarray):
        seqs_len = mi.to_tensor(seqs_len, dtype='int32')

    if mi.all(seqs_len >= data_len):
        return None

    batch_size = len(seqs_len)
    tile_size = 1
    if tile_mask:
        if not hasattr(tile_mask, '__len__'): tile_mask = [tile_mask]
        for _i in tile_mask: tile_size *= _i

    # mask is added to the product before softmax in paddlepaddle
    if attn_mask is None or attn_mask.ndim != 4 or batch_size * tile_size > attn_mask.shape[0] or \
            attn_mask.shape[-1] != data_len or attn_mask.shape[-2] != data_len:
        # the 2nd dimension is attn_nhead
        mask_shape = (batch_size * tile_size, 1, 1, data_len)
        logger.info(f'Generate new mask with shape: {mask_shape}, seqs_len: {seqs_len}')
        attn_mask = mi.full(mask_shape, 0.0) # -np.inf)
    else:
        attn_mask[:] = 0.0

    for i in range(batch_size):
        # print(type(tile_size), type(seqs_len))
        # print(seqs_len)
        # attn_mask[i, 0, :seqs_len[i], :seqs_len[i]] = 0
        # assign both just in case (Only the first of the two below is needed)
        attn_mask[i * tile_size : (i + 1) * tile_size, :, :, seqs_len[i]:] = -np.inf

    return attn_mask

    # below is the old method creating [batch_size*tile_size, nhead=1, data_len, data_len]

    # mask is added to the product before softmax in paddlepaddle
    if attn_mask is None or attn_mask.ndim != 4 or batch_size * tile_size > attn_mask.shape[0] or \
            attn_mask.shape[-1] != data_len or attn_mask.shape[-2] != data_len:
        # the 2nd dimension is attn_nhead
        mask_shape = (batch_size * tile_size, 1, data_len, data_len)
        logger.info(f'Generate new mask with shape: {mask_shape}, seqs_len: {seqs_len}')
        attn_mask = mi.full(mask_shape, 0.0) # -np.inf)
    else:
        attn_mask[:] = 0.0

    for i in range(batch_size):
        # attn_mask[i, 0, :seqs_len[i], :seqs_len[i]] = 0
        # assign both just in case (Only the first of the two below is needed)
        attn_mask[i * tile_size : (i + 1) * tile_size, :, :seqs_len[i], seqs_len[i]:] = -np.inf
        attn_mask[i * tile_size : (i + 1) * tile_size, :, seqs_len[i]:, :seqs_len[i]] = -np.inf

    return attn_mask


def get_attn_mask_2d(data_len, seqs_len, attn_mask=None, mi=np):
    """ This is specifically for efficient attention masking
        attn_mask is the old mask to be re-used if possible (not yet implemented)
        mask dimension: [batch_size, nhead=1, height*width, height*width]
    """
    if seqs_len is None:
        return None
    elif not hasattr(seqs_len, '__len__'):
        seqs_len = mi.to_tensor(seqs_len)

    x_height = data_len[0]
    x_width = data_len[1]

    # no mask needed if seqs_len are greater than or equal to data lengths
    if all(seqs_len.numpy() >= x_height) and all(seqs_len.numpy() >= x_width):
        return None

    batch_size = len(seqs_len)

    # mask is added to the product before softmax in paddlepaddle
    # if attn_mask is None or attn_mask.ndim !=4 or \
                # attn_mask.shape[-2] < x_height or attn_mask.shape[-1] < x_width:
    # yet to find a way to reuse the matrix
    attn_mask = mi.full((batch_size, x_height, x_width), 0.0) # -np.inf)
    # else:
    #     attn_mask[:] = 0.0
    #     attn_mask = attn_mask.reshape(batch_size, )

    for i in range(batch_size):
        # attn_mask[i, 0, :seqs_len[i], :seqs_len[i]] = 0
        # assign both just in case (Oonly the first of the two below is needed)
        attn_mask[i, :seqs_len[i], seqs_len[i]:] = -np.inf
        attn_mask[i, seqs_len[i]:, :seqs_len[i]] = -np.inf

    attn_mask = mi.tile(attn_mask, [1, x_width, x_height])

    # 2nd dim == 1 is the attn_nhead
    return attn_mask.reshape([batch_size, 1, x_height * x_width, x_height * x_width])


def pfarm_metric(guess, label, batch=False, beta=1.0, threshold=None,
        epsilon=1e-9, overflow=0.5):
    """ return precision, fscore, accuracy, recall, mcc
    inputs (guess and label) can be paddle.Tensor, np.ndarray,
    return np.ndarrays (batch=True) or scalars (batch=False)
    """
    assert guess.ndim == label.ndim, \
        f'Guess of {guess.shape} and Label of {label.shape} must have the same ndims!'

    if batch:
        sum_axis = tuple(range(1, guess.ndim))
        # batch_size = input.shape[0]
        data_size = guess[0].size
    else:
        sum_axis = tuple(range(0, guess.ndim))
        # batch_size = 1
        data_size = guess.size

    if threshold is not None: # discretize the input (assume binary)
        guess = (guess > threshold).astype(label.dtype)
        # if threshold == 0.5:
        #     guess = guess.round()
        # else:
        #     # if 'float' not in str(guess.dtype): guess = guess.astype('float32')
        #     # if 'float' not in str(label.dtype): label = label.astype('float32')
        #     guess = (guess - threshold + 0.5).round()
            # label += -threshold + 0.5
        # if 'float' in str(guess.dtype): guess = guess.round()
        # if 'float' in str(label.dtype): label = label.round()

    ##### decided to go with CPU for the remaining calculations
    ##### the rationale is that the values are usually scalars or arrays of batch_size

    p = np.array(label.sum(sum_axis), dtype='float32') # all positive
    pp = np.array(guess.sum(sum_axis), dtype='float32') # all predicted positive
    tp = np.array((guess * label).sum(sum_axis), dtype='float32') # true positive

    fp = pp - tp # (input * (1.0 - label)).sum(sum_axis)
    fn = p - tp # ((1.0 - input) * label).sum(sum_axis)
    tn = data_size - pp - fn # ((1 - input) * (1 - label)).sum()

    # if type(p) is np.ndarray:
    # if batch:
    #     mcc = (tp * tn - fp * fn + epsilon) / ((pp * p * (tn + fp) * (data_size - p)).astype('float32').sqrt() + epsilon)
    # else:
    #     mcc = (tp * tn - fp * fn + epsilon) / (math.sqrt(float(pp) * p * (tn + fp) * (data_size - p)) + epsilon)

    mcc = (tp * tn - fp * fn + epsilon) / (np.sqrt(pp * p * (tn + fp) * (data_size - p)) + epsilon)

    acc = (tp + tn) / data_size

    recall = (tp + epsilon) / (p + epsilon)
    # if not hasattr(recall, '__len__'):
    #     if p == 0: recall = overflow
    # elif isinstance(recall, np.ndarray):
    #     recall[p == 0.0] = overflow
    # else:
    #     for _i in mi.nonzero(p == 0.0):
    #         recall[_i] = overflow
        # recall[mi.nonzero(p == 0.0).squeeze()] = 1.0

    # print(p, pp, pp, tp, recall)

    precision = (tp + epsilon) / (pp  + epsilon)
    # if not hasattr(precision, '__len__'):
    #     if pp == 0: precision = overflow
    # elif isinstance(precision, np.ndarray):
    #     precision[pp == 0.0] = overflow
    # else: # squeeze is needed for paddle CPU version
    #     # print(pp)
    #     for _i in mi.nonzero(pp == 0.0):
    #         precision[_i] = overflow
            # precision[mi.nonzero(pp == 0.0).squeeze()] = 1.0

    beta2 = beta * beta
    if beta2 == 1.0:
        fscore = (2.0 * tp + epsilon) / (2.0 * tp + fp + fn + epsilon)
        # fscore = 2.0 * precision * recall / (precision + recall + epsilon)
    else:
        # fscore = (1.0 + beta2) * tp / ((1.0 + beta2) * tp + beta2 * fp + fn + epsilon)
        fscore = ((1.0 + beta2) * precision * recall + epsilon) / (beta2 * precision + recall + epsilon)

    # if not hasattr(fscore, '__len__'):
    #     if precision + recall == 0: fscore = 0.0
    # elif isinstance(fscore, np.ndarray):
    #     fscore[mi.nonzero(precision + recall == 0.0)[0]] = 0.0
    # else:
    #     fscore[mi.nonzero(precision + recall == 0.0)] = 0.0

    return np.stack([precision, fscore, acc, recall, mcc], axis=-1).astype('float32')


def pfarm_variance(guess, label=None, batch=False, beta=1.0, threshold=0.5, epsilon=1e-11):
    """ return estimated variances of precision, fscore, accuracy, recall, mcc
    inputs (guess and label) can be paddle.Tensor, np.ndarray,
    return np.ndarrays (batch=True) or scalars (batch=False)

    NOTE: variance estimation is only approximiate, as we assume pp_var, tp_var, fp_var etc to be indepdendent!!!
    """
    if label is None:
        label = molstru.ppmat2ctmat(guess, threshold=threshold, dtype=guess.dtype)

    assert guess.ndim == label.ndim, \
        f'Guess of {guess.shape} and Label of {label.shape} must have the same ndims!'

    if batch:
        sum_axis = tuple(range(1, guess.ndim))
        # batch_size = input.shape[0]
        data_size = guess[0].size
    else:
        sum_axis = tuple(range(0, guess.ndim))
        # batch_size = 1
        data_size = guess.size

    # the variance of guess: sqrt(p(1-p))
    guess_var = np.sqrt(guess * (1.0 - guess))

    p = label.sum(sum_axis) # all positive
    pp = guess.sum(sum_axis) # all predicted positive
    tp = (guess * label).sum(sum_axis) # true positive

    fp = pp - tp # (input * (1.0 - label)).sum(sum_axis)
    fn = p - tp # ((1.0 - input) * label).sum(sum_axis)
    tn = data_size - pp - fn # ((1 - input) * (1 - label)).sum()

    pp_var = np.sqrt((guess_var ** 2).sum(sum_axis))
    tp_var = np.sqrt(((guess_var * label) ** 2).sum(sum_axis))

    fp_var = np.sqrt(((guess_var * (1.0 - label)) ** 2).sum(sum_axis))
    fn_var = tp_var
    tn_var = fp_var

    # NOTE: the variances above are NOT independent, but assumed to be!!!

    mcc = (tp * tn - fp * fn + epsilon) / (np.sqrt(pp * p * (tn + fp) * (data_size - p)) + epsilon)
    mcc_var = np.sqrt(
        ((tp_var * tn) ** 2 + (tp * tn_var) ** 2 + (fp_var * fn) ** 2 + (fp * fn_var) ** 2 + epsilon) /
            (pp * p * (tn + fp) * (data_size - p) + epsilon) + \
        ((tp * tn - fp * fn) ** 2 + epsilon) / ((pp * p * (tn + fp) * (data_size - p)) ** 3 + epsilon) * \
            ((data_size - p) * p + epsilon) ** 2 * ((pp_var * (tn + fp)) ** 2 + (pp * tn_var) ** 2 + (pp * fp_var) ** 2 + epsilon)
    )

    acc = (tp + tn) / data_size
    acc_var = np.sqrt(tp_var ** 2 + tn_var ** 2) / data_size

    recall = (tp + epsilon) / (p + epsilon)
    recall_var = (tp_var + epsilon) / (p + epsilon)

    precision = (tp + epsilon) / (pp + epsilon)
    precision_var = np.sqrt((tp_var ** 2 + epsilon) / (pp ** 2 + epsilon) + \
                            ((tp * pp_var) ** 2 + epsilon) / (pp ** 4 + epsilon))

    beta2 = beta * beta
    if beta2 == 1.0:
        # fscore = (2.0 * tp + epsilon) / (2.0 * tp + fp + fn + epsilon)
        # fscore_var = np.sqrt(
        #     (4.0 * tp_var ** 2 + epsilon) / ((2.0 * tp + fp + fn) ** 2 + epsilon) + \
        #     (4.0 * tp ** 2 * (4.0 * tp_var ** 2 + fp_var ** 2 + fn_var ** 2) + epsilon) /
        #         ((2.0 * tp + fp + fn) ** 4 + epsilon)
        # )

        fscore = (2.0 * tp + epsilon) / (p + pp + epsilon)
        fscore_var = 2.0 * np.sqrt(((((label * (p + pp) - 1) / (p + pp + epsilon) ** 2) * guess_var) ** 2).sum(sum_axis))

        # fscore = 2.0 * precision * recall / (precision + recall + epsilon)
    else:
        # fscore = (1.0 + beta2) * tp / ((1.0 + beta2) * tp + beta2 * fp + fn + epsilon)
        fscore = ((1.0 + beta2) * precision * recall + epsilon) / (beta2 * precision + recall + epsilon)
        fscore_var = np.sqrt(
            ((1.0 + beta2) ** 2 * ((precision_var * recall) ** 2 + (precision * recall_var) ** 2) + epsilon) /
                ((beta2 * precision + recall) ** 2 + epsilon) + \
            (((1.0 + beta2) * precision * recall) ** 2 * (beta2 * beta2 * precision_var ** 2 + recall_var ** 2) + epsilon) /
                ((beta2 * precision + recall) ** 4 + epsilon)
        )

    # return np.stack([precision, fscore, acc, recall, mcc], axis=-1).astype('float32')
    return np.stack([precision_var, fscore_var, acc_var, recall_var, mcc_var], axis=-1).astype('float32')


def save_loss_csv(save_file, loss_df, groupby=None):
    """  """
    df = loss_df
    if groupby is not None:
        col = [_col for _col in groupby if _col in loss_df.columns]
        if col:
            logger.info(f'Grouping csv data by: {col} ...')
            df = loss_df.groupby(col).mean().reset_index()

    if df is not None:
        logger.info(f'Writing csv data to: {save_file} ...')
        df.to_csv(save_file, index=False, float_format='%.4g')
    # np.savetxt(save_file, train_loss, fmt='%6d ' + '%8.4f ' * 4 + ' %5d'*4)


def save_predict_matrix(y_model, save_dir='./', seqs_len=None, names=None):
    """  """
    num_seqs = len(y_model)
    save_dir = Path(save_dir)

    if names is None:
        names = list(range(num_seqs))

    ndim = y_model[0].ndim
    for i in range(num_seqs):
        if seqs_len is None:
            y_save = y_model[i]
        else:
            seq_len = int(seqs_len[i])
            if ndim == 0:
                y_save = y_model[i]
            elif ndim == 1:
                y_save = y_model[i][:seq_len]
            elif ndim == 2:
                y_save = y_model[i][:seq_len, :seq_len]
            elif ndim == 3:
                y_save = y_model[i][:seq_len, :seq_len, :seq_len]
            elif ndim == 4:
                y_save = y_model[i][:seq_len, :seq_len, :seq_len, :seq_len]
            else:
                logger.critical('too many dimensions to save')

        if not isinstance(y_save, np.ndarray):
            pass

        if ndim <= 2:
            np.savetxt(save_dir / f'{names[i]}.txt', y_save, fmt='%10.8f')
        else:
            np.save(save_dir / f'{names[i]}.npy', y_save)


def save_all_results(midata, save_prefix, save_dir=None,
        args=misc.Struct(label_genre=[''], save_genre=['seq'], named_after='idx',
                         save_lumpsum=True, save_individual=False, output_threshold=None)):
    """ save all relevant model files in midata dataframe
        model prediction is saved in the 'predict" column
    """
    save_dir = Path.cwd() if save_dir is None else Path(save_dir)

    csv_exclude = ['seq', 'ct', 'ct_guess', 'bp', 'bpmat', 'bpmat_guess', 'ctmat', 'ctmat_guess',
                   'ppmat', 'ppmat_guess', 'upp', 'upp_guess',
                   'tangle', 'tangle_guess']
    pkl_exclude = []

    if type(args.save_genre) not in (tuple, list):
        args.save_genre = [args.save_genre]
    if isinstance(args.label_genre, str):
        args.label_genre = [args.label_genre]

    # rename some colnames for consistency
    if 'ct' in args.label_genre: # not ct but ppmat is predicted
        if 'ppmat_guess' in midata:
            logger.warning('ppmat_guess already exists in dataframe, but it will be dropped!')
            midata.drop(columns=['ppmat_guess'], inplace=True, errors='ignore')
        midata.rename(columns={"ct_guess":"ppmat_guess"}, inplace=True, copy=False)

    # save results
    logger.info(f'Saving results to {misc.str_color(save_dir)} ...')

    if args.save_lumpsum:
        logger.info(f'Lumpsum files will be saved with prefix: {misc.str_color(save_prefix)} ...')
        brew_midat.save_lumpsum_files(midata, save_dir=save_dir, save_prefix=save_prefix,
            save_pkl=True, save_csv=True, save_fasta=True,
            save_unknown=False, save_duplicate=False, save_conflict=False,
            csv_exclude=csv_exclude, pkl_exclude=pkl_exclude)

    if args.save_individual:
        if 'ct' in args.label_genre:
            args.save_genre += ['seq', 'bpseq', 'ppmat']
            # 'ctmat' column will be used to generate 'bpseq' files
            midata.drop(columns=['ppmat'], inplace=True, errors='ignore')
            midata.rename(columns={"ppmat_guess":"ppmat"}, inplace=True, copy=False)
            threshold = 0.5 if args.output_threshold is None else args.output_threshold
            # bpseq will be calculated from ctmat
            midata['ctmat'] = [(_ppm > threshold).astype('int32') for _ppm in midata['ppmat']]
            # midata['ct'] = misc.mpi_map(molstru.ctmat2ct, ctmat_all, desc='ctmat2ct')
        if 'upp' in args.label_genre:
            args.save_genre += ['seq', 'upp']
            midata.drop(columns=['upp'], inplace=True, errors='ignore')
            midata.rename(columns={"upp_guess":"upp"}, inplace=True, copy=False)
        # else:
            # logger.warning(f'Unsupported label_genre: {args.label_genre}!!!')

        logger.info(f'Individual files will be named after: {args.named_after} ...')
        logger.info(f'  save genre: {args.save_genre}, save_dir: {save_dir}')
        brew_midat.save_individual_files(midata, save_dir=save_dir, named_after=args.named_after,
            save_genre=list(set(args.save_genre)))


def optuna_server(direction='maximize'):

    def objective(trial):

        classifier_name = trial.suggest_categorical('classifier', ['SVC', 'RandomForest'])
        if classifier_name == 'SVC':
            svc_c = trial.suggest_loguniform('svc_c', 1e-10, 1e10)
            # classifier_obj = sklearn.svm.SVC(C=svc_c)
        else:
            rf_max_depth = int(trial.suggest_loguniform('rf_max_depth', 2, 32))
            # classifier_obj = sklearn.ensemble.RandomForestClassifier(max_depth=rf_max_depth)

        params = {'learning_rate': trial.suggest_loguniform('learning_rate', 0.01, 0.5),
                'max_depth': trial.suggest_int('max_depth', 1, 30),
                'num_leaves': trial.suggest_int('num_leaves', 2, 100),
                'min_data_in_leaf': trial.suggest_int('min_data_in_leaf', 10, 1000),
                'feature_fraction': trial.suggest_uniform('feature_fraction', 0.1, 1.0),
                'subsample': trial.suggest_uniform('subsample', 0.1, 1.0)}

        return 1.0

    # options for create_study
    sampler = optuna.integration.SkoptSampler(skopt_kwargs={
            'base_estimator':'RF',
            'n_random_starts':10,
            'base_estimator':'ET',
            'acq_func':'EI',
            'acq_func_kwargs': {'xi':0.02}
            })

    pruner=optuna.pruners.SuccessiveHalvingPruner()

    import joblib
    joblib.dump(study, 'artifacts/study.pkl')
    study = joblib.load('artifacts/study.pkl')