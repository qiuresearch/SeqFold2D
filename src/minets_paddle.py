import paddle as mi
import paddle.nn as nn
import paddle.nn.functional as F
from paddle.static import InputSpec

import math
import numpy as np
from itertools import cycle
import logging
# from mpl_toolkits.axes_grid1 import make_axes_locatable

# homebrew
import misc
import mitas_utils
logger = logging.getLogger(__name__)


def set_global_initializer(weight_fn, bias=0.0):
    """ Done in MyEmbeddingLayer, not sure the best or even correct place for it """
    weight_fn = weight_fn.lower()
    if weight_fn == 'kaimingnormal':
        weight_init = nn.initializer.KaimingNormal()
    elif weight_fn == 'xaviernormal':
        weight_init = nn.initializer.XavierNormal()
    elif weight_fn == 'xavieruniform':
        weight_init = nn.initializer.XavierUniform()
    elif weight_fn == 'kaiminguniform':
        weight_init = nn.initializer.KaimingUniform()
    elif weight_fn == 'normal':
        weight_init = nn.initializer.Normal()
    elif weight_fn == 'uniform':
        weight_init = nn.initializer.Uniform()
    elif weight_fn == 'bilinear':
        weight_init = nn.initializer.Bilinear()
    elif weight_fn == 'constant':
        weight_init = nn.initializer.Constant(value=1.0)
    elif weight_fn == 'truncatednormal':
        weight_init = nn.initializer.TruncatedNormal(mean=0.0, std=1.0)
    else:
        logger.critical(f'Unrecognized weight init fn: {weight_fn}!!!')

    logger.info(f'Set global initializer, weight: {misc.str_color(weight_fn, style="reverse")}, bias: {bias} ...')
    nn.initializer.set_global_initializer(weight_init, nn.initializer.Constant(bias))
    return None


def get_act_layer(act_fn):
    """ None is returned if unrecognized """

    if act_fn in [None, False, 'none', 'None', 'NONE']: return None

    act_fn = act_fn.lower()
    if act_fn == 'relu':
        return nn.ReLU()
    elif act_fn == 'relu6':
        return nn.ReLU6()
    elif act_fn == 'leakyrelu':
        return nn.LeakyReLU(negative_slope=0.01)
    elif act_fn == 'prelu':
        return nn.PReLU(init=0.25)
    elif act_fn == 'gelu':
        return nn.GELU(approximate=True)
    elif act_fn == 'elu':
        return nn.ELU(alpha=1.0)
    elif act_fn == 'tanh':
        return nn.Tanh()
    elif act_fn == 'maxout':
        return nn.Maxout(groups=3)
    elif act_fn == 'sigmoid':
        return nn.Sigmoid()
    elif act_fn == 'hardsigmoid':
        return nn.Hardsigmoid()
    elif act_fn == 'softmax':
        return nn.Softmax()
    elif act_fn == 'swish':
        return nn.Swish()
    else:
        logger.critical(f'cannot recognize act_fn: {act_fn}')
        return None


def get_norm_layer(**kwargs):
    """ None is returned if unrecognized """

    if kwargs['fn'] in [None, False, 'none', 'None', 'NONE']: return None

    norm_layer = MyNormLayer(**kwargs)
    if norm_layer.is_None:
        return None
    elif hasattr(norm_layer, 'NormFn'):
        return norm_layer.NormFn
    else:
        return norm_layer


def get_dropout_layer(dropout, data_fmt=None):
    """ choose dropout fn based on data_fmt """
    if dropout <= 0.0: return None

    if data_fmt is None or len(data_fmt) <= 3 or len(data_fmt) >= 6:
        return nn.Dropout(dropout)#, name=f'Dropout{dropout:0.2g}') # axis=None
    elif len(data_fmt) == 4:
        return nn.Dropout2D(dropout, data_format=data_fmt)#, name=f'Dropout2D{dropout:0.2g}')
    elif len(data_fmt) == 5:
        return nn.Dropout3D(dropout, data_format=data_fmt)#, name=f'Dropout3D{dropout:0.2g}')
    else:
        logger.critical(f'Unknown data_fmt: {data_fmt} !!!')
        return None


def stack_linear_block(channel_sizes, data_fmt=None, pre_act_norm=False, act_after_norm=False,
                act_fn=None, norm_args=dict(fn=None), dropout=0.0, resnet=False, resnet_beta=None,
                weight_attr=None, bias_attr=None, is_return=False):
    """ return a list of linear layers from dim_sizes[0] to dim_sizes[-1] """

    assert len(channel_sizes) > 1, 'require at least two channel sizes!!!'

    act_fns = cycle(act_fn) if type(act_fn) in (tuple, list) else cycle([act_fn])

    in_channels = channel_sizes[0]

    layers_order = ['norm', 'dropout']
    layers_order.insert(len(layers_order) if act_after_norm else 0, 'act')
    layers_order.insert(len(layers_order) if pre_act_norm else 0, 'linear')
        
    num_layers = len(channel_sizes) - 1
    block_layers = []
    for i in range(1, num_layers + 1):

        for j, layer_id in enumerate(layers_order):

            if layer_id == 'linear':

                # disable bias_attr if layernorm
                if bias_attr is None and norm_args['fn'] == 'layer' and (
                        (j+1 < len(layers_order) and layers_order[j+1] == 'norm') or
                        (j+1 == len(layers_order) and i < num_layers and layers_order[0] == 'norm')):
                    auto_bias_attr = False
                else:
                    auto_bias_attr = bias_attr
                
                block_layers.append(nn.Linear(in_channels, channel_sizes[i],
                    weight_attr=weight_attr, bias_attr=auto_bias_attr,
                    ))
                    # weight_attr=mi.ParamAttr(initializer=nn.initializer.XavierNormal())))
                in_channels = channel_sizes[i]

                # no act/norm/dropout... for the last layer if is_return and not pre_act_norm
                if is_return and i == num_layers:
                    break

            elif layer_id == 'act':
                block_layers.append(get_act_layer(next(act_fns)))

            elif layer_id == 'norm':
                block_layers.append(get_norm_layer(**norm_args, in_channels=in_channels, data_fmt=data_fmt))

            elif layer_id == 'dropout':
                block_layers.append(get_dropout_layer(dropout, data_fmt=data_fmt))

            else:
                logger.critical(f'Unrecognized layer id: {layer_id}!!!')

    if resnet and not is_return:
        assert resnet_beta is not None, 'resnet_beta is required!'
        assert channel_sizes[0] == in_channels, \
            f'ResNet ERROR! in_channels: {channel_sizes[0]} != out_channels: {channel_sizes[-1]}'
        block_layers.append(MyResNetLayer(beta=resnet_beta))

    return list(filter(None, block_layers))


def stack_lstm_block(channel_sizes, data_fmt=None, direction=None,
            act_fn=None, norm_args=dict(fn=None), pre_act_norm=False, act_after_norm=False,
            dropout=0.0, resnet=False, resnet_beta=None, is_return=False):
    """ return a list of lstm monolayers (forward or bidirect) from ndims[0] to ndims[-1] """

    # ======= Parameters
    assert len(channel_sizes) > 1, 'require at least two channel sizes!!!'
    assert isinstance(direction, str), 'direction is a required argument'

    direction = direction.lower()
    if direction == 'forward':
        num_directions = 1
    elif direction.startswith('bidirect'): # or bidirectional
        num_directions = 2
    else:
        logger.critical(f'Unknown direction: {direction} !!!')

    act_fns = cycle(act_fn) if type(act_fn) in (tuple, list) else cycle([act_fn])

    layers_order = ['norm', 'dropout']
    layers_order.insert(len(layers_order) if act_after_norm else 0, 'act')
    layers_order.insert(len(layers_order) if pre_act_norm else 0, 'lstm')

    # ======= Layers
    in_channels = channel_sizes[0]
    num_layers = len(channel_sizes) - 1
    block_layers = []
    for i in range(1, num_layers + 1):

        for j, layer_id in enumerate(layers_order):

            if layer_id == 'lstm':
                block_layers.append(nn.LSTM(
                            input_size = in_channels,
                            hidden_size = channel_sizes[i],
                            num_layers = 1,   # always one layer in one block, multiple layers are done in Tower
                            direction = direction,
                            dropout = 0.0, # this is inbetween layers, not needed as num_layers = 1
                            ))
                in_channels = channel_sizes[i] * num_directions

                if is_return and i == num_layers:
                    pass

            elif layer_id == 'act':
                block_layers.append(get_act_layer(next(act_fns)))

            elif layer_id == 'norm':
                block_layers.append(get_norm_layer(**norm_args, in_channels=in_channels, data_fmt=data_fmt))

            elif layer_id == 'dropout':
                block_layers.append(get_dropout_layer(dropout, data_fmt=data_fmt))

            else:
                logger.critical(f'Unrecognized layer id: {layer_id}!!!')

    if resnet and not is_return:
        if channel_sizes[0] != in_channels:
            logger.critical(f'ResNet ERROR! in_channels: {channel_sizes[0]} != out_dim[-1]: {in_channels}')
        assert resnet_beta is not None, 'resnet_beta is required!'
        block_layers.append(MyResNetLayer(beta=resnet_beta))

    return list(filter(None, block_layers))


def stack_conv1d_block(channel_sizes, data_fmt=None, kernel_size=None, stride=None, dilation=None,
            padding=None, padding_mode=None, max_pool=None, act_fn=None, norm_args=dict(fn=None),
            act_after_norm=False, pre_act_norm=False, dropout=0.0, resnet=False, resnet_beta=None,
            weight_attr=None, bias_attr=None, is_return=False):
    """ return a list of conv1d layers from nchannels[0] to nchannels[-1] """

    # ======= Parameters
    assert isinstance(data_fmt, str), 'data_fmt is a required argument!'
    assert (len(channel_sizes) - 1) == len(kernel_size) == len(stride) == len(dilation) == len(padding), \
        'channel_sizes-1, kernel_size, stride, dilation, and padding must have the same length!!!'

    data_fmt = data_fmt.upper()
    if max(padding) > 0 and padding_mode is None:
        padding_mode = 'zeros'
        logger.info(f'Use default padding mode: {padding_mode}')

    act_fns = cycle(act_fn) if type(act_fn) in (tuple, list) else cycle([act_fn])

    layers_order = ['norm', 'dropout']
    layers_order.insert(len(layers_order) if act_after_norm else 0, 'act')
    layers_order.insert(len(layers_order) if pre_act_norm else 0, 'conv1d_maxpool')

    # ======= Layers
    num_layers = len(channel_sizes) - 1
    block_layers = []
    in_channels = channel_sizes[0]
    for i in range(1, num_layers + 1):

        for j, layer_id in enumerate(layers_order):

            if layer_id == 'conv1d_maxpool':

                # disable bias_attr if layernorm
                if bias_attr is None and (max_pool is None or max_pool <= 1) and norm_args['fn'] == 'layer' and (
                        (j+1 < len(layers_order) and layers_order[j+1] == 'norm') or
                        (j+1 == len(layers_order) and i < num_layers and layers_order[0] == 'norm')):
                    auto_bias_attr = False
                else:
                    auto_bias_attr = bias_attr                

                block_layers.append(nn.Conv1D(
                        in_channels = in_channels,
                        out_channels = channel_sizes[i],
                        stride = int(stride[i - 1]),
                        kernel_size = int(kernel_size[i -1]),
                        dilation = int(dilation[i - 1]),
                        padding = int(padding[i - 1]),
                        padding_mode = padding_mode,
                        data_format = data_fmt,
                        weight_attr=weight_attr,
                        bias_attr=auto_bias_attr,
                        ))
                in_channels = channel_sizes[i]

                if max_pool is not None and max_pool > 1:
                    block_layers.append(nn.MaxPool1D(in_channels, stride=1, padding=max_pool // 2))

                if is_return and i == num_layers:
                    break

            elif layer_id == 'act':
                block_layers.append(get_act_layer(next(act_fns)))

            elif layer_id == 'norm':
                block_layers.append(get_norm_layer(**norm_args, in_channels=in_channels, data_fmt=data_fmt))

            elif layer_id == 'dropout':
                block_layers.append(get_dropout_layer(dropout, data_fmt=data_fmt))

            else:
                logger.critical(f'Unrecognized layer id: {layer_id}!!!')

    if resnet and not is_return:
        if channel_sizes[0] != channel_sizes[-1]:
            logger.critical(f'ResNet ERROR! in_channels: {channel_sizes[0]} != out_dim[-1]: {channel_sizes[-1]}')
        assert resnet_beta is not None, 'resnet_beta is required!'
        block_layers.append(MyResNetLayer(beta=resnet_beta))

    return list(filter(None, block_layers))


def stack_conv2d_block(channel_sizes, data_fmt=None, kernel_size=None, stride=None, dilation=None,
            padding=None, padding_mode=None, max_pool=None, act_fn=None, norm_args=dict(fn=None),
            act_after_norm=False, pre_act_norm=False, dropout=0.0,  resnet=False, resnet_beta=None,
            weight_attr=None, bias_attr=None, is_return=False):
    """ return a list of conv2d layers from nchannels[0] to nchannels[-1] """

    # ======= Parameters
    assert isinstance(data_fmt, str), 'data_fmt is a required argument!'
    assert (len(channel_sizes) - 1) == len(kernel_size) == len(stride) == len(dilation) == len(padding), \
        'channel_sizes-1, kernel_size, stride, dilation, and padding must have the same length!!!'

    data_fmt = data_fmt.upper()
    if max(padding) > 0 and padding_mode is None:
        padding_mode = 'zeros'
        logger.info(f'Use default padding mode: {padding_mode}')

    act_fns = cycle(act_fn) if type(act_fn) in (tuple, list) else cycle([act_fn])

    layers_order = ['norm', 'dropout']
    layers_order.insert(len(layers_order) if act_after_norm else 0, 'act')
    layers_order.insert(len(layers_order) if pre_act_norm else 0, 'conv2d_maxpool')

    # ======= Layers
    num_layers = len(channel_sizes) - 1
    block_layers = []
    in_channels = channel_sizes[0]
    for i in range(1, num_layers + 1):

        for j, layer_id in enumerate(layers_order):

            # disable bias_attr if layernorm
            if bias_attr is None and (max_pool is None or max_pool <= 1) and norm_args['fn'] == 'layer' and (
                    (j+1 < len(layers_order) and layers_order[j+1] == 'norm') or
                    (j+1 == len(layers_order) and i < num_layers and layers_order[0] == 'norm')):
                auto_bias_attr = False
            else:
                auto_bias_attr = bias_attr     

            if layer_id == 'conv2d_maxpool':
                block_layers.append(nn.Conv2D(
                        in_channels = in_channels,
                        out_channels = int(channel_sizes[i]),
                        kernel_size = int(kernel_size[i - 1]),
                        dilation = int(dilation[i - 1]),
                        stride = int(stride[i - 1]),
                        padding = int(padding[i - 1]),
                        padding_mode = padding_mode,
                        data_format = data_fmt,
                        weight_attr = weight_attr,
                        bias_attr = auto_bias_attr,
                        ))
                in_channels = channel_sizes[i]

                if max_pool is not None and max_pool > 1:
                    block_layers.append(nn.MaxPool2D(in_channels, stride=1, padding=max_pool // 2,
                            data_format=data_fmt))

                if is_return and i == num_layers:
                    break

            elif layer_id == 'act':
                block_layers.append(get_act_layer(next(act_fns)))

            elif layer_id == 'norm':
                block_layers.append(get_norm_layer(**norm_args, in_channels=in_channels, data_fmt=data_fmt))

            elif layer_id == 'dropout':
                block_layers.append(get_dropout_layer(dropout, data_fmt=data_fmt))

            else:
                logger.critical(f'Unrecognized layer id: {layer_id}!!!')

    if resnet and not is_return:
        if channel_sizes[0] != channel_sizes[-1]:
            logger.critical(f'ResNet ERROR! in_channels: {channel_sizes[0]} != out_dim[-1]: {channel_sizes[-1]}')
        assert resnet_beta is not None, 'resnet_beta is required!'
        block_layers.append(MyResNetLayer(beta=resnet_beta))

    return list(filter(None, block_layers))


class PosiEncoderxD(nn.Layer):
    """ Attempt to deal with """
    def __init__(self, in_channels, pe_dim=None, data_fmt=None, genre='trig', join_method='add',
                mlp_num=3, mlp_dim=None):
        super(PosiEncoderxD, self).__init__()

        self.in_channels = in_channels
        self.pe_dim = self.in_channels if pe_dim is None else pe_dim # final output PE dim

        self.genre = genre.upper() if genre else None
        self.join_method = join_method.upper()
        self.data_fmt = data_fmt.upper() if data_fmt else None
        self.mlp_num = mlp_num
        self.mlp_dim = self.pe_dim * 4 if mlp_dim is None else mlp_dim

        if self.genre == 'TRIG':
            self.pe_dim_init = self.pe_dim # not used yet for TRIG
        elif self.genre == 'ROTARY':
            self.pe_dim_init = self.pe_dim
            self.join_method = 'NA'
        elif self.genre == 'BERT':
            self.pe_dim_init = 111
            self.MLP = nn.Sequential(
                    nn.Linear(self.pe_dim_init, self.mlp_dim),
                    nn.ReLU(),
                    nn.Linear(self.mlp_dim, self.mlp_dim),
                    nn.ReLU(),
                    nn.Linear(self.mlp_dim, self.pe_dim),
                    nn.LayerNorm(self.pe_dim)
            )
        else:
            logger.critical(f'Unknown posenc curve type: {self.genre}!!!')

        self.register_buffer('pos_mat', mi.empty((0,0), dtype=mi.get_default_dtype()), persistable=False)

        logger.info(f'======= {self.__class__.__name__} =======')
        logger.info(f'            genre: {self.genre}')
        logger.info(f'         data_fmt: {self.data_fmt}')
        logger.info(f'           PE dim: {self.pe_dim}')
        logger.info(f'      join method: {self.join_method}')
        logger.info(f'      PE dim init: {self.pe_dim_init}')
        logger.info(f'          MLP dim: {self.mlp_dim}')
        logger.info(f'          MLP num: {self.mlp_num}')

        in_channels = self.in_channels

        if self.join_method == 'ADD':
            assert self.in_channels == self.pe_dim, 'In_channels must be equal to pe_dim for ADD'
        elif self.join_method == 'CONCAT':
            in_channels += self.pe_dim
        elif self.join_method == 'MULTIPLY':
            logger.warning('not yet implemented')
        elif self.join_method == 'NA':
            pass
        else:
            logger.critical(f'Unknown PosEncoder join method: {self.join_method}!!!')

        self.out_channels = in_channels

    def forward(self, x, seqs_len=None, beta=1.0):

        if self.genre == 'ROTARY':  # this is dealt in the ScaledDotProductAttention layer!!!
            return x

        if self.data_fmt == 'NCL': # convert to NLC
            x = x.transpose([0, 2, 1])
        elif self.data_fmt == 'NCHW': # convert to NHWC
            x = x.transpose([0, 2, 3, 1])
        else:
            pass

        # get the positional encoding matrix
        if self.genre in ['TRIG', 'ROTARY']:
            # note that pos_mat doesn't have the batch_size dim
            # create a new pos_mat if needed
            if self.pos_mat is None or x.ndim != self.pos_mat.ndim or \
                    np.any(np.array(x.shape[1:]) > np.array(self.pos_mat.shape[1:])):

                if x.ndim == 3:
                    self.pos_mat = mitas_utils.posencoder_trig_1d(x.shape[1:], curve=self.genre, mi=mi)
                elif x.ndim == 4:
                    self.pos_mat = mitas_utils.posencoder_trig_2d(x.shape[1:], curve=self.genre, mi=mi)
                else:
                    logger.critical(f'No position encoder for x.ndim={x.ndim}!')

                self.pos_mat = self.pos_mat.unsqueeze(0)

        elif self.genre == 'BERT':
            if x.ndim == 3:
                self.pos_mat = mitas_utils.posencoder_bert_1d(x.shape, seqs_len=seqs_len, mi=mi)
                self.pos_mat = self.MLP(self.pos_mat)
            else:
                logger.critical(f'Input must be 3D for BERT position encoding!!!')


        # apply position encoding
        if self.genre == 'ROTARY':
            if x.ndim == 3:
                sin, cos = self.pos_mat[:, :x.shape[-2], 0::2], self.pos_mat[:, :x.shape[-2], 1::2]
                x1, x2 = x[..., 0::2], x[..., 1::2]
                # this separates odd and even columns
                # x = mi.concat([x1*cos - x2*sin, x1*sin + x2*cos], axis=-1)
                # this restores the order
                x = mi.flatten(mi.stack([x1*cos - x2*sin, x1*sin + x2*cos], axis=-1), -2, -1)
            else:
                logger.warning(f'Rotary position encoder only works for x.ndim=3, not {x.ndim}!!!')

        elif self.join_method == 'ADD':
            # apply to x now with the batch dim
            if x.ndim == 3: # [NLC]
                if beta == 1.0:
                    x += self.pos_mat[:, :x.shape[-2], :x.shape[-1]]
                else:
                    x += beta * self.pos_mat[:, :x.shape[-2], :x.shape[-1]]
            elif x.ndim == 4: # [NHWC]
                if beta == 1.0:
                    x += self.pos_mat[:, :x.shape[-3], :x.shape[-2], :x.shape[-1]]
                else:
                    x += beta * self.pos_mat[:, :x.shape[-3], :x.shape[-2], :x.shape[-1]]
            else:
                logger.critical(f'failed to apply position encoder with ndim={x.ndim}!!!')

        elif self.join_method == 'CONCAT':
            if x.ndim == 3: # [NLC]
                if beta == 1.0:
                    pe = self.pos_mat[:, :x.shape[-2], :x.shape[-1]]
                else:
                    pe = beta * self.pos_mat[:, :x.shape[-2], :x.shape[-1]]
            elif x.ndim == 4: # [NHWC]
                if beta == 1.0:
                    pe = self.pos_mat[:, :x.shape[-3], :x.shape[-2], :x.shape[-1]]
                else:
                    pe = beta * self.pos_mat[:, :x.shape[-3], :x.shape[-2], :x.shape[-1]]
            else:
                logger.critical(f'failed to apply position encoder with ndim={x.ndim}!!!')

            x = mi.concat([x, pe], axis=-1)

        else:
            logger.error(f'Cannot apply the positional encoding to data!!!')

        if self.data_fmt == 'NCL': # convert to NLC
            x = x.transpose([0, 2, 1])
        elif self.data_fmt == 'NCHW': # convert to NHWC
            x = x.transpose([0, 3, 1, 2])
        else:
            pass

        return x


class AttentionMask(nn.Layer):
    """ return the Attention Mask for various types of attentions """
    def __init__(self, force_nlc=False, attn_method='paddle'):
        """ if force_nlc=True, matrix is reshaped to NLC """
        super(AttentionMask, self).__init__()
        self.force_nlc = force_nlc
        self.attn_method = attn_method.upper()
        self.register_buffer('mask_mat', mi.empty([0], dtype='float32'), persistable=False)
        self.empty_mask = mi.empty([0], dtype='float32') # np.empty(0)

    def forward(self, data_len, seqs_len=None):
        """ data_len is the L in NLC or [H, W] in NHWC """
        if seqs_len is None:
            return self.empty_mask
        if not hasattr(data_len, '__len__'):
            data_ndim = 1
            max_len = data_len
            data_len = [data_len]
        else:
            data_ndim = len(data_len)
            max_len = max(data_len)

        # no mask needed if max(data_len) is smaller than seqs_len
        if mi.all(seqs_len >= max_len):
            return self.empty_mask

        # TODO: currently assumes H=W=D for multi-dim data
        num_masks = len(seqs_len)

        with mi.no_grad():
            if self.attn_method in ['EFFICIENT']:
                # Efficient attention mask applies to Q, K, V separately
                if data_ndim == 1 or (data_ndim > 1 and not self.force_nlc):
                    # 1D attention or axial attention over the last data_len
                    self.mask_mat = mitas_utils.get_attn_mask_efficient_1d(data_len[-1], seqs_len,
                                attn_mask=self.mask_mat, tile_mask=data_len[0:-1], mi=mi)
                    num_masks *= math.prod(data_len[0:-1])
                elif data_ndim == 2 and self.force_nlc:
                    # 2D attention, though reshaped to NLC mid-way
                    self.mask_mat = mitas_utils.get_attn_mask_efficient_2d(data_len, seqs_len,
                            attn_mask=self.mask_mat, mi=mi)
                else:
                    logger.critical(f'Attn_method: {self.attn_method} or data_len: {data_len} not supported!')

            elif data_ndim == 1 or (data_ndim > 1 and not self.force_nlc):
                # 1D attention or axial attention over the last data_len
                self.mask_mat = mitas_utils.get_attn_mask_1d(data_len[-1], seqs_len,
                            attn_mask=self.mask_mat, tile_mask=data_len[:-1], mi=mi)
                num_masks *= math.prod(data_len[0:-1])
            elif data_ndim == 2 and self.force_nlc:
                # 2D Attention, though reshaped to NLC mid-way
                self.mask_mat = mitas_utils.get_attn_mask_2d(data_len, seqs_len,
                            attn_mask=self.mask_mat, mi=mi)
            else:
                logger.critical(f'Attn_method: {self.attn_method} or data_len: {data_len} not supported!')

        if self.mask_mat is None:
            return self.empty_mask
        else:
            return self.mask_mat[0 : num_masks]


class MatPaddingMaskOut(nn.Layer):
    """  """
    def __init__(self, data_fmt='NLC', dtype='float32', reverse=False):
        """ the default is to mask out (0.) padding unless reverse=True """
        super(MatPaddingMaskOut, self).__init__()
        self.dtype = dtype
        self.data_fmt = data_fmt.upper()
        self.register_buffer('mask_mat', mi.empty([0], dtype=self.dtype), persistable=False)
        self.mask_nil = np.empty(0, dtype=self.dtype)
        self.mask_fmt = self.data_fmt.replace('C', '')
        self.axes2add = [i for i, v in enumerate(self.data_fmt) if v == 'C']
        if len(self.mask_fmt) not in [2, 3]:
            logger.error(f'Unsupported data_fmt for MatMask: {self.data_fmt} !!!')

        self.reverse = reverse
        self.mask_in = 0.0 if self.reverse else 1.0
        self.mask_out = 1.0 - self.mask_in

    def forward(self, data_len, seqs_len=None):
        """ data_len is L in NLC and [H, W] for NHWC """
        if seqs_len is None: return self.mask_nil
        if not hasattr(data_len, '__len__'): data_len = [data_len]
        # no mask needed if max(data_len) is smaller than seqs_len
        # if np.array(seqs_len).min() >= max(data_len):
        if min(seqs_len) >= max(data_len):
            return self.mask_nil

        batch_size = len(seqs_len)
        new_shape = [batch_size, *data_len]
        # could use any(list) here, but ....
        if len(new_shape) != self.mask_mat.ndim or new_shape[0] > self.mask_mat.shape[0] or \
            new_shape[0] > self.mask_mat.shape[1] or new_shape[-1] > self.mask_mat.shape[-1]:
            logger.info(f'Generating new mask_mat with shape: {new_shape} ...')
            self.mask_mat = mi.full(new_shape, self.mask_out, dtype=self.dtype)
        else:
            self.mask_mat[:] = self.mask_out

        # assign values and return
        if self.mask_mat.ndim == 2:
            for i in range(batch_size):
                self.mask_mat[i, :seqs_len[i]] = self.mask_in
            return self.mask_mat[:batch_size, :data_len[0]].unsqueeze(self.axes2add)
        elif self.mask_mat.ndim == 3:
            for i in range(batch_size):
                self.mask_mat[i, :seqs_len[i], :seqs_len[i]] = self.mask_in
            return self.mask_mat[:batch_size, :data_len[0], :data_len[1]].unsqueeze(self.axes2add)
        else:
            logger.error(f'Not yet support mask_mat.ndim: {self.mask_mat.ndim}!!!')
            return self.mask_nil


class MatDiagonalMaskOut(nn.Layer):
    """  """
    def __init__(self, data_fmt='LL', dtype='float32', offset=0, reverse=False):
        """ the default is to mask out (0.) diagonal elements, unless reverse=True
            data_fmt: NLLC. Only used to unsqueeze the mask if needed
            diagonal: None or False -> No mask
                      an integer n -> the diagonal and n lower and upper neighbors
                      [n1, n2] -> the diagonal and n1 lower (i<j) and n2 upper neighbors
        """
        super(MatDiagonalMaskOut, self).__init__()
        self.dtype = dtype
        self.data_fmt = data_fmt.upper()
        self.register_buffer('mask_mat', mi.empty([0], dtype=self.dtype), persistable=False)
        self.mask_nil = np.empty(0, dtype=self.dtype)
        self.mask_fmt = self.data_fmt.replace('C', '').replace('N', '')
        self.axes2add = [i for i, v in enumerate(self.data_fmt) if v in ['C', 'N']]

        self.offset = offset
        if self.offset is not None:
            if not hasattr(self.offset, '__len__'):
                self.offset = [-self.offset, self.offset]
            if len(self.offset) == 1:
                self.offset = [-self.offset[0], self.offset[0]]
            if len(self.mask_fmt) != 2:
                logger.error(f'Cannot apply diagonal mask for data_fmt: {self.data_fmt}!!!')
                self.offset = None

        self.reverse = reverse
        self.mask_in = 0.0 if self.reverse else 1.0
        self.mask_out = 1.0 - self.mask_in

    def forward(self, data_shape):
        """ data_len [H, W] for NHWC """
        if self.offset is None: return self.mask_nil
        if not hasattr(data_shape, '__len__'): data_shape = [data_shape, data_shape]
        assert len(data_shape) == 2, f'data_shape: {data_shape} must have two elements!'

        if data_shape[0] > self.mask_mat.shape[0] or data_shape[1] > self.mask_mat.shape[1]:
            logger.info(f'Generating new mask_mat with shape: {data_shape} ...')
            self.mask_mat = mi.full(data_shape, self.mask_in, dtype=self.dtype)

            ij_delta = mi.linspace(1, data_shape[0], data_shape[0], dtype='int32').unsqueeze(0) - \
                       mi.linspace(1, data_shape[1], data_shape[1], dtype='int32').unsqueeze(1)

            self.mask_mat[mi.logical_and(ij_delta >= self.offset[0],
                                         ij_delta <= self.offset[1])] = self.mask_out
        if self.axes2add:
            return self.mask_mat[:data_shape[0], :data_shape[1]].unsqueeze(self.axes2add)
        else:
            return self.mask_mat[:data_shape[0], :data_shape[1]]


class MatReshape(nn.Layer):
    def __init__(self, out_fmt=None):
        super(MatReshape, self).__init__()
        self.out_fmt = out_fmt

    def forward(self, x):
        """ """
        return x.reshape(self.out_fmt)


class MatTranspose(nn.Layer):
    def __init__(self, perm=None, in_fmt='NLC', out_fmt='NCL'):
        """ NLLC will cause error, use NHWC instead (no repeated letters) """
        super(MatTranspose, self).__init__()
        if perm is None:
            assert len(in_fmt) == len(out_fmt), f'Must be the same ndim: {in_fmt} != {out_fmt}!!!'
            perm = [in_fmt.index(_x) for _x in out_fmt]

        self.perm = perm
        self.same_fmt = True if all(np.diff(self.perm) == 1) else False

    def forward(self, x):
        """ """
        if self.same_fmt:
            return x
        else:
            return x.transpose(self.perm)


class Seq2MatCastxD(nn.Layer):
    def __init__(self, method='concat', in_fmt='NCL', out_fmt='NCHW'):
        super(Seq2MatCastxD, self).__init__()

        self.method = method.upper()
        self.in_fmt = in_fmt.upper()
        self.out_fmt = out_fmt.upper()

        assert self.method in ['CONCAT', 'ADD', 'MULTIPLY'], f"Unknown method: {self.method}"
        assert self.in_fmt in ['NLC', 'NCL'], f"Unknown in_fmt: {self.in_fmt}!"
        assert self.out_fmt in ['NCHW', 'NHWC'], f"Unknown out_fmt: {self.out_fmt}!"

        dim_map = dict(CONCAT=2, ADD=1, MULTIPLY=1)
        self.mul_channels = dim_map.get(self.method, 1)

        logger.info(f'======= {self.__class__.__name__} =======')
        logger.info(f'           method: {self.method}')
        logger.info(f'           in_fmt: {self.in_fmt}')
        logger.info(f'          out_fmt: {self.out_fmt}')
        logger.info(f'     mul_channels: {self.mul_channels}')

    def forward(self, xh, xw):
        """ xh: [NCH], xw: [NCW], H and W can be different """
        if self.in_fmt == 'NCL':
            pass
        elif self.in_fmt == 'NLC':
            xh = xh.transpose([0, 2, 1])
            xw = xw.transpose([0, 2, 1])

        N, C, H = xh.shape
        N1, C1, W = xw.shape
        assert N == N1, f"Two matrices must have the same N: {N} != {N1}!"

        xh = xh.unsqueeze(3).expand((N, C, H, W))
        xw = xw.unsqueeze(2).expand((N, C1, H, W))

        if self.method.startswith('CONCAT'):
            x = mi.concat([xh, xw], axis=1) # --> [N, C+C1, L, L]
        elif self.method.startswith('ADD'):
            assert C == C1, f"Cannot add two matrices with different C: {C} != {C1}"
            x = xh + xw
        elif self.method.startswith('MULTIPLY'):
            assert C == C1, f"Cannot multiply two matrices with different C: {C} != {C1}"
            x = xh * xw

        if self.out_fmt == 'NCHW':
            pass
        elif self.out_fmt == 'NHWC':
            x = x.transpose([0, 2, 3, 1])

        return x


class AxisNorm(nn.Layer):
    def __init__(self, axis=-1, epsilon=1e-6):
        super(AxisNorm, self).__init__()
        self.axis = axis if hasattr(axis, '__len__') else [axis]
        self.epsilon = epsilon

    def forward(self, x):
        if len(self.axis) == 1: # use paddle implementation
            x -= mi.mean(x, axis=self.axis, keepdim=True)
            x /= mi.sqrt(mi.var(x, axis=self.axis, keepdim=True) + self.epsilon)
        else:
            norm_axes = self.axis
            norm_consts = mi.to_tensor(np.prod([x.shape[i] for i in self.axis]),
                        dtype='float32')
            mean = mi.sum(x, axis=norm_axes, keepdim=True) / norm_consts
            variance = mi.sum(x ** 2, axis=norm_axes, keepdim=True) / norm_consts - mean ** 2
            x = (x - mean) / mi.sqrt(variance + self.epsilon)

        return x


class MyNormLayer(nn.Layer):
    def __init__(self, fn='layer', mask=False, in_channels=None, data_fmt=None, axis=None,
                epsilon=1e-6, trainable=True, tracking_runing_stats=False, momentum=0.9):
        """ data_fmt: NLC/NCL """
        super(MyNormLayer, self).__init__()

        self.is_None = False
        self.fn = fn.lower()
        self.mask = mask
        # ======= Parameters
        assert isinstance(data_fmt, str), 'data_fmt is a required argument'
        self.in_channels = in_channels
        self.data_fmt = data_fmt.upper()
        self.data_ndim = len(self.data_fmt)
        self.trainable = trainable

        self.axis = [axis] if isinstance(axis, int) else axis
        if self.axis:
            self.axis = [i if i >= 0 else (i + self.data_ndim) for i in self.axis]

        normfn_kwargs = dict(
            weight_attr=None if self.trainable else False,
            bias_attr=None if self.trainable else False,
        )
        # ======= Masking: mostly my own normalization methods (slower than paddle version)
        # data_fmt will be converted to NLC or NHWC
        if self.mask:
            need_mask_isa = False  # check if we do need masking
            if self.fn.startswith(('batch', 'inst')):
                need_mask_isa = True
            elif self.fn.startswith('layer'):
                # layer norm for the last dim does not need mask
                # it is the same as axis norm with axis=-1
                if len(self.axis) == 1: # can only be the last axis if one axis
                    self.NormFn = nn.LayerNorm(in_channels, **normfn_kwargs)
                else:
                    need_mask_isa = True
                    trainable = False
            elif self.fn.startswith('axis'):
                # axis norm with axis=-1 does not need mask
                if len(self.axis) == 1 and (self.axis[0] == self.data_ndim - 1):
                    self.NormFn = AxisNorm(self.axis)
                else:
                    need_mask_isa = True
                    trainable = False
            else:
                self.is_None = True
                logger.critical(f'cannot recognize norm_fn: {self.fn}')

            if need_mask_isa:
                self.register_buffer('mask_mat', None, persistable=False)
                self.epsilon = epsilon

                self.track_running_stats = tracking_runing_stats
                self.momentum = momentum
                if self.track_running_stats:
                    self.running_mean = None
                    self.running_variance = None

                self.trainable = trainable
                if self.trainable:
                    self.gamma = mi.create_parameter([self.in_channels], 'float32',
                                default_initializer=nn.initializer.Constant(value=1.0))
                    self.beta = mi.create_parameter([self.in_channels], 'float32', is_bias=True)
                else:
                    logger.info('No affine transform for normalization...')

        # ======= No masking: ndim <=3
        elif self.data_ndim <= 3:
            if self.fn.startswith('batch'):
                # along [C] dim, normalize the [NL] 2D array. Supports: NLC, NCL, NC
                self.NormFn = nn.BatchNorm1D(in_channels, data_format=data_fmt, **normfn_kwargs)
            elif self.fn.startswith('inst'):
                # for each dim along [C], normalize the [L] 1D array (no normalization along N)
                # InstanceNorm2D will normalize for [HW] for each channel
                if data_fmt == 'NCL':
                    self.NormFn = nn.InstanceNorm1D(in_channels, data_format='NCL', **normfn_kwargs)
                else:
                    self.NormFn = nn.Sequential(
                        MatTranspose(in_fmt=data_fmt, out_fmt='NCL'),
                        nn.InstanceNorm1D(in_channels, data_format='NCL', **normfn_kwargs),
                        MatTranspose(in_fmt='NCL', out_fmt=data_fmt),
                        )
            elif self.fn.startswith('layer'):
                # normalize the ndarray specified by the passed shape, starting from the last dim
                # an integer will normalize the last dimension only
                # a shape af a two-element tuple will normalize the last two dims
                # the passed shape must match the shapes of the data, starting from the last dim
                if len(self.axis) == 1:
                    self.NormFn = nn.LayerNorm(in_channels, **normfn_kwargs)
                else:
                    self.NormFn = nn.LayerNorm(self.axis, **normfn_kwargs)
            elif self.fn.startswith('axis'):
                self.NormFn = AxisNorm(self.axis)
            else:
                self.is_None = True
                logger.warning(f'cannot recognize norm_fn: {self.fn}')

        # ======= No masking: ndim == 4
        elif self.data_ndim == 4:
            if self.fn.startswith('batch'):
                # for each dim along [C], norm_fn the [NHW] 3D array
                self.NormFn = nn.BatchNorm2D(in_channels, data_format=data_fmt, **normfn_kwargs)
            elif self.fn.startswith('inst'):
                if data_fmt == 'NCHW':
                    self.NormFn = nn.InstanceNorm2D(in_channels, data_format='NCHW', **normfn_kwargs)
                else:
                    self.NormFn = nn.Sequential(
                        MatTranspose(in_fmt=data_fmt, out_fmt='NCHW'),
                        nn.InstanceNorm2D(in_channels, data_format='NCHW', **normfn_kwargs),
                        MatTranspose(in_fmt='NCHW', out_fmt=data_fmt),
                    )
            elif self.fn.startswith('layer'):
                if len(self.axis) == 1:
                    self.NormFn = nn.LayerNorm(in_channels, **normfn_kwargs)
                else:
                    self.NormFn = nn.LayerNorm(self.axis, **normfn_kwargs)
            elif self.fn.startswith('axis'):
                self.NormFn = AxisNorm(self.axis)
            else:
                self.is_None = True
                logger.critical(f'cannot recognize norm_fn: {self.fn}')
        else:
            self.is_None = True
            logger.critical(f'Unable to initialize norm function with {self.fn}')


    def forward(self, x, seqs_len):
        # ======= No masking: paddle methods
        if hasattr(self, 'NormFn'): return self.NormFn(x)

        # ======= Convert to NLC, NHWC
        if self.data_fmt in ['NHWC', 'NLC']: #
            pass
        elif self.data_fmt == 'NCHW': # convert to NHWC
            x = x.transpose([0, 2, 3, 1])
        elif self.data_fmt == 'NCL': # convert to NLC
            x = x.transpose([0, 2, 1])
        else:
            logger.error(f'Uknown data_fmt: {self.data_fmt}!!!')

        # ======= Masking for ndim <=3
        if len(self.data_fmt) <= 3: # NLC (and NC?)
            batch_size, data_len, in_channels = x.shape

            # multiply mask if necessary
            if seqs_len is not None and min(seqs_len) < data_len:
                new_mask_shape = [batch_size, data_len, 1]

                if self.mask_mat is not None and list(self.mask_mat.shape) >= new_mask_shape:
                    self.mask_mat[:] = 1.0
                else:
                    self.mask_mat = mi.ones(new_mask_shape)

                for i in range(batch_size):
                    self.mask_mat[i, seqs_len[i]:, 0] = 0.0

                x *= self.mask_mat[:batch_size, :data_len, :]

            # seqs_len is given default value if not set
            if seqs_len is None:
                logger.warning(f'seqs_len ({seqs_len}) is required for masked norm!')
                seqs_len = mi.to_tensor([data_len] * batch_size)
            seqs_len = mi.to_tensor(seqs_len, dtype='float32') # must be a tensor

            if self.fn.startswith('batch') or \
                    (self.fn.startswith('axis') and self.axis == [0, 1]):
                norm_axes = [0, 1]
                norm_consts = mi.sum(seqs_len)
            elif self.fn.startswith('inst') or \
                    (self.fn.startswith('axis') and self.axis == [1]):
                norm_axes = [1]
                norm_consts = mi.unsqueeze(seqs_len, axis=[1, 2])
            elif self.fn.startswith('axis') and self.axis == [0]:
                norm_axes = [0]
                norm_consts = mi.sum(self.mask_mat[:batch_size, :data_len, :],
                    axis=0, keepdim=True) + self.epsilon
            elif self.fn.startswith('layer') and len(self.axis) == 2:
                # only if last two dims are normalized
                norm_axes = [1, 2]
                norm_consts = mi.unsqueeze(in_channels * seqs_len, axis=[1, 2])
            else:
                logger.critical(f'Unknown norm: {self.fn} axis: {self.axis} with masking!!!')
        # ======= Masking for ndim==4
        elif len(self.data_fmt) == 4: # NCHW or NHWC
            batch_size, height, width, in_channels = x.shape

            # only support one seq_len for both H and W
            if seqs_len is not None and (min(seqs_len) < height or min(seqs_len) < width):
                new_mask_shape = [batch_size, height, width, 1]
                if self.mask_mat is not None and self.mask_mat.shape >= new_mask_shape:
                    self.mask_mat[:] = 1.0
                else:
                    self.mask_mat = mi.ones(new_mask_shape)

                for i in range(batch_size):
                    if seqs_len[i] < height:
                        self.mask_mat[i, seqs_len[i]:, :, 0] = 0.0
                    if seqs_len[i] < width:
                        self.mask_mat[i, :, seqs_len[i]:, 0] = 0.0

                x *= self.mask_mat[:batch_size, :height, :width, :]

            # seqs_len is given default value if not set
            if seqs_len is None:
                logger.warning(f'seqs_len ({seqs_len}) is required for masked norm!')
                seqs_len = mi.to_tensor([max([height, width])] * batch_size)
            seqs_len = mi.to_tensor(seqs_len, dtype='float32') # must be a tensor

            if self.fn.startswith('batch') or \
                    (self.fn.startswith('axis') and self.axis == [0, 1, 2]):
                norm_axes = [0, 1, 2]
                norm_consts = mi.sum(seqs_len ** 2)
            elif self.fn.startswith('inst') or \
                    (self.fn.startswith('axis') and self.axis == [1, 2]):
                norm_axes = [1, 2]
                norm_consts = mi.unsqueeze(seqs_len ** 2, axis=[1, 2, 3])
            elif self.fn.startswith('axis') and self.axis == [0]:
                norm_axes = [0]
                norm_consts = mi.sum(self.mask_mat[:batch_size, :height, :width, :],
                                    axis=0, keepdim=True) + self.epsilon
            elif self.fn.startswith('axis') and self.axis == [1]:
                norm_axes = [1]
                norm_consts = mi.unsqueeze(seqs_len, axis=[1, 2, 3])
            elif self.fn.startswith('axis') and self.axis == [2]:
                norm_axes = [2]
                norm_consts = mi.unsqueeze(seqs_len, axis=[1, 2, 3])
            elif self.fn.startswith('layer') and len(self.axis) == 2:
                norm_axes = [2, 3]
                norm_consts = mi.unsqueeze(in_channels * seqs_len, axis=[1, 2, 3])
            elif self.fn.startswith('layer') and len(self.axis) == 3:
                norm_axes = [1, 2, 3]
                norm_consts = mi.unsqueeze(in_channels * (seqs_len ** 2), axis=[1, 2, 3])
            else:
                logger.critical(f'Unsupported method {self.fn} with masking!!!')
        else:
            logger.critical(f'Unsupported data_fmt: {self.data_fmt}!!!')

        # ======= Apply normalization
        mean = mi.sum(x, axis=norm_axes, keepdim=True) / norm_consts
        variance = mi.sum(x ** 2, axis=norm_axes, keepdim=True) / norm_consts - mean ** 2
        x = (x - mean) / mi.sqrt(variance + self.epsilon)
        if self.trainable:
            x = x * self.gamma + self.beta

        if self.training and self.track_running_stats:
            if self.running_mean is None:
                self.running_mean = mean
                self.running_variance = variance
            else:
                self.running_mean = (1 - self.momentum) * self.running_mean \
                            + self.momentum * mean
                self.running_variance = (1 - self.momentum) * self.running_variance \
                            + self.momentum * variance

        # ======= Convert fmt back to input
        if self.data_fmt in ['NLC', 'NHWC']:
            pass
        elif self.data_fmt == 'NCHW': # convert to NCHW
            x = x.transpose([0, 3, 2, 1])
        elif self.data_fmt == 'NCL': # convert to NLC
            x = x.transpose([0, 2, 1])
        else:
            logger.error(f'Uknown data_fmt: {self.data_fmt}!!!')

        return x

    def debug(self):
        pass


class MyResNetLayer(nn.Layer):
    """ name holder only, addition is done at the tower/net levels """
    def __init__(self, beta=1.0):
        super(MyResNetLayer, self).__init__()
        self.beta = beta

    def forward(self, x, **kwargs):
        if self.beta == 1.0:
            return x
        elif self.beta == 0.0:
            logger.critical('Are you sure beta=0.0 for resnet!!!')
            return 0
        else:
            return self.beta * x


class ScaledDotProductAttention(nn.Layer):
    ''' Scaled Dot-Product Attention
        Adopted from https://github.com/jadore801120/attention-is-all-you-need-pytorch

        Note1: no trainable parameters!
        Note2: only the last two dims are operated on. Any ndim>=2 should work!
    '''

    def __init__(self, scale=None, act_fn='softmax', dropout=0.1, posenc=None, return_attn=True):
        """ here scale is sqrt(channel_dim)
            the only posenc implemented is the rotary position encoding
        """
        super().__init__()

        self.scale = None if scale is None else float(scale)
        self.act_fn = act_fn.upper()
        self.dropout = dropout
        self.posenc = posenc
        self.return_attn = return_attn

        if self.posenc is not None and self.posenc.upper() == 'ROTARY':
            self.register_buffer("sin", mi.empty((0,0)), persistable=False)
            self.register_buffer("cos", mi.empty((0,0)), persistable=False)

        if self.act_fn == 'SOFTMAX':
            self.Activate = nn.Softmax(axis=-1)
        else:
            logger.critical(f'Unrecognized act_fn: {self.act_fn} ...')

        self.Dropout = nn.Dropout(self.dropout)

    def generate_posenc_rotary(self, in_channels, seq_len=600, base=10000):
        logger.info(f'Generating new {self.posenc} matrices with base: {base}, in_channels: {in_channels}, seq_len:{seq_len} ...')
        omega_k = mi.exp(-math.log(base) / in_channels * (mi.arange(0, in_channels, 2, dtype=mi.get_default_dtype())))
        j = mi.arange(seq_len, dtype=mi.get_default_dtype())
        omega_jk = mi.matmul(j.unsqueeze(1), omega_k.unsqueeze(0))
        self.sin = omega_jk.sin()
        self.cos = omega_jk.cos()
        # self.register_buffer("sin", omega_jk.sin(), persistable=False)
        # self.register_buffer("cos", omega_jk.cos(), persistable=False)

    def apply_posenc_rotary(self, x, offset=0):
        seq_len = x.shape[-2]
        enc_len = int(x.shape[-1] / 2)
        if len(self.sin) < seq_len or self.sin.shape[-1] < enc_len:
            self.generate_posenc_rotary(x.shape[-1], seq_len=seq_len)

        sin, cos = (self.sin[offset:offset+seq_len, :enc_len],
                    self.cos[offset:offset+seq_len, :enc_len],)

        x1, x2 = x[..., 0::2], x[..., 1::2]

        # [cos_nθ, -sin_nθ] [x1]
        # [sin_nθ,  cos_nθ] [x2]
        # => [x1 * cos_nθ - x2 * sin_nθ, x1 * sin_nθ + x2 * cos_nθ]
        # Note that even and odd channels are separated!!!
        return mi.concat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], axis=-1)

    def forward(self, q, k, v, mask=None):
        """ q: [N, nhead, len_q, dim_q]
            k: [N, nhead, len_k, dim_k]
            v: [n, nhead, len_v, dim_v]
            mask: [len_q, len_k] matrix, 0 for selection -np.inf for exclusion
        Conditions:
            dim_q == dim_k
            len_k == len_v
        """

        # apply rotary position encoding
        if self.posenc is not None and self.posenc.upper() == 'ROTARY':
            q = self.apply_posenc_rotary(q)
            k = self.apply_posenc_rotary(k)

        # [N, nhead, len_q, dim_q] x [N, nhead, dim_k, len_k] ->  [N, nhead, len_q, len_k]
        # Only need to transpose the last two dimensions
        new_axes = list(range(k.ndim - 2)) + [k.ndim - 1, k.ndim -2] # [0, 1, 3, 2]
        if self.scale is None or self.scale == 1.0:
            attn_mat = mi.matmul(q, k.transpose(new_axes))
        else:
            attn_mat = mi.matmul(q / self.scale, k.transpose(new_axes))

        # Mask is 0 for selection, -np.inf for exclusion
        if mask is not None and len(mask):
            attn_mat += mask
            # attn = attn.masked_fill(mask == 0, -1e9)

        attn_mat = self.Activate(attn_mat)
        attn_mat = self.Dropout(attn_mat)

        # [N, nhead, len_q, len_k] x [N, nhead, len_v, dim_v] -> [N, nhead, len_q, dim_v]
        output = mi.matmul(attn_mat, v)

        # output: [N, nhead, len_q, dim_v], attn: [N, nhead, len_q, len_k]
        if self.return_attn:
            return output, attn_mat
        else:
            return output


class ScaledEfficientAttention(nn.Layer):
    ''' Scaled Efficient Attention

        Note1: no trainable parameters!
        Note2: only the last two dimensions are operated on. ndim can be 3, 4, 5, etc.
    '''

    def __init__(self, scale=None, act_fn='softmax', dropout=0.1, posenc=None, return_attn=True):
        """ """
        super().__init__()

        self.scale = scale
        if self.scale is not None:
            self.scale = math.sqrt(float(self.scale))

        self.act_fn = act_fn.upper()
        self.dropout = dropout

        self.return_attn = return_attn

        if self.act_fn == 'SOFTMAX':
            self.ActQ = nn.Softmax(axis=-1)
            self.ActK = nn.Softmax(axis=-2)
        else:
            logger.critical(f'Unrecognized act_fn: {self.act_fn} ...')

        self.Dropout = nn.Dropout(self.dropout)

    def forward(self, q, k, v, mask=None):
        """ q: [N, nhead, len_q, dim_q]
            k: [N, nhead, len_k, dim_k]
            v: [n, nhead, len_v, dim_v]
            mask: [len_q, len_k] matrix, 0 for selection, -np.inf for exclusion
        Conditions:
            dim_q == dim_k
            len_k == len_v
        """

        if self.scale is not None and self.scale != 1.0:
            q /= self.scale
            k /= self.scale

        if mask is not None and len(mask):
            # we can leave q alone as softmax is over the channel dim
            # also leave v alone as it is multiplied by softmax(k)
            k += mask

        # [N, nhead, len_k, dim_k] x [N, nhead, len_v, dim_v] ->  [N, nhead, dim_k, len_k]
        # Only need to transpose the last two dimensions
        new_axes = list(range(k.ndim - 2)) + [k.ndim - 1, k.ndim - 2] # [0, 1, 3, 2]
        template = mi.matmul(self.ActK(k).transpose(new_axes), v)
        # attn = mi.matmul(q / self.temperature, k.transpose(new_axes))

        # [N, nhead, len_q, dim_q] x [N, nhead, dim_k, dim_v] -> [N, nhead, len_q, dim_v]
        output = mi.matmul(self.Dropout(self.ActQ(q)), template)

        # output: [N, nhead, len_q, dim_v], attn: [N, nhead, dim_k, dim_v]
        if self.return_attn:
            return output, template
        else:
            return output


class MyMultiHeadAttention(nn.Layer):
    ''' Multi-Head Attention module
        Adopted from https://github.com/jadore801120/attention-is-all-you-need-pytorch
    '''

    def __init__(self, q_dim, k_dim, v_dim, nhead=2, temperature=None, posenc=None,
                    method='dotproduct', act_fn='softmax', dropout=0.1):
        """ only the first (in_channels) and last (out_channels) of q_dim, k_dim, v_dim are used.
        For q, k, and v, a fc layer is first applied from dim[0] to dim[-1].
        The dims can be scalars giving dim[0]=dim[-1]

        Conditions:
            q_dim[-1] == k_dim[-1]

        """
        super(MyMultiHeadAttention, self).__init__()

        q_dim = q_dim if hasattr(q_dim, '__len__') else [q_dim]
        k_dim = k_dim if hasattr(k_dim, '__len__') else [k_dim]
        v_dim = v_dim if hasattr(v_dim, '__len__') else [v_dim]

        assert q_dim[-1] == k_dim[-1], f"query dim: {q_dim[-1]} != key dim: {k_dim[-1]}"
        assert len(q_dim) < 3 and len(k_dim) < 3 and len(v_dim) < 3, '<3 dims for each input'

        self.q_dim, self.k_dim, self.v_dim = q_dim, k_dim, v_dim
        self.nhead = int(nhead)

        assert self.q_dim[-1] % self.nhead == 0 and self.k_dim[-1] % self.nhead == 0 and \
               self.v_dim[-1] % self.nhead == 0, \
               f"Q/K/V dims must be a multiple of nhead={self.nhead}!!!"

        # sqrt(num_channels) so as to set variance of q*k == 1
        self.temperature = 1.0 if temperature is None else float(temperature)
        self.scale = math.sqrt(q_dim[-1]) * self.temperature
        self.method = method.upper()
        self.act_fn = act_fn
        self.dropout = float(dropout)

        # weight_attr = mi.ParamAttr(name=None, initializer=nn.initializer.Normal(mean=0.0, std=2.0))

        self.Linear2Q = nn.Linear(q_dim[0], q_dim[-1], bias_attr=None, #False,
                    weight_attr=mi.ParamAttr(initializer=nn.initializer.Normal(
                                mean=0.0, std=np.sqrt(2.0 / (q_dim[0] + q_dim[-1])))))

        self.Linear2K = nn.Linear(k_dim[0], k_dim[-1], bias_attr=None, # False,
                    weight_attr=mi.ParamAttr(initializer=nn.initializer.Normal(
                                mean=0.0, std=np.sqrt(2.0 / (k_dim[0] + k_dim[-1])))))

        self.Linear2V = nn.Linear(v_dim[0], v_dim[-1], bias_attr=None, # False,
                    weight_attr=mi.ParamAttr(initializer=nn.initializer.Normal(
                                mean=0.0, std=np.sqrt(2.0 / (v_dim[0] + v_dim[-1])))))

        # nn.init.normal_(self.q_linear.weight, mean=0, std=np.sqrt(2.0 / (q_dim[0] + q_dim[-1])))
        # nn.init.normal_(self.k_linear.weight, mean=0, std=np.sqrt(2.0 / (k_dim[0] + k_dim[-1])))
        # nn.init.normal_(self.v_linear.weight, mean=0, std=np.sqrt(2.0 / (v_dim[0] + v_dim[-1])))

        if self.method in ['DOTPRODUCT', 'DOT_PRODUCT']:
            self.scale *= np.power(q_dim[-1], 0.5)
            self.Attend = ScaledDotProductAttention(scale=self.scale,
                        act_fn=self.act_fn, dropout=self.dropout, posenc=posenc)
        elif self.method in ['EFFICIENT']:
            self.Attend = ScaledEfficientAttention(scale=self.scale,
                        act_fn=self.act_fn, dropout=self.dropout, posenc=posenc)
        else:
            logger.critical(f'Unknown attn_method: {self.method}')

        self.LinearZ = nn.Linear(v_dim[-1], q_dim[-1], bias_attr=None, # False,
                    weight_attr=mi.ParamAttr(initializer=nn.initializer.XavierNormal()))
        # nn.init.xavier_normal_(self.fc.weight)

        self.out_channels = q_dim[-1]

    def forward(self, q, k=None, v=None, mask=None):
        """  """
        if k is None: k = q
        if v is None: v = k

        q_dim, k_dim, v_dim, nhead = self.q_dim[-1], self.k_dim[-1], self.v_dim[-1], self.nhead
        batch_size, q_len, k_len, v_len = q.shape[0], q.shape[1], k.shape[1], v.shape[1]

        assert q.ndim == k.ndim == v.ndim == 3, 'q, k, v must be of NLC format'
        assert q_dim == k_dim, "query and key must be of the same dimension"
        assert k_len == v_len, "key and value must be of the same length"

        q = self.Linear2Q(q).reshape([batch_size, q_len, nhead, -1])
        k = self.Linear2K(k).reshape([batch_size, k_len, nhead, -1])
        v = self.Linear2V(v).reshape([batch_size, v_len, nhead, -1])

        # Transpose to [N, nhead, len, dim]
        q = q.transpose([0, 2, 1, 3])
        k = k.transpose([0, 2, 1, 3])
        v = v.transpose([0, 2, 1, 3])

        # if mask is not None:
            # mask = mask.unsqueeze(1)   # For head axis broadcasting

        v4q, attn_mat = self.Attend(q, k, v, mask=mask)
        # Transpose to move the head dimension back: [N, q_len, nhead, v_dim]
        v4q = v4q.transpose([0, 2, 1, 3]).reshape([batch_size, q_len, -1])

        v4q = self.LinearZ(v4q)

        return v4q


class MyAttnxDLayer(nn.Layer):
    def __init__(self, in_channels, nhead=2, axis=1, axis_share_attn=True,
                data_fmt='NLC', temperature=None,
                norm_fn='layer', norm_before=False, attn_posenc=None,
                attn_method='dotproduct', attn_act_fn='softmax', attn_dropout=None,
                join_method='add', dropout=0.1,
                ff_dim=None, ff_act_fn='relu', ff_dropout=None, ff_norm_fn=None,
                ):
        """ A transformer-like encoder layer consisting of the following:
                1) Home-brew multi-head multi-axis attention
                2) Customizable join method: add (default), concat
                3) Feedforward layer with act/norm/dropout
                4) Attend over multiple axes by adding the attended values
                5) Temperature is used to modulate the scaled product
        """
        super(MyAttnxDLayer, self).__init__()

        self.in_channels = int(in_channels)
        self.data_fmt = data_fmt.upper() # only used for normalization layers
        self.norm_fn = norm_fn
        self.norm_before = norm_before
        self.dropout = float(dropout)

        # Layer 1: self-attention
        self.nhead = int(nhead)
        self.axis = axis if hasattr(axis, '__len__') else [axis]
        self.axis_share_attn = axis_share_attn
        self.temperature = 1.0 if temperature is None else temperature

        self.attn_posenc = attn_posenc
        self.attn_method = attn_method.upper()
        self.attn_act_fn = attn_act_fn
        self.attn_dropout = self.dropout if attn_dropout is None else float(attn_dropout)

        # Layer 2: feedforward
        self.join_method = join_method.upper()

        self.ff_dim = self.in_channels * 4 if ff_dim is None else int(ff_dim)
        self.ff_act_fn = ff_act_fn
        self.ff_dropout = self.dropout if ff_dropout is None else ff_dropout
        self.ff_norm_fn = norm_fn if ff_norm_fn is None else ff_norm_fn

        # Set up layers
        in_channels = self.in_channels

        # Self-attention layer(s)
        self.AttnLayers = []
        for i, iL in enumerate(self.axis):
            if self.axis_share_attn and i > 0:
                self.AttnLayers.append(self.AttnLayers[0])
                continue
            self.AttnLayers.append(MyMultiHeadAttention(
                        q_dim=in_channels,
                        k_dim=in_channels,
                        v_dim=in_channels,
                        nhead=self.nhead,
                        temperature=self.temperature,
                        posenc=self.attn_posenc,
                        method=self.attn_method,
                        act_fn=self.attn_act_fn,
                        dropout=self.attn_dropout,
                        ))
            self.add_sublayer(f'Attn{i}', self.AttnLayers[-1])

        # Dropout for attention layer output before join and norm
        self.AttnDropout = nn.Dropout(self.dropout)

        # Join input and attention
        if self.join_method.startswith('ADD'):
            # assert q_dim[-1] == v_dim[-1], "attn_mode: add requires the same q and v dimensions"
            # assert q_dim[0] == q_dim[-1], "attn_mode: add requires the same q dimensions"
            # assert in_channels == out_channels, \
                        # "Attn method: add requires the same dimensions for input and output "
            self.AttnNorm = get_norm_layer(in_channels=in_channels, data_fmt=self.data_fmt,
                        fn=self.norm_fn, axis=[-1], mask=False)
        elif self.join_method.startswith('CONCAT'): # no Norm in this case
            in_channels += in_channels
        else:
            logger.critical(f'Unsupported attention join method: {self.join_method}')

        if self.norm_before:
            self.FFNorm1 = get_norm_layer(in_channels=in_channels, data_fmt=self.data_fmt,
                        fn=self.ff_norm_fn, axis=[-1], mask=False)

        self.Attn2FF = nn.Linear(in_channels, self.ff_dim, bias_attr=None, # False,
                    weight_attr=mi.ParamAttr(initializer=nn.initializer.XavierNormal()))
        in_channels = self.ff_dim

        self.FFActivate = get_act_layer(self.ff_act_fn)
        # self.FFNorm2 = get_norm_layer(in_channels=in_channels, data_fmt=self.data_fmt,
        #       fn=self.ff_norm_fn, axis=[-1], mask=False)
        self.FFDropout = nn.Dropout(self.ff_dropout)

        self.FF2Out = nn.Linear(in_channels, self.in_channels, bias_attr=None, # False,
                    weight_attr=mi.ParamAttr(initializer=nn.initializer.XavierNormal()))
        in_channels = self.in_channels

        self.OutDropout = nn.Dropout(self.ff_dropout)
        if not self.norm_before:
            self.FFNorm1 = get_norm_layer(in_channels=in_channels, data_fmt=self.data_fmt,
                    fn=self.ff_norm_fn, axis=[-1], mask=False)

        self.out_channels = in_channels

    def attention_qqq(self, q, mask=None):
        """ self-attention of a single input, q=k=v """
        q0_ndim = q.ndim
        q_attn_out = None
        for i, iL in enumerate(self.axis):
            # permute to [..., L, C] to perform attention over L dim
            # only deal with cases ndim > 3 and not axis!=ndim-2
            if q0_ndim > 3 and ((iL != q0_ndim - 2) or (iL != -2)):
                perm = list(range(q0_ndim))
                perm.insert(-1, iL)
                perm.pop(iL)
                q = q.transpose(perm)

            # reshape to NLC for 4D and higher
            if q0_ndim > 3:
                q_shape = q.shape
                q = q.reshape([-1, q.shape[-2], q.shape[-1]])

            # x_attn is used to store the passed value to query
            q_attn = self.AttnLayers[i](q, mask=mask)

            # restore shape for 4D and higher
            if q0_ndim > 3:
                q = q.reshape(q_shape)
                q_attn = q_attn.reshape(q_shape)

            # restore permutation to original
            if q0_ndim > 3 and ((iL != q0_ndim - 2) or (iL != -2)):
                perm = list(range(q0_ndim))
                perm.insert(iL, q0_ndim - 2)
                perm.pop(-2)
                q = q.transpose(perm)
                q_attn = q_attn.transpose(perm)

            if q_attn_out is None:
                q_attn_out = q_attn
            else:
                q_attn_out += q_attn

        return self.AttnDropout(q_attn_out)

    def attention_qkk(self, q, k, mask=None):
        """ cross-attention between q and k=v """
        assert q.ndim == k.ndim, 'q.ndim != k.ndim'

        q0_ndim = q.ndim
        q_attn_out = None
        for i, iL in enumerate(self.axis):
            # permute to [..., L, C] to perform attention over L dim
            # only deal with cases ndim > 3 and not axis!=ndim-2
            if q0_ndim > 3 and ((iL != q0_ndim - 2) or (iL != -2)):
                perm = list(range(q0_ndim))
                perm.insert(-1, iL)
                perm.pop(iL)
                q = q.transpose(perm)
                k = k.transpose(perm)

            # reshape to NLC for 4D and higher
            if q0_ndim > 3:
                q_shape = q.shape
                k_shape = k.shape
                q = q.reshape([-1, q.shape[-2], q.shape[-1]])
                k = k.reshape([-1, k.shape[-2], k.shape[-1]])

            # x_attn is used to store the passed value to query
            q_attn = self.AttnLayers[i](q, k, k, mask=mask)

            # restore shape for 4D and higher
            if q0_ndim > 3:
                q = q.reshape(q_shape)
                q_attn = q_attn.reshape(q_shape)
                k = k.reshape(k_shape)

            # restore permutation to original
            if q0_ndim > 3 and ((iL != q0_ndim - 2) or (iL != -2)):
                perm = list(range(q0_ndim))
                perm.insert(iL, q0_ndim - 2)
                perm.pop(-2)
                q = q.transpose(perm)
                q_attn = q_attn.transpose(perm)
                k = k.transpose(perm)

            if q_attn_out is None:
                q_attn_out = q_attn
            else:
                q_attn_out += q_attn

        return self.AttnDropout(q_attn_out)

    def attention_qkv(self, q, k, v, mask=None):
        """ cross-attention between q, k, and v """
        assert q.ndim == k.ndim == v.ndim, 'q.ndim == k.ndim == v.ndim'

        q0_ndim = q.ndim
        q_attn_out = None
        for i, iL in enumerate(self.axis):
            # permute to [..., L, C] to perform attention over L dim
            # only deal with cases ndim > 3 and not axis!=ndim-2
            if q0_ndim > 3 and ((iL != q0_ndim - 2) or (iL != -2)):
                perm = list(range(q0_ndim))
                perm.insert(-1, iL)
                perm.pop(iL)
                q = q.transpose(perm)
                k = k.transpose(perm)
                v = v.transpose(perm)

            # reshape to NLC for 4D and higher
            if q0_ndim > 3:
                q_shape = q.shape
                k_shape = k.shape
                v_shape = v.shape
                q = q.reshape([-1, q.shape[-2], q.shape[-1]])
                k = k.reshape([-1, k.shape[-2], k.shape[-1]])
                v = v.reshape([-1, v.shape[-2], v.shape[-1]])

            # x_attn is used to store the passed value to query
            q_attn = self.AttnLayers[i](q, k, v, mask=mask)

            # restore shape for 4D and higher
            if q0_ndim > 3:
                q = q.reshape(q_shape)
                q_attn = q_attn.reshape(q_shape)
                k = k.reshape(k_shape)
                v = v.reshape(v_shape)

            # restore permutation to original
            if q0_ndim > 3 and ((iL != q0_ndim - 2) or (iL != -2)):
                perm = list(range(q0_ndim))
                perm.insert(iL, q0_ndim - 2)
                perm.pop(-2)
                q = q.transpose(perm)
                q_attn = q_attn.transpose(perm)
                k = k.transpose(perm)
                v = v.transpose(perm)

            if q_attn_out is None:
                q_attn_out = q_attn
            else:
                q_attn_out += q_attn

        return self.AttnDropout(q_attn_out)

    def forward(self, q, k=None, v=None, mask=None):
        """ the input.ndim can be any number >=3
            all channel_sizes must be the same (unable to stack layers if otherwise)
        """

        if self.norm_before:
            q = self.AttnNorm(q, seqs_len=None)

        # Self-Attention layer
        if k is None and v is None:
            q_attn_out = self.attention_qqq(q, mask=mask)
        elif v is None:
            q_attn_out = self.attention_qkk(q, k, mask=mask)
        elif k is None:
            q_attn_out = 0.0
            logger.warning('not yet implemented')
        else:
            q_attn_out = self.attention_qkv(q, k, v, mask=mask)

        # join
        if self.join_method.startswith('ADD'):
            q += q_attn_out
            if not self.norm_before:
                q_attn_out = self.AttnNorm(q_attn_out, seqs_len=None)
        elif self.join_method.startswith('CONCAT'):
            q = mi.concat((q, q_attn_out), axis=-1)
        else:
            logger.critical(f'Unsupported join_method: {self.join_method}')

        # Feedforward layer
        if self.norm_before:
            q = self.FFNorm1(q, seqs_len=None)

        q0 = q

        # not sure whether this sequence is optimal or even correct
        q = self.Attn2FF(q)
        q = self.FFActivate(q)
        # x = self.FFNorm2(x, seqs_len=None) # paddlepaddle does not use this
        q = self.FFDropout(q)
        q = self.FF2Out(q)
        q = self.FFActivate(q)
        q = self.OutDropout(q)

        q += q0

        if not self.norm_before:
            q = self.FFNorm1(q, seqs_len=None)
        return q


class MyEmbdxDLayer(nn.Layer):
    """ """
    def __init__(self, args, in_channels=None, in_dtype=None, out_dtype=None, padding_idx=0,
                data_fmt=None, out_fmt=None):
        super(MyEmbdxDLayer, self).__init__()
        set_global_initializer(args.param_init)
        # nn.initializer.set_global_initializer(nn.initializer.KaimingNormal(), nn.initializer.Constant(0.0))

        # ======= Arguments
        self.data_fmt = args.input_fmt.upper() if data_fmt is None else data_fmt.upper()
        if self.data_fmt not in ['NC', 'NL', 'NCL', 'NLC',
            'NHW', 'NLL', 'NHWC', 'NCHW', 'NLLC', 'NCLL']:
            logger.critical(f'Unsupported data fmt: {self.data_fmt}')

        self.data_genre = args.input_genre
        # self.data_ndim = int(args.input_ndim)

        # input will be checked against self.in_dtype
        self.in_dtype = 'float32' if in_dtype is None else in_dtype.lower()
        self.in_channels = args.input_dim if in_channels is None else in_channels
        self.in_channels = int(self.in_channels)

        # get default input_shape for mi.summary() calls
        fmt2len_dict = {'N' : 2, 'H' : 16, 'W': 16, 'L' : 16, 'C': self.in_channels}
        x_shape = tuple(fmt2len_dict[_s] for _s in self.data_fmt)
        # self.in_shapes = [x_shape, (len_dict['N'], 1)] #(x, seqs_len)
        self.in_shapes = [InputSpec(shape=x_shape, dtype='float32', name='x')]
        if 'bpmat' in args.input_genre:
            self.in_shapes.append(
                InputSpec(shape=(fmt2len_dict['N'], fmt2len_dict['H'], fmt2len_dict['W'], 4),
                    dtype='float32', name='bpmat')
                )
        self.in_shapes.append(
            InputSpec(shape=(fmt2len_dict['N'],), dtype='int32', name='seqs_len')
            )

        self.embed_fn = args.embed_fn.lower() if isinstance(args.embed_fn, str) else None
        self.embed_dim = 8 if args.embed_dim is None else int(args.embed_dim)
        self.padding_idx = args.embed_padding_idx if padding_idx is None else padding_idx
        # self.embed_num = int(args.embed_num)
        self.act_fn = args.embed_act_fn
        self.norm_args = dict(
                fn = args.embed_norm_fn,
                axis = args.embed_norm_axis,
                mask = args.embed_norm_mask,
                trainable = args.embed_norm_trainable,
                )
        self.act_after_norm = args.embed_act_after_norm
        self.pre_act_norm = args.embed_pre_act_norm

        self.out_fmt = None if out_fmt is None else out_fmt.upper() # not yet used

        logger.info(f'======= {self.__class__.__name__} =======')
        logger.info(f'        data_fmt: {self.data_fmt}')
        logger.info(f'      data_genre: {self.data_genre}')
        logger.info(f'     in_channels: {self.in_channels}')
        logger.info(f'        in_dtype: {self.in_dtype}')
        logger.info(f'        embed_fn: {self.embed_fn}')
        logger.info(f'       embed_dim: {self.embed_dim}')
        logger.info(f'     padding_idx: {self.padding_idx}')
        logger.info(f'          act_fn: {self.act_fn}')
        logger.info(f'         norm_fn: {self.norm_args["fn"]}')
        logger.info(f'       norm_axis: {self.norm_args["axis"]}')
        logger.info(f'       norm_mask: {self.norm_args["mask"]}')
        logger.info(f'  norm_trainable: {self.norm_args["trainable"]}')
        logger.info(f'  act_after_norm: {self.act_after_norm}')
        logger.info(f'    pre_act_norm: {self.pre_act_norm}')
        logger.info(f'         out_fmt: {self.out_fmt}')
        # logger.info(f'         dropout: {self.dropout}')
        # logger.info(f'          resnet: {self.resnet}')
        # logger.info(f'     resnet_beta: {self.resnet_beta}')
        # logger.info(f'       is_return: {self.is_return}')

        # ======= Layers
        in_channels = self.in_channels  # used for consistency in style
        self.LayerList = []
        in_fmt = self.data_fmt

        # ------- Transform input (should take care of this in preprocessing)

        # ------- Embed
        if self.embed_fn in [None, 'none']:
            pass
        else:
            if self.embed_fn == 'embed':
                assert self.data_fmt in ['NL', 'NHW', 'NLL'], f'embed cannot be applied to {self.data_fmt}'
                assert self.data_genre.startswith(('seq2quant', 'quant'))
                if self.in_dtype != 'int32':
                    logger.warning(f'data type must be int for embed, current: {self.in_dtype}')
                    self.in_dtype = 'int32' # enforce int32
                self.LayerList.append(nn.Embedding(
                    in_channels,  # num_embeddings, i.e., the vocabulary size
                    self.embed_dim,
                    padding_idx = self.padding_idx,
                    sparse = False,
                    ))
                in_channels = self.embed_dim
                in_fmt = in_fmt + 'C'

            elif self.embed_fn == 'linear':
                if not self.data_fmt.endswith('C') and 'C' in self.data_fmt:
                    assert self.data_fmt in ['NCL', 'NCHW', 'NCLL']
                    in_fmt = self.data_fmt.replace('C', '') + 'C'
                    self.LayerList.append(MatTranspose(in_fmt=self.data_fmt, out_fmt=in_fmt))
                elif 'C' not in self.data_fmt:
                    logger.warning('No C exists in self.data_fmt: {self.data_fmt}')

                # should use nn.Linear() instead
                self.LayerList.append(nn.Sequential(*stack_linear_block(
                    [in_channels, self.embed_dim],
                    data_fmt = self.out_fmt,
                    bias_attr = False,
                    act_fn = self.act_fn,
                    norm_args = self.norm_args,
                    pre_act_norm = False,
                    act_after_norm  = False,
                    dropout = 0.0,
                    resnet = False,
                    resnet_beta = 1.0,
                    is_return = True, # turn off act/norm/dropout for the last layer (added below)
                    )))
                in_channels = self.embed_dim
            else:
                logger.warning(f'Embedding fn: {self.embed_fn} is not supported yet!')

        # ------- Still can apply act_fn and norm_fn even without any embedding done
        if self.pre_act_norm:
            logger.error('Cannot do pre_act_norm for embedding!!!')
        room_order = ['act', 'norm']
        if self.act_after_norm: room_order = room_order[::-1]
        for room_type in room_order:
            if room_type == 'act':
                self.LayerList.append(get_act_layer(self.act_fn))
            if room_type == 'norm':
                self.LayerList.append(get_norm_layer(**self.norm_args, in_channels=in_channels,
                    data_fmt=in_fmt))

        # ------- transform if needed
        if self.out_fmt not in [None, in_fmt]:
            self.LayerList.append(MatTranspose(in_fmt=in_fmt, out_fmt=self.out_fmt))
        else:
            self.out_fmt = in_fmt

        self.LayerList = list(filter(None, self.LayerList))
        for i, OneLayer in enumerate(self.LayerList):
            self.add_sublayer(f'Layer{i}', OneLayer)

        self.out_channels = in_channels

        # if not isinstance(seqs_len, mi.Tensor) or seqs_len.dtype.name != 'INT32':
        #     seqs_len = mi.to_tensor(seqs_len, dtype='int32')
        # if not isinstance(seqs_len, mi.Tensor):
        #     seqs_len = mi.to_tensor(seqs_len, dtype='int32')

    def forward(self, x, seqs_len):
        """ x: must be a mi.Tensor; seqs_len: must be a mi.Tensor or np.ndarray """

        if mi.in_dynamic_mode():
            logger.debug(f'Applying {self.__class__.__name__}')

        if not isinstance(x, mi.Tensor) or not str(x.dtype).endswith(self.in_dtype):
            x = x.astype(self.in_dtype)

        # InputShape still gives float32 even if set dtype=int32
        # if not str(seqs_len.dtype).endswith('int32'):
        if seqs_len.dtype is not mi.int32:
            seqs_len = seqs_len.astype(mi.int32) # works for both numpy and paddle

        if mi.any(seqs_len < 1.0):
            seqs_len[:] = x.shape[1] - 1

        for OneLayer in self.LayerList:
            if isinstance(OneLayer, nn.Embedding): # not necessary yet ...
                x = OneLayer(x)
            elif isinstance(OneLayer, MyNormLayer): # accepts seqs_len
                x = OneLayer(x, seqs_len)
            else:
                x = OneLayer(x) # other layers don't accept seqs_len

        return x, seqs_len

    def summary(self, input_size=None):
        if input_size is None:
            input_size = self.in_shapes
        return mi.summary(self, input_size)


class MyLinearTower(nn.Layer):
    def __init__(self, args, in_channels=None, data_fmt=None, is_return=False):
        """ is_return:True will turn off Act/Norm/Dropout for the last block """
        super(MyLinearTower, self).__init__()

        # ======= Parameters
        self.data_fmt = args.input_fmt if data_fmt is None else data_fmt
        self.in_channels = args.input_dim if in_channels is None else in_channels
        self.num_blocks = int(args.linear_num)
        self.channel_sizes = [int(_i) for _i in args.linear_dim] \
                if hasattr(args.linear_dim, '__len__') else [int(args.linear_dim)]

        self.act_fn = args.act_fn if args.linear_act_fn is None else args.linear_act_fn
        self.norm_args = dict(
            fn = args.norm_fn if args.linear_norm_fn is None else args.linear_norm_fn,
            axis = args.norm_axis if args.linear_norm_axis is None else args.linear_norm_axis,
            mask = args.norm_mask if args.linear_norm_mask is None else args.linear_norm_mask,
            trainable = args.norm_trainable if args.linear_norm_trainable is None else args.linear_norm_trainable,
            )
        self.norm_in = args.norm_in if args.linear_norm_in is None else args.linear_norm_in
        self.norm_out = args.norm_out if args.linear_norm_out is None else args.linear_norm_out
        self.act_after_norm = args.act_after_norm if args.linear_act_after_norm is None else args.linear_act_after_norm
        self.pre_act_norm = args.pre_act_norm if args.linear_pre_act_norm is None else args.linear_pre_act_norm
        self.dropout = args.dropout if args.linear_dropout is None else args.linear_dropout
        self.resnet = args.linear_resnet
        self.resnet_beta = args.linear_resnet_beta
        self.is_return = is_return

        logger.info(f'======= {self.__class__.__name__} =======')
        logger.info(f'        data_fmt: {self.data_fmt}')
        logger.info(f'     in_channels: {self.in_channels}')
        logger.info(f'         norm_in: {self.norm_in}')
        logger.info(f'   channel_sizes: {self.channel_sizes}')
        logger.info(f'      num_blocks: {self.num_blocks}')
        logger.info(f'          act_fn: {self.act_fn}')
        logger.info(f'         norm_fn: {self.norm_args["fn"]}')
        logger.info(f'       norm_axis: {self.norm_args["axis"]}')
        logger.info(f'       norm_mask: {self.norm_args["mask"]}')
        logger.info(f'  norm_trainable: {self.norm_args["trainable"]}')
        logger.info(f'  act_after_norm: {self.act_after_norm}')
        logger.info(f'    pre_act_norm: {self.pre_act_norm}')
        logger.info(f'         dropout: {self.dropout}')
        logger.info(f'          resnet: {self.resnet}')
        logger.info(f'     resnet_beta: {self.resnet_beta}')
        logger.info(f'       is_return: {self.is_return}')
        logger.info(f'        norm_out: {self.norm_out}')

        # ======= Layers
        in_channels = self.in_channels
        self.LayerList = []

        if self.norm_in:
            self.LayerList.append(get_norm_layer(**dict(self.norm_args, fn=self.norm_in),
                in_channels=in_channels, data_fmt=self.data_fmt))

        for i in range(self.num_blocks):
            is_return = self.is_return and (i == self.num_blocks - 1)

            channel_sizes = [in_channels] + self.channel_sizes
            self.LayerList.extend(stack_linear_block(
                    channel_sizes,
                    data_fmt = self.data_fmt,
                    act_fn = self.act_fn,
                    norm_args = self.norm_args,
                    act_after_norm = self.act_after_norm,
                    pre_act_norm = self.pre_act_norm,
                    dropout = self.dropout,
                    resnet = self.resnet and not is_return,
                    resnet_beta = self.resnet_beta,
                    is_return = is_return,
            ))
            in_channels = channel_sizes[-1]

        if self.norm_out:
            self.LayerList.append(get_norm_layer(**dict(self.norm_args, fn=self.norm_out),
                in_channels=in_channels, data_fmt=self.data_fmt))

        for i, OneLayer in enumerate(self.LayerList):
            self.add_sublayer(f'Layer{i}', OneLayer)

        self.out_channels = in_channels

    def forward(self, x, seqs_len):
        # if not isinstance(x, mi.Tensor) or x.dtype.name != 'FP32':
        #     x = mi.(x, dtype='float32')
        if mi.in_dynamic_mode():
            logger.debug(f'Running {self.__class__.__name__}')

        if self.resnet:
            x0 = x

        for OneLayer in self.LayerList:
            if isinstance(OneLayer, MyResNetLayer): # self.linear_resnet
                x = x0 + OneLayer(x)
                x0 = x
            elif isinstance(OneLayer, MyNormLayer): # accepts seqs_len
                x = OneLayer(x, seqs_len)
            else:
                x = OneLayer(x) # other layers don't accept seqs_len

        return x

    def summary(self, input_size=None):
        if input_size is None:
            input_size = (2, 512, self.in_channels)
        return mi.summary(self, input_size)


class MyLSTM1DTower(nn.Layer):
    def __init__(self, args, in_channels=None, data_fmt='NLC', lstm_num=None, lstm_dim=None,
                is_return=False, initial_states=None, train_initial=None, return_states=False):
        super(MyLSTM1DTower, self).__init__()

        # ======= arguments
        self.data_fmt = args.input_fmt if data_fmt is None else data_fmt
        self.in_channels = args.input_dim if in_channels is None else in_channels

        self.num_blocks = int(lstm_num) if lstm_num else int(args.lstm_num)
        lstm_dim = args.lstm_dim if lstm_dim is None else lstm_dim
        self.channel_sizes = [int(_i) for _i in lstm_dim] \
                if hasattr(lstm_dim, '__len__') else [int(lstm_dim)]

        self.direction = args.lstm_direction.lower()
        self.num_directions = 1 if self.direction == 'forward' else 2
        self.act_fn = args.act_fn if args.lstm_act_fn is None else args.lstm_act_fn
        self.norm_args = dict(
            fn = args.norm_fn if args.lstm_norm_fn is None else args.lstm_norm_fn,
            axis = args.norm_axis if args.lstm_norm_axis is None else args.lstm_norm_axis,
            mask = args.norm_mask if args.lstm_norm_mask is None else args.lstm_norm_mask,
            trainable = args.norm_trainable if args.lstm_norm_trainable is None else args.lstm_norm_trainable,
        )
        self.norm_in = args.norm_in if args.lstm_norm_in is None else args.lstm_norm_in
        self.norm_out = args.norm_out if args.lstm_norm_out is None else args.lstm_norm_out
        self.act_after_norm = args.act_after_norm if args.lstm_act_after_norm is None else args.lstm_act_after_norm
        self.pre_act_norm = args.pre_act_norm if args.lstm_pre_act_norm is None else args.lstm_pre_act_norm
        self.dropout = args.dropout if args.lstm_dropout is None else args.lstm_dropout
        self.resnet = args.lstm_resnet
        self.resnet_beta = args.lstm_resnet_beta
        self.train_initial = args.lstm_train_initial if train_initial is None else train_initial
        self.initial_states = initial_states
        self.return_states = return_states
        self.is_return = is_return

        logger.info(f'======= {self.__class__.__name__} =======')
        logger.info(f'        data_fmt: {self.data_fmt}')
        logger.info(f'     in_channels: {self.in_channels}')
        logger.info(f'         norm_in: {self.norm_in}')
        logger.info(f'   channel_sizes: {self.channel_sizes}')
        logger.info(f'      num_blocks: {self.num_blocks}')
        logger.info(f'       direction: {self.direction}')
        logger.info(f'  num_directions: {self.num_directions}')
        logger.info(f'          act_fn: {self.act_fn}')
        logger.info(f'         norm_fn: {self.norm_args["fn"]}')
        logger.info(f'       norm_axis: {self.norm_args["axis"]}')
        logger.info(f'       norm_mask: {self.norm_args["mask"]}')
        logger.info(f'  norm_trainable: {self.norm_args["trainable"]}')
        logger.info(f'  act_after_norm: {self.act_after_norm}')
        logger.info(f'    pre_act_norm: {self.pre_act_norm}')
        logger.info(f'         dropout: {self.dropout}')
        logger.info(f'          resnet: {self.resnet}')
        logger.info(f'     resnet_beta: {self.resnet_beta}')
        logger.info(f'   train_initial: {self.train_initial}')
        logger.info(f'  initial states: {self.initial_states is not None}')
        logger.info(f'   return_states: {self.return_states}')
        logger.info(f'       is_return: {self.is_return}')
        logger.info(f'        norm_out: {self.norm_out}')

        # ======= layers
        # initial states for lstm
        assert all(np.array(self.channel_sizes) == self.channel_sizes[0]), \
            f'lstm_dim: {self.channel_sizes} must be the same so as to set initial states!!!'

        dims_initial = [self.num_directions, 1, self.channel_sizes[0]]
        if self.train_initial: # trainable initial states
            self.h_initial = mi.create_parameter(dims_initial, 'float32', name='h_initial',
                default_initializer=nn.initializer.Normal(mean=0.0, std=1.0),
                attr=mi.ParamAttr(learning_rate=0.5, regularizer=mi.regularizer.L2Decay(1e-3)))
            self.c_initial = mi.create_parameter(dims_initial, 'float32', name='c_initial',
                default_initializer=nn.initializer.Normal(mean=0.0, std=1.0),
                attr=mi.ParamAttr(learning_rate=0.5, regularizer=mi.regularizer.L2Decay(1e-3)))
        elif self.initial_states is None:
            self.register_buffer('h_initial', mi.normal(mean=0.0, std=1.0, shape=dims_initial))
            self.register_buffer('c_initial', mi.normal(mean=0.0, std=1.0, shape=dims_initial))
        else:
            assert len(self.initial_states) == 2, 'initial state must be a list/tuple of two items'
            self.h_initial = self.initial_states[0]
            self.c_initial = self.initial_states[1]

        in_channels = self.in_channels
        self.LayerList = []

        if self.norm_in:
            self.LayerList.append(get_norm_layer(**dict(self.norm_args, fn=self.norm_in),
                in_channels=in_channels, data_fmt=self.data_fmt))

        for i in range(self.num_blocks):
            is_return = self.is_return and (i == self.num_blocks -1)
            self.LayerList.extend(stack_lstm_block(
                    [in_channels] + self.channel_sizes,
                    # num_layers = 1, # one layer for each element of self.lstm_dim
                    direction = self.direction,
                    act_fn = self.act_fn,
                    norm_args = self.norm_args,
                    act_after_norm = self.act_after_norm,
                    pre_act_norm = self.pre_act_norm,
                    dropout = self.dropout,
                    resnet = self.resnet and not is_return,
                    resnet_beta = self.resnet_beta,
                    data_fmt = self.data_fmt,
                    is_return = is_return,
            ))
            in_channels = self.channel_sizes[-1] * self.num_directions

        if self.norm_out:
            self.LayerList.append(get_norm_layer(**dict(self.norm_args, fn=self.norm_out),
                in_channels=in_channels, data_fmt=self.data_fmt))

        for i, OneLayer in enumerate(self.LayerList):
            self.add_sublayer(f'Layer{i}', OneLayer)

        self.out_channels = in_channels

    def forward(self, x, seqs_len):
        logger.debug(f'Running {self.__class__.__name__}')

        if seqs_len is not None and not isinstance(seqs_len, mi.Tensor):
            seqs_len = mi.to_tensor(seqs_len, dtype='int32')

        batch_size, input_len, in_channels = x.shape

        initial_states = (mi.tile(self.h_initial, [1, batch_size, 1]),
                          mi.tile(self.c_initial, [1, batch_size, 1]))

        if self.resnet:
            x0 = x

        for OneLayer in self.LayerList:
            if isinstance(OneLayer, nn.LSTM):
                x, (h, c) = OneLayer(x,  sequence_length=seqs_len, initial_states=initial_states)
                h = F.layer_norm(h, h.shape[-1], weight=None, bias=None)
                c = F.layer_norm(c, c.shape[-1], weight=None, bias=None)
                initial_states = (h, c)
            elif isinstance(OneLayer, MyResNetLayer):
                x = x0 + OneLayer(x)
                x0 = x
            elif isinstance(OneLayer, MyNormLayer):
                x = OneLayer(x, seqs_len=seqs_len)
            else:
                x = OneLayer(x)

        # return x
        if self.return_states:
            return x, initial_states
        else:
            return x

    def summary(self, input_size=None):
        if input_size is None:
            input_size = (2, 512, self.in_channels)
        return mi.summary(self, input_size)


class MyConv1DTower(nn.Layer):
    def __init__(self, args, in_channels=None, data_fmt=None, is_return=False):
        super(MyConv1DTower, self).__init__()

        # ======= Parameters
        self.data_fmt = args.input_fmt if data_fmt is None else data_fmt.upper()
        self.in_channels = int(args.input_dim) if in_channels is None else int(in_channels)
        self.num_blocks = int(args.conv1d_num)

        # convert all conv1d pars to np.array
        fn_par2npa = lambda x: np.array([int(i) for i in x] if hasattr(x, '__len__') else [int(x)])
        self.channel_sizes = fn_par2npa(args.conv1d_dim)
        self.kernel_size = fn_par2npa(args.conv1d_kernel)
        self.stride = fn_par2npa(args.conv1d_stride)
        self.dilation = fn_par2npa(args.conv1d_dilation)

        # dim, stride, dilation, kernel_size should have the same length
        fn_fix_length = lambda x: mitas_utils.fix_length1d(x, len(self.channel_sizes), constant_values=x[-1])
        self.stride = fn_fix_length(self.stride)
        self.dilation = fn_fix_length(self.dilation)
        self.kernel_size = fn_fix_length(self.kernel_size)

        # padding is set to return length/stride
        if args.conv1d_padding is None:
            self.padding = mitas_utils.calc_padding(self.kernel_size, stride=self.stride,
                            dilation=self.dilation).astype(int)
        else:
            self.padding = fn_fix_length(fn_par2npa(args.conv1d_padding))

        self.act_fn = args.act_fn if args.conv1d_act_fn is None else args.conv1d_act_fn
        self.norm_args = dict(
            fn = args.norm_fn if args.conv1d_norm_fn is None else args.conv1d_norm_fn,
            axis = args.norm_axis if args.conv1d_norm_axis is None else args.conv1d_norm_axis,
            mask = args.norm_mask if args.conv1d_norm_mask is None else args.conv1d_norm_mask,
            trainable = args.norm_trainable if args.conv1d_norm_trainable is None else args.conv1d_norm_trainable,
        )
        self.norm_in = args.norm_in if args.conv1d_norm_in is None else args.conv1d_norm_in
        self.norm_out = args.norm_out if args.conv1d_norm_out is None else args.conv1d_norm_out
        self.act_after_norm = args.act_after_norm if args.conv1d_act_after_norm is None else args.conv1d_act_after_norm
        self.pre_act_norm = args.pre_act_norm if args.conv1d_pre_act_norm is None else args.conv1d_pre_act_norm
        self.dropout = args.dropout if args.conv1d_dropout is None else args.conv1d_dropout
        self.resnet = args.conv1d_resnet
        self.resnet_beta = args.conv1d_resnet_beta
        self.is_return = is_return

        logger.info(f'======= {self.__class__.__name__} =======')
        logger.info(f'        data_fmt: {self.data_fmt}')
        logger.info(f'     in_channels: {self.in_channels}')
        logger.info(f'         norm_in: {self.norm_in}')
        logger.info(f'   channel_sizes: {self.channel_sizes}')
        logger.info(f'      num_blocks: {self.num_blocks}')
        logger.info(f'     kernel_size: {self.kernel_size}')
        logger.info(f'          stride: {self.stride}')
        logger.info(f'        dilation: {self.dilation}')
        logger.info(f'         padding: {self.padding}')
        logger.info(f'          act_fn: {self.act_fn}')
        logger.info(f'         norm_fn: {self.norm_args["fn"]}')
        logger.info(f'       norm_axis: {self.norm_args["axis"]}')
        logger.info(f'       norm_mask: {self.norm_args["mask"]}')
        logger.info(f'  norm_trainable: {self.norm_args["trainable"]}')
        logger.info(f'  act_after_norm: {self.act_after_norm}')
        logger.info(f'    pre_act_norm: {self.pre_act_norm}')
        logger.info(f'         dropout: {self.dropout}')
        logger.info(f'          resnet: {self.resnet}')
        logger.info(f'     resnet_beta: {self.resnet_beta}')
        logger.info(f'       is_return: {self.is_return}')
        logger.info(f'        norm_out: {self.norm_out}')

        # ======= Layers
        in_channels = self.in_channels
        self.LayerList = []

        if self.norm_in:
            self.LayerList.append(get_norm_layer(**dict(self.norm_args, fn=self.norm_in),
                in_channels=in_channels, data_fmt=self.data_fmt))

        for i in range(self.num_blocks):
            is_return = self.is_return and (i == self.num_blocks -1)
            self.LayerList.extend(stack_conv1d_block(
                    [in_channels] + list(self.channel_sizes),
                    stride = self.stride,
                    kernel_size = self.kernel_size,
                    dilation = self.dilation,
                    padding = self.padding,
                    padding_mode = 'zeros',
                    max_pool = 1,
                    act_fn = self.act_fn,
                    norm_args = self.norm_args,
                    act_after_norm = self.act_after_norm,
                    pre_act_norm = self.pre_act_norm,
                    dropout = self.dropout,
                    data_fmt = self.data_fmt,
                    resnet = self.resnet and not is_return,
                    resnet_beta = self.resnet_beta,
                    is_return = is_return,
            ))
            in_channels = self.channel_sizes[-1]

        if self.norm_out:
            self.LayerList.append(get_norm_layer(**dict(self.norm_args, fn=self.norm_out),
                in_channels=in_channels, data_fmt=self.data_fmt))

        for i, OneLayer in enumerate(self.LayerList):
            self.add_sublayer(f'Layer{i}', OneLayer)

        self.out_channels = in_channels

    def forward(self, x, seqs_len):
        logger.debug(f'Running {self.__class__.__name__}')

        # if not isinstance(x, mi.Tensor) or x.dtype.name != 'FP32':
        #     x = mi.(x, dtype='float32')

        if self.resnet:
            x0 = x

        for OneLayer in self.LayerList:
            if  isinstance(OneLayer, MyResNetLayer):
                x = x0 + OneLayer(x)
                x0 = x
            elif isinstance(OneLayer, MyNormLayer):
                x = OneLayer(x, seqs_len)
            else:
                x = OneLayer(x)

        return x

    def summary(self, input_size=None):
        if input_size is None:
            input_size = (2, 512, self.in_channels)
        return mi.summary(self, input_size)


class MyConv2DTower(nn.Layer):
    def __init__(self, args, in_channels=None, data_fmt=None, is_return=False):
        super(MyConv2DTower, self).__init__()

        # ======= Parameters
        self.data_fmt = args.input_fmt if data_fmt is None else data_fmt
        self.in_channels = int(args.input_dim) if in_channels is None else int(in_channels)
        self.num_blocks = int(args.conv2d_num)

        # convert all conv pars to np.array
        fn_par2npa = lambda x: np.array([int(i) for i in x] if hasattr(x, '__len__') else [int(x)])
        self.channel_sizes = fn_par2npa(args.conv2d_dim)
        self.kernel_size = fn_par2npa(args.conv2d_kernel)
        self.stride = fn_par2npa(args.conv2d_stride)
        self.dilation = fn_par2npa(args.conv2d_dilation)

        # dim, stride, dilation, kernel_size should have the same length
        fn_fix_length = lambda x: mitas_utils.fix_length1d(x, len(self.channel_sizes), constant_values=x[-1])
        self.stride = fn_fix_length(self.stride)
        self.dilation = fn_fix_length(self.dilation)
        self.kernel_size = fn_fix_length(self.kernel_size)

        # padding is set to return length/stride
        if args.conv2d_padding is None:
            self.padding = mitas_utils.calc_padding(self.kernel_size, stride=self.stride,
                        dilation=self.dilation).astype(int)
        else:
            self.padding = fn_fix_length(fn_par2npa(args.conv2d_padding))

        self.act_fn = args.act_fn if args.conv2d_act_fn is None else args.conv2d_act_fn
        self.norm_args = dict(
            fn = args.norm_fn if args.conv2d_norm_fn is None else args.conv2d_norm_fn,
            axis = args.norm_axis if args.conv2d_norm_axis is None else args.conv2d_norm_axis,
            mask = args.norm_mask if args.conv2d_norm_mask is None else args.conv2d_norm_mask,
            trainable = args.norm_trainable if args.conv2d_norm_trainable is None else args.conv2d_norm_trainable,
        )
        self.norm_in = args.norm_in if args.conv2d_norm_in is None else args.conv2d_norm_in
        self.norm_out = args.norm_out if args.conv2d_norm_out is None else args.conv2d_norm_out
        self.act_after_norm = args.act_after_norm if args.conv2d_act_after_norm is None else args.conv2d_act_after_norm
        self.pre_act_norm = args.pre_act_norm if args.conv2d_pre_act_norm is None else args.conv2d_pre_act_norm
        self.dropout = args.dropout if args.conv2d_dropout is None else args.conv2d_dropout
        self.resnet = args.conv2d_resnet
        self.resnet_beta = args.conv2d_resnet_beta
        self.is_return = is_return

        logger.info(f'======= {self.__class__.__name__} =======')
        logger.info(f'        data_fmt: {self.data_fmt}')
        logger.info(f'     in_channels: {self.in_channels}')
        logger.info(f'         norm_in: {self.norm_in}')
        logger.info(f'   channel_sizes: {self.channel_sizes}')
        logger.info(f'      num_blocks: {self.num_blocks}')
        logger.info(f'     kernel_size: {self.kernel_size}')
        logger.info(f'          stride: {self.stride}')
        logger.info(f'        dilation: {self.dilation}')
        logger.info(f'         padding: {self.padding}')
        logger.info(f'          act_fn: {self.act_fn}')
        logger.info(f'         norm_fn: {self.norm_args["fn"]}')
        logger.info(f'       norm_axis: {self.norm_args["axis"]}')
        logger.info(f'       norm_mask: {self.norm_args["mask"]}')
        logger.info(f'  norm_trainable: {self.norm_args["trainable"]}')
        logger.info(f'  act_after_norm: {self.act_after_norm}')
        logger.info(f'    pre_act_norm: {self.pre_act_norm}')
        logger.info(f'         dropout: {self.dropout}')
        logger.info(f'          resnet: {self.resnet}')
        logger.info(f'     resnet_beta: {self.resnet_beta}')
        logger.info(f'       is_return: {self.is_return}')
        logger.info(f'        norm_out: {self.norm_out}')

        # ======= Layers
        in_channels = self.in_channels
        self.LayerList = []

        if self.norm_in:
            self.LayerList.append(get_norm_layer(**dict(self.norm_args, fn=self.norm_in),
                in_channels=in_channels, data_fmt=self.data_fmt))

        for i in range(self.num_blocks):
            # is_return only applies to the last block
            is_return = self.is_return and (i == self.num_blocks -1)
            self.LayerList.extend(stack_conv2d_block(
                [in_channels] + list(self.channel_sizes),
                stride = self.stride,
                kernel_size = self.kernel_size,
                dilation = self.dilation,
                padding = self.padding,
                padding_mode = 'zeros',
                act_fn = self.act_fn,
                norm_args = self.norm_args,
                act_after_norm = self.act_after_norm,
                pre_act_norm = self.pre_act_norm,
                dropout = self.dropout,
                data_fmt = self.data_fmt,
                resnet = self.resnet and not is_return,
                resnet_beta = self.resnet_beta,
                is_return = is_return,
            ))
            in_channels = self.channel_sizes[-1]

        if self.norm_out:
            self.LayerList.append(get_norm_layer(**dict(self.norm_args, fn=self.norm_out),
                in_channels=in_channels, data_fmt=self.data_fmt))

        for i, OneLayer in enumerate(self.LayerList):
            self.add_sublayer(f'Layer{i}', OneLayer)

        self.out_channels = in_channels

    def forward(self, x, seqs_len):
        logger.debug(f'Running {self.__class__.__name__}')

        # if not isinstance(x, mi.Tensor) or x.dtype.name != 'FP32':
        #     x = mi.(x, dtype='float32')

        if self.resnet:
            x0 = x

        for OneLayer in self.LayerList:
            if isinstance(OneLayer, MyResNetLayer):
                x = x0 + OneLayer(x)
                x0 = x
            elif isinstance(OneLayer, MyNormLayer):
                x = OneLayer(x, seqs_len)
            else:
                x = OneLayer(x)
        return x

    def summary(self, input_size=None):
        if input_size is None:
            input_size = (2, 512, self.in_channels)
        return mi.summary(self, input_size)


class MyAttnxDTower(nn.Layer):
    def __init__(self, args, in_channels=None, data_fmt=None, attn_dual=False,  attn_trio=False,
                 attn_axis=None, attn_method=None, mask='auto'):
        """ data_fmt: must be NLC or NHWC, or NLLC
            Only support a single channel used throughout
        """
        super(MyAttnxDTower, self).__init__()

        # ======= Parameters
        self.in_channels = int(args.input_dim)if in_channels is None else int(in_channels)
        self.data_fmt = data_fmt.upper() if data_fmt else args.input_fmt.upper()
        if self.data_fmt not in ['NLC', 'NHWC', 'NLLC']:
            logger.error(f'{self.data_fmt} not supported!, must be: NLC/NHWC/NLLC')

        self.mask = mask # not used yet
        self.posenc = args.attn_posenc
        self.posenc_join = args.attn_posenc_join
        self.posenc_dim = args.attn_posenc_dim
        self.posenc_mlp_num = args.attn_posenc_mlp_num
        self.posenc_mlp_dim = args.attn_posenc_mlp_dim

        self.force_nlc = args.attn_force_nlc

        self.num_blocks = int(args.attn_num)
        # normalize before each sublayer: attention and ff
        self.norm_before = args.attn_norm_before
        self.norm_in = args.norm_in if args.attn_norm_in is None else args.attn_norm_in
        self.norm_out = args.norm_out if args.attn_norm_out is None else args.attn_norm_out

        # dropout before residual connection at the end of each sublayer: attention and ff
        self.dropout = args.dropout if args.attn_dropout is None else args.attn_dropout

        # Layer 1: self-attention
        self.attn_method = args.attn_method.upper() if attn_method is None else attn_method.upper()
        assert self.attn_method in ['PADDLE', 'DOTPRODUCT', 'EFFICIENT']
        self.attn_axis = args.attn_axis if attn_axis is None else attn_axis
        if not hasattr(self.attn_axis, '__len__'):
            self.attn_axis = [self.attn_axis]
        self.attn_nhead = int(args.attn_nhead)
        self.attn_temperature = args.attn_temperature
        self.attn_act_fn = args.attn_act_fn # activation of attention weights (softmax by default)
        self.attn_dropout = self.dropout if args.attn_attn_dropout is None else args.attn_attn_dropout

        # Layer 2: feed forward
        self.join_method = args.attn_join
        self.ff_dim = args.attn_ffdim if args.attn_ffdim else self.in_channels * 4
        self.ff_act_fn = args.act_fn if args.attn_ffact_fn is None else args.attn_ffact_fn
        self.ff_dropout = self.dropout if args.attn_ffdropout is None else args.attn_ffdropout

        # Output normalization?
        if self.norm_out and not self.norm_before:
            logger.warning('The output would be normalized TWICE!!! with norm_out=True and norm_before=False')

        logger.info(f'======= {self.__class__.__name__} =======')
        logger.info(f'          method: {self.attn_method}')
        logger.info(f'        data_fmt: {self.data_fmt}')
        logger.info(f'         norm_in: {self.norm_in}')
        logger.info(f'position encoder: {self.posenc}')
        logger.info(f'            mask: {self.mask} (not yet used)')
        logger.info(f'       force NLC: {self.force_nlc}')
        logger.info(f'      num_blocks: {self.num_blocks}')
        logger.info(f'     in_channels: {self.in_channels}')
        logger.info(f'            axis: {self.attn_axis}')
        logger.info(f'           nhead: {self.attn_nhead}')
        logger.info(f'     temperature: {self.attn_temperature}')
        logger.info(f'     attn_act_fn: {self.attn_act_fn}')
        logger.info(f'    attn dropout: {self.attn_dropout}')
        logger.info(f'     join_method: {self.join_method}')
        logger.info(f'          ff_dim: {self.ff_dim}')
        logger.info(f'       ff_act_fn: {self.ff_act_fn}')
        logger.info(f'      ff_dropout: {self.ff_dropout}')
        logger.info(f'         dropout: {self.dropout}')
        logger.info(f'     norm_before: {self.norm_before}')
        logger.info(f'        norm_out: {self.norm_out}')
        # logger.info(f'    pre_act_norm: {self.pre_act_norm}')

        # ======= Layers
        in_channels = self.in_channels

        self.PosEncoder = PosiEncoderxD(
                        in_channels=in_channels,
                        pe_dim=self.posenc_dim,
                        data_fmt=self.data_fmt,
                        genre=self.posenc,
                        mlp_num=self.posenc_mlp_num,
                        mlp_dim=self.posenc_mlp_dim,
                        join_method=self.posenc_join,
                        )
        in_channels = self.PosEncoder.out_channels

        self.AttnMask = AttentionMask(force_nlc=self.force_nlc, attn_method=self.attn_method)

        ############## a temporary place for developing seq-mat cross-attention
        self.attn_dual = attn_dual
        self.attn_trio = attn_trio
        self.DualLayers = []
        if self.attn_dual:
            assert self.data_fmt == 'NLC', 'Only NLC supported for attn_dual!'
            self.data_fmt = 'NHWC' # q is changed to [N, L, 1, C]
            self.Seq2Mat = Seq2MatCastxD(method='multiply', in_fmt='NLC', out_fmt='NHWC')
            for i in range(self.num_blocks):
                # self.DualLayers.append(nn.Sequential(
                #     nn.Linear(in_channels, in_channels),
                #     nn.LeakyReLU(),
                #     nn.LayerNorm(in_channels),
                #     nn.Dropout(0.1),
                # ))
                self.DualLayers.append(MyAttnxDLayer(
                    in_channels,
                    nhead=self.attn_nhead,
                    axis=[1,2],
                    axis_share_attn=True,
                    data_fmt='NHWC',
                    temperature=1.0,
                    dropout=self.dropout,
                    attn_posenc=self.posenc,       # this now for rotary position encoding only
                    attn_method=self.attn_method,
                    attn_act_fn=self.attn_act_fn,
                    attn_dropout=self.attn_dropout,
                    join_method=self.join_method,
                    ff_dim=self.ff_dim,
                    ff_act_fn=self.ff_act_fn,
                    ff_dropout=self.ff_dropout,
                    norm_before=self.norm_before,
                    ))
        for i, DualLayer in enumerate(self.DualLayers):
            self.add_sublayer(f'DualLayer{i}', DualLayer)

        if self.norm_in is not None:
            self.NormIn = get_norm_layer(in_channels=in_channels, fn=self.norm_in, mask=False,
                                data_fmt='NLC' if self.force_nlc else self.data_fmt)

        self.LayerList = []

        if self.attn_method == 'PADDLE':
            assert self.attn_temperature is None, \
                        'attn_temperature not supported when attn_method=paddle!!!'
            assert self.force_nlc or len(self.data_fmt) <=3, "Paddle only takes NLC data"
            assert self.attn_axis[0] in [1, -2], 'Paddle only attends over L in NLC'

            # turn off bias in attention?
            bias_attr = mi.ParamAttr(name=None, initializer=None, learning_rate=1.0,
                        regularizer=None, trainable=False, do_model_average=True, need_clip=True)

            attn_layer = nn.TransformerEncoderLayer(
                d_model = in_channels,
                nhead = self.attn_nhead,
                dim_feedforward = self.ff_dim, # feed_forward dimension
                dropout = self.dropout, # between layers (default: 0.1)
                attn_dropout = self.attn_dropout, # for self-attention target
                activation = self.ff_act_fn, # (default: relu) for the feedforward layer
                act_dropout = self.ff_dropout, # after activation in feedforward
                normalize_before = self.norm_before, # layer_norm before dropout/resnet?
                weight_attr = None,
                bias_attr = None,
            )

            self.LayerList.append(nn.TransformerEncoder(
                attn_layer,
                num_layers= self.num_blocks,
                norm = nn.LayerNorm(in_channels) if self.norm_out else None,
            ))
        else: # 'DOTPRODUCT' or 'EFFICIENT'

            # temperature is a multiplier of the scale in ScaledAttention
            if self.attn_temperature is None:
                temperature = [1.0] * self.num_blocks
            else:
                self.attn_temperature = self.attn_temperature.lower()
                if self.attn_temperature.startswith('cool'):
                    temperature = [0.1] * self.num_blocks
                elif self.attn_temperature.startswith('warm'):
                    temperature = [1.0] * self.num_blocks
                elif self.attn_temperature.startswith('hot'):
                    temperature = [10.0] * self.num_blocks
                elif self.attn_temperature.startswith('anneal'):
                    temperature = np.logspace(1, -1, num=self.num_blocks)
            data_fmt = 'NLC' if self.force_nlc else self.data_fmt

            assert len(data_fmt) - len(self.attn_axis) > 1, \
                f'Cannot attend over axis: {self.attn_axis} for data_fmt: {data_fmt}!'

            for i in range(self.num_blocks):
                # self.attn_layers.append(AxialformerEncoderLayer(in_channels, axis=self.attn_axis))
                # for axis in self.attn_axis:
                self.LayerList.append(MyAttnxDLayer(
                    in_channels,
                    nhead=self.attn_nhead,
                    axis=self.attn_axis,
                    data_fmt=data_fmt,
                    temperature=temperature[i],
                    dropout=self.dropout,
                    attn_posenc=self.posenc,
                    attn_method=self.attn_method,
                    attn_act_fn=self.attn_act_fn,
                    attn_dropout=self.attn_dropout,
                    join_method=self.join_method,
                    ff_dim=self.ff_dim,
                    ff_act_fn=self.ff_act_fn,
                    ff_dropout=self.ff_dropout,
                    norm_before=self.norm_before,
                    ))

        # if self.norm_out:
        #     self.LayerList.append(get_norm_layer(**dict(self.norm_args, fn=self.norm_out),
        #         in_channels=in_channels, data_fmt=self.data_fmt))

        if self.norm_out is not None:
            self.LayerList.append(get_norm_layer(in_channels=in_channels, fn=self.norm_out, mask=False,
                                data_fmt='NLC' if self.force_nlc else self.data_fmt))
        # close out
        for i, OneLayer in enumerate(self.LayerList):
            self.add_sublayer(f'Layer{i}', OneLayer)

        self.out_channels = in_channels

    def forward(self, q, k=None, v=None, seqs_len=None):
        if mi.in_dynamic_mode():
            logger.debug(f'Running {self.__class__.__name__}')

        q0_shape = q.shape
        if self.norm_in is not None:
            if isinstance(self.NormIn, MyNormLayer):
                q = self.NormIn(q, seqs_len)
            else:
                q = self.NormIn(q)

        q = self.PosEncoder(q, seqs_len=seqs_len, beta=1.0)

        if self.attn_dual:
            assert q.ndim == 3, 'q must be 1D for cross attention!'
            k = self.Seq2Mat(q, q)
            q.unsqueeze_(-2)
            mask = None
        else:
            # only pass data_len to AttnMask
            mask = self.AttnMask(q0_shape[1:-1], seqs_len=seqs_len)

        if self.force_nlc and len(q0_shape) > 3:
            q = q.reshape([q0_shape[0], -1, q0_shape[-1]])

        for i, OneLayer in enumerate(self.LayerList):
            if isinstance(OneLayer, nn.TransformerEncoder):
                # paddle doesn't accept empty mask
                q = OneLayer(q, mask if mask is not None and len(mask) else None)
            elif isinstance(OneLayer, MyAttnxDLayer):
                q = OneLayer(q, k, mask=mask) # it had a v? not sure why

                if self.attn_dual:
                    k += self.Seq2Mat(q.squeeze(-2), q.squeeze(-2))
                    # this is the post-processing for mat
                    k = self.DualLayers[i](k)
                else:
                    k = q

            elif isinstance(OneLayer, MyNormLayer): # AxialAttention doesnot support mask or batch_size > 1
                q = OneLayer(q, seqs_len)
            else:
                q = OneLayer(q)

        if self.force_nlc and len(q0_shape) > 3:
            q = q.reshape(q0_shape)

        if self.attn_dual:
            return q.squeeze_(-2), k
        else:
            return q

    def summary(self, input_size=None):
        if input_size is None:
            input_size = (2, 512, self.in_channels)
        return mi.summary(self, input_size)


def MyInitTower(args, in_channels=None, data_fmt=None, **kwargs):
    new_args = misc.Struct(vars(args)).update(
                linear_dim = args.init_dim,
                linear_num = args.init_num,
                linear_act_fn = args.init_act_fn,
                linear_norm_fn = args.init_norm_fn,
                linear_norm_in = args.init_norm_in,
                linear_norm_out = args.init_norm_out,
                linear_norm_axis = args.init_norm_axis,
                linear_norm_mask = args.init_norm_mask,
                linear_norm_trainable = args.init_norm_trainable,
                linear_act_after_norm = args.init_act_after_norm,
                linear_pre_act_norm = args.init_pre_act_norm,
                linear_dropout = args.init_dropout,
                linear_resnet = args.init_resnet,
                linear_resnet_beta = args.init_resnet_beta,
                )
    new_args.update(kwargs)

    return MyLinearTower(new_args,
                data_fmt = args.input_fmt if data_fmt is None else data_fmt,
                in_channels = in_channels,
                is_return = False,
        )


def MyReturnTower(args, in_channels=None, data_fmt=None, **kwargs):
    new_args = misc.Struct(vars(args)).update(
                linear_dim = args.return_dim,
                linear_num = args.return_num,
                linear_act_fn = args.return_act_fn,
                linear_norm_fn = args.return_norm_fn,
                linear_norm_in = args.return_norm_in,
                linear_norm_out = args.return_norm_out,
                linear_norm_axis = args.return_norm_axis,
                linear_norm_mask = args.return_norm_mask,
                linear_norm_trainable = args.return_norm_trainable,
                linear_act_after_norm = args.return_act_after_norm,
                linear_pre_act_norm = args.return_pre_act_norm,
                linear_dropout = args.return_dropout,
                linear_resnet = args.return_resnet,
                linear_resnet_beta = args.return_resnet_beta,
                )
    new_args.update(kwargs)

    return MyLinearTower(new_args,
                data_fmt = args.input_fmt if data_fmt is None else data_fmt,
                in_channels = in_channels,
                is_return = True,
        )


class LinearNet(nn.Layer):
    """ This ignores all inter-residue interactions  """
    def __init__(self, args):
        super(LinearNet, self).__init__()

        self.data_fmt = args.input_fmt.upper()
        # self.data_ndim = int(args.input_ndim)
        self.in_channels = int(args.input_dim)

        in_channels = self.in_channels # keep record of the running channel dim

        self.Embed = MyEmbdxDLayer(args, in_channels=in_channels, data_fmt=self.data_fmt)
        in_channels = self.Embed.out_channels

        self.InLinear = MyLinearTower(args, in_channels=in_channels) # data_fmt=self.Embed.out_fmt
        in_channels = self.InLinear.out_channels

        self.OutLinear = MyReturnTower(args, in_channels=in_channels)

    def summary(self, input_size=None):
        if input_size is None:
            input_size = self.Embed.in_shapes
        return mi.summary(self, input_size)

    # @mi.jit.to_static
    def forward(self, x, seqs_len):

        x, seqs_len = self.Embed(x, seqs_len)
        x = self.InLinear(x, seqs_len)
        x = self.OutLinear(x, seqs_len)

        return x


class Seq2Seq_LSTMNet(nn.Layer):
    def __init__(self, args):
        super(Seq2Seq_LSTMNet, self).__init__()

        self.data_fmt = args.input_fmt.upper()
        # self.data_ndim = int(args.input_ndim)
        self.in_channels = int(args.input_dim)

        in_channels = self.in_channels # keep record of the running channel dim

        self.Embed = MyEmbdxDLayer(args, in_channels=in_channels, out_fmt='NLC')
        in_channels = self.Embed.out_channels

        self.InLinear = MyInitTower(args, in_channels=in_channels, data_fmt='NLC')
        in_channels = self.InLinear.out_channels

        self.LSTM = MyLSTM1DTower(args, in_channels=in_channels, data_fmt='NLC')
        in_channels = self.LSTM.out_channels

        self.OutLinear = MyReturnTower(args, in_channels=in_channels, data_fmt='NLC')

    # @property
    def summary(self, input_size=None):
        if input_size is None:
            input_size = self.Embed.in_shapes
        return mi.summary(self, input_size)

    def forward(self, x, seqs_len=None):

        x, seqs_len = self.Embed(x, seqs_len=seqs_len)
        x = self.InLinear(x, seqs_len=seqs_len)
        x = self.LSTM(x, seqs_len=seqs_len)
        x = self.OutLinear(x, seqs_len=seqs_len)
        return x


class Seq2Seq_ConvNet(nn.Layer):
    """ This information  """
    def __init__(self, args):
        super(Seq2Seq_ConvNet, self).__init__()

        self.data_fmt = args.input_fmt.upper()
        # self.data_ndim = int(args.input_ndim)
        self.in_channels = int(args.input_dim)

        in_channels = self.in_channels # keep record of the running channel dim

        self.Embed = MyEmbdxDLayer(args, in_channels=in_channels, out_fmt='NLC')
        in_channels = self.Embed.out_channels

        self.InLinear = MyInitTower(args, in_channels=in_channels, data_fmt='NLC')
        in_channels = self.InLinear.out_channels

        self.Conv1D = MyConv1DTower(args, in_channels=in_channels, data_fmt='NLC')
        in_channels = self.Conv1D.out_channels

        self.OutLinear = MyReturnTower(args, in_channels=in_channels, data_fmt='NLC')

    def forward(self, x, seqs_len): #, predict=False):

        x, seqs_len = self.Embed(x, seqs_len)
        x = self.InLinear(x, seqs_len)
        x = self.Conv1D(x, seqs_len)
        x = self.OutLinear(x, seqs_len)

        return x

    def summary(self, input_size=None):
        if input_size is None:
            input_size = self.Embed.in_shapes
        return mi.summary(self, input_size)


class Seq2Seq_AttnNet(nn.Layer):
    def __init__(self, args):
        super(Seq2Seq_AttnNet, self).__init__()

        self.data_fmt = args.input_fmt.upper()
        # self.data_ndim = int(args.input_ndim)
        self.in_channels = int(args.input_dim)

        in_channels = self.in_channels # keep record of the running channel dim

        self.Embed = MyEmbdxDLayer(args, in_channels=in_channels, out_fmt='NLC')
        in_channels = self.Embed.out_channels

        self.InLinear = MyInitTower(args, in_channels=in_channels, data_fmt='NLC')
        in_channels = self.InLinear.out_channels

        self.Attn = MyAttnxDTower(args, in_channels=in_channels, data_fmt='NLC')
        in_channels = self.Attn.out_channels

        self.OutLinear = MyReturnTower(args, in_channels=in_channels, data_fmt='NLC')

    def forward(self, x, seqs_len):

        x, seqs_len = self.Embed(x, seqs_len)
        x = self.InLinear(x, seqs_len)
        x = self.Attn(x, seqs_len)
        x = self.OutLinear(x, seqs_len)

        return x

    # @property
    def summary(self, input_size=None):
        if input_size is None:
            input_size = self.Embed.in_shapes
        return mi.summary(self, input_size)


class Seq2Seq_Attn1DConv1DNet(nn.Layer):
    def __init__(self, args):
        super(Seq2Seq_Attn1DConv1DNet, self).__init__()

        self.data_fmt = args.input_fmt.upper()
        # self.data_ndim = int(args.input_ndim)
        self.in_channels = int(args.input_dim)

        in_channels = self.in_channels # keep record of the running channel dim

        self.Embed = MyEmbdxDLayer(args, in_channels=in_channels, out_fmt='NLC')
        in_channels = self.Embed.out_channels

        self.InLinear = MyInitTower(args, in_channels=in_channels, data_fmt='NLC')
        in_channels = self.InLinear.out_channels

        self.Attn = MyAttnxDTower(args, in_channels=in_channels, data_fmt='NLC')
        in_channels = self.Attn.out_channels

        self.Conv1D = MyConv1DTower(args, in_channels=in_channels, data_fmt='NLC')
        in_channels = self.Conv1D.out_channels

        self.OutLinear = MyReturnTower(args, in_channels=in_channels, data_fmt='NLC')

    def forward(self, x, seqs_len):

        x, seqs_len = self.Embed(x, seqs_len)
        x = self.InLinear(x, seqs_len)
        x = self.Attn(x, seqs_len)
        x = self.Conv1D(x, seqs_len)
        x = self.OutLinear(x, seqs_len)

        return x

    # @property
    def summary(self, input_size=None):
        if input_size is None:
            input_size = self.Embed.in_shapes
        return mi.summary(self, input_size)


class Seq2Seq_Attn1DConv1D2PNet(nn.Layer):
    def __init__(self, args):
        super(Seq2Seq_Attn1DConv1D2PNet, self).__init__()

        self.data_fmt = args.input_fmt.upper()
        # self.data_ndim = int(args.input_ndim)
        self.in_channels = int(args.input_dim)

        in_channels = self.in_channels # keep record of the running channel dim

        self.Embed = MyEmbdxDLayer(args, in_channels=in_channels, out_fmt='NLC')
        in_channels = self.Embed.out_channels

        self.InLinear = MyInitTower(args, in_channels=in_channels, data_fmt='NLC')
        in_channels = self.InLinear.out_channels

        self.Attn = MyAttnxDTower(args, in_channels=in_channels, data_fmt='NLC')
        in_channels = self.Attn.out_channels

        self.Conv1DSin = MyConv1DTower(args, in_channels=in_channels, data_fmt='NLC')

        self.Conv1DCos = MyConv1DTower(args, in_channels=in_channels, data_fmt='NLC')

        in_channels = self.Conv1DSin.out_channels
        self.OutLinearSin = MyReturnTower(args, in_channels=in_channels, data_fmt='NLC')

        in_channels = self.Conv1DCos.out_channels
        self.OutLinearCos = MyReturnTower(args, in_channels=in_channels, data_fmt='NLC')

    def forward(self, x, seqs_len):

        x, seqs_len = self.Embed(x, seqs_len)
        x = self.InLinear(x, seqs_len)
        x = self.Attn(x, seqs_len)

        x_sin = self.Conv1DSin(x, seqs_len)
        x_sin = self.OutLinearSin(x_sin, seqs_len)
        x_sin = mi.sin(x_sin)
        # x_sin = F.softsign(x_sin)

        x_cos = self.Conv1DCos(x, seqs_len)
        x_cos = self.OutLinearCos(x_cos, seqs_len)
        x_cos = mi.cos(x_cos)
        # x_cos = F.softsign(x_cos)

        return mi.stack([x_sin, x_cos], axis=-1)

    # @property
    def summary(self, input_size=None):
        if input_size is None:
            input_size = self.Embed.in_shapes
        return mi.summary(self, input_size)


class Seq2Seq_Conv1DLSTMNet(nn.Layer):
    """ This information  """
    def __init__(self, args):
        super(Seq2Seq_Conv1DLSTMNet, self).__init__()

        self.data_fmt = args.input_fmt.upper()
        # self.data_ndim = int(args.input_ndim)
        self.in_channels = int(args.input_dim)

        in_channels = self.in_channels # keep record of the running channel dim

        self.Embed = MyEmbdxDLayer(args, in_channels=in_channels, out_fmt='NLC')
        in_channels = self.Embed.out_channels

        self.InLinear = MyInitTower(args, in_channels=in_channels, data_fmt='NLC')
        in_channels = self.InLinear.out_channels

        self.Conv1D = MyConv1DTower(args, in_channels=in_channels, data_fmt='NLC')
        in_channels = self.Conv1D.out_channels

        self.LSTM = MyLSTM1DTower(args, in_channels=in_channels, data_fmt='NLC')
        in_channels = self.LSTM.out_channels

        self.OutLinear = MyReturnTower(args, in_channels=in_channels, data_fmt='NLC')

    def forward(self, x, seqs_len=None):

        x, seqs_len = self.Embed(x, seqs_len=seqs_len)
        x = self.InLinear(x, seqs_len=seqs_len)
        x = self.Conv1D(x, seqs_len=seqs_len)
        x = self.LSTM(x, seqs_len=seqs_len)
        x = self.OutLinear(x, seqs_len=seqs_len)

        return x

    def summary(self, input_size=None):
        if input_size is None:
            input_size = self.Embed.in_shapes
        return mi.summary(self, input_size)


class Seq2Seq_AttnLSTMNet(nn.Layer):
    def __init__(self, args):
        super(Seq2Seq_AttnLSTMNet, self).__init__()

        self.data_fmt = args.input_fmt.upper()
        # self.data_ndim = int(args.input_ndim)
        self.in_channels = int(args.input_dim)

        in_channels = self.in_channels # keep record of the running channel dim

        self.Embed = MyEmbdxDLayer(args, in_channels=in_channels, out_fmt='NLC')
        in_channels = self.Embed.out_channels

        self.InLinear = MyLinearTower(args, in_channels=in_channels, data_fmt='NLC')
        in_channels = self.InLinear.out_channels

        self.Attn = MyAttnxDTower(args, in_channels=in_channels, data_fmt='NLC')
        in_channels = self.Attn.out_channels

        self.LSTM = MyLSTM1DTower(args, in_channels=in_channels, data_fmt='NLC')
        in_channels = self.LSTM.out_channels

        self.OutLinear = MyReturnTower(args, in_channels=in_channels, data_fmt='NLC')

    # @property
    def summary(self, input_size=None):
        if input_size is None:
            input_size = self.Embed.in_shapes
        return mi.summary(self, input_size)

    def forward(self, x, seqs_len=None):

        x, seqs_len = self.Embed(x, seqs_len=seqs_len)
        x = self.InLinear(x, seqs_len=seqs_len)
        x = self.Attn(x, seqs_len=seqs_len)
        x = self.LSTM(x, seqs_len=seqs_len)
        x = self.OutLinear(x, seqs_len=seqs_len)

        return x


class Seq2Seq_AttnLSTMConvNet(nn.Layer):
    def __init__(self, args):
        super(Seq2Seq_AttnLSTMConvNet, self).__init__()

        self.data_fmt = args.input_fmt.upper()
        # self.data_ndim = int(args.input_ndim)
        self.in_channels = int(args.input_dim)

        in_channels = self.in_channels # keep record of the running channel dim

        self.Embed = MyEmbdxDLayer(args, in_channels=in_channels, out_fmt='NLC')
        in_channels = self.Embed.out_channels

        self.InLinear = MyInitTower(args, in_channels=in_channels, data_fmt='NLC')
        in_channels = self.InLinear.out_channels

        self.Attn = MyAttnxDTower(args, in_channels=in_channels, data_fmt='NLC')
        in_channels = self.Attn.out_channels

        self.LSTM = MyLSTM1DTower(args, in_channels=in_channels, data_fmt='NLC')
        in_channels = self.LSTM.out_channels

        self.Conv1D = MyConv1DTower(args, in_channels=in_channels, data_fmt='NLC')
        in_channels = self.Conv1D.out_channels

        self.OutLinear = MyReturnTower(args, in_channels=in_channels, data_fmt='NLC')

    # @property
    def summary(self, input_size=None):
        if input_size is None:
            input_size = self.Embed.in_shapes
        return mi.summary(self, input_size)

    def forward(self, x, seqs_len):

        x, seqs_len = self.Embed(x, seqs_len)
        x = self.InLinear(x, seqs_len)
        x = self.Attn(x, seqs_len)
        x = self.LSTM(x, seqs_len)
        x = self.Conv1D(x, seqs_len)
        x = self.OutLinear(x, seqs_len)

        return x


class Seq2Mat_LinearNet(nn.Layer):
    """   """
    def __init__(self, args):
        super(Seq2Mat_LinearNet, self).__init__()

        self.data_fmt = args.input_fmt.upper()
        # self.data_ndim = int(args.input_ndim)
        self.in_channels = int(args.input_dim)

        in_channels = self.in_channels # keep record of the running channel dim

        self.Embed = MyEmbdxDLayer(args, in_channels=in_channels, out_fmt='NLC')
        in_channels = self.Embed.out_channels

        self.InLinear = MyInitTower(args, in_channels=in_channels, data_fmt='NLC')
        in_channels = self.InLinear.out_channels

        self.Seq2Mat = Seq2MatCastxD(method='concat', in_fmt='NLC', out_fmt='NHWC')
        in_channels = in_channels * 2 # * 2 due to concatenation

        self.Linear = MyLinearTower(args, in_channels=in_channels, data_fmt='NHWC')
        in_channels = self.Linear.out_channels

        self.OutLinear = MyReturnTower(args, in_channels=in_channels, data_fmt='NHWC')

    def forward(self, x, seqs_len):

        x, seqs_len = self.Embed(x, seqs_len)
        x = self.InLinear(x, seqs_len)

        x = self.Seq2Mat(x, x)
        x = (x + mi.transpose(x, perm=[0, 2, 1, 3])) / 2.0

        x = self.Linear(x, seqs_len=seqs_len)

        x = self.OutLinear(x, seqs_len=seqs_len)

        return x # mi.squeeze(x[:,:,1], axis=-1)

    def summary(self, input_size=None):
        if input_size is None:
            input_size = self.Embed.in_shapes
        return mi.summary(self, input_size)


class Seq2Mat_Conv2DNet(nn.Layer):
    """ This ignores all inter-residue information  """
    def __init__(self, args):
        super(Seq2Mat_Conv2DNet, self).__init__()

        self.data_fmt = args.input_fmt.upper()
        # self.data_ndim = int(args.input_ndim)
        self.in_channels = int(args.input_dim)

        in_channels = self.in_channels # keep record of the running channel dim

        self.Embed = MyEmbdxDLayer(args, in_channels=in_channels, out_fmt='NLC')
        in_channels = self.Embed.out_channels

        self.InLinear = MyInitTower(args, in_channels=in_channels, data_fmt='NLC')
        in_channels = self.InLinear.out_channels

        self.Seq2Mat = Seq2MatCastxD(method='concat', in_fmt='NLC', out_fmt='NHWC')
        in_channels = in_channels * 2 # * 2 due to concatenation

        self.Linear = MyLinearTower(args, in_channels=in_channels, data_fmt='NHWC')
        in_channels = self.Linear.out_channels

        self.Conv2D = MyConv2DTower(args, in_channels=in_channels, data_fmt='NHWC')
        in_channels = self.Conv2D.out_channels

        self.OutLinear = MyReturnTower(args, in_channels=in_channels, data_fmt='NHWC')

    def forward(self, x, seqs_len=None):
        x = self.Embed(x)
        x = self.InLinear(x, seqs_len=seqs_len)

        # # for each channel/channel, get a LxL matrix
        # x = mi.transpose(x, perm=[0, 2, 1]) # [NLC] --> [NCL]
        # new_shape = [x.shape[0], x.shape[1], x.shape[2], x.shape[2]]
        # x = mi.concat([mi.broadcast_to(mi.unsqueeze(x, axis=3), shape=new_shape),
        #                mi.broadcast_to(mi.unsqueeze(x, axis=2), shape=new_shape)],
        #                axis=1) # [NCLL] --> [N, 2*C, L, L]

        x = self.Seq2Mat(x, x)

        x = self.Linear(x, seqs_len=seqs_len)

        x = self.Conv2D(x, seqs_len=seqs_len)

        x = (x + mi.transpose(x, perm=[0, 2, 1, 3])) / 2.0

        x = self.OutLinear(x, seqs_len=seqs_len)

        # x = mi.squeeze(x[:, :, :, 0], axis=-1) # -> [N, L, L, 1] -> [NLL]

        return x # mi.squeeze(x[:,:,1], axis=-1)

    def summary(self, input_size=None):
        if input_size is None:
            input_size = self.Embed.in_shapes
        return mi.summary(self, input_size)


class Seq2Mat_Attn1DNet(nn.Layer):
    """   """
    def __init__(self, args, seq2mat_method='multiply'):
        super(Seq2Mat_Attn1DNet, self).__init__()

        self.data_fmt = args.input_fmt.upper()
        # self.data_ndim = int(args.input_ndim)
        self.in_channels = int(args.input_dim)
        self.seq2mat_method = seq2mat_method.upper()

        in_channels = self.in_channels # keep record of the running channel dim

        self.Embed = MyEmbdxDLayer(args, in_channels=in_channels, out_fmt='NLC')
        in_channels = self.Embed.out_channels

        self.InLinear = MyInitTower(args, in_channels=in_channels, data_fmt='NLC')
        in_channels = self.InLinear.out_channels

        self.AttnEncoder = MyAttnxDTower(args, in_channels=in_channels, data_fmt='NLC')
        in_channels = self.AttnEncoder.out_channels

        self.Seq2Mat = Seq2MatCastxD(method='multiply', in_fmt='NLC', out_fmt='NHWC')
        in_channels = in_channels * self.Seq2Mat.mul_channels

        # self.Linear = MyLinearTower(args, in_channels=in_channels, data_fmt='NLC')
        # in_channels = self.Linear.out_channels

        self.OutLinear = MyReturnTower(args, in_channels=in_channels, data_fmt='NHWC')

    def forward(self, x, seqs_len=None):

        x, seqs_len = self.Embed(x, seqs_len=seqs_len)
        x = self.InLinear(x, seqs_len=seqs_len)

        x = self.AttnEncoder(x, seqs_len=seqs_len)

        x = self.Seq2Mat(x, x)

        # x = self.Linear(x, seqs_len=seqs_len)

        if self.seq2mat_method in ['ADD', 'MULTIPLY']:
            pass
        else:
            x = (x + mi.transpose(x, perm=[0, 2, 1, 3])) / 2.0

        x = self.OutLinear(x, seqs_len=seqs_len)

        return x # mi.squeeze(x[:,:,1], axis=-1)

    def summary(self, input_size=None):
        if input_size is None:
            input_size = self.Embed.in_shapes
        return mi.summary(self, input_size)


class Seq2Mat_LSTMNet(nn.Layer):
    """   """
    def __init__(self, args):
        super(Seq2Mat_LSTMNet, self).__init__()

        self.data_fmt = args.input_fmt.upper()
        # self.data_ndim = int(args.input_ndim)
        self.in_channels = int(args.input_dim)

        in_channels = self.in_channels # keep record of the running channel dim

        self.Embed = MyEmbdxDLayer(args, in_channels=in_channels, out_fmt='NLC')
        in_channels = self.Embed.out_channels

        self.InLinear = MyInitTower(args, in_channels=in_channels, data_fmt='NLC')
        in_channels = self.InLinear.out_channels

        self.LSTM = MyLSTM1DTower(args, in_channels=in_channels, data_fmt='NLC')
        in_channels = self.LSTM.out_channels

        self.Linear = MyLinearTower(args, in_channels=in_channels, data_fmt='NLC')
        in_channels = self.Linear.out_channels

        self.Seq2Mat = Seq2MatCastxD(method='concat', in_fmt='NLC', out_fmt='NHWC')
        in_channels = in_channels # due to concatenation

        self.OutLinear = MyReturnTower(args, in_channels=in_channels, data_fmt='NHWC')

    # @mi.jit.to_static
    def forward(self, x, seqs_len):
        x, seqs_len = self.Embed(x, seqs_len)
        x = self.InLinear(x, seqs_len)
        x = self.LSTM(x, seqs_len)
        x = self.Linear(x, seqs_len)

        x = self.Seq2Mat(*mi.chunk(x, 2, axis=-1))# x, x)

        x = self.OutLinear(x, seqs_len)

        x = (x + mi.transpose(x, perm=[0, 2, 1, 3])) / 2.0

        return x # mi.squeeze(x[:,:,1], axis=-1)

    def summary(self, input_size=None):
        if input_size is None:
            input_size = self.Embed.in_shapes
        return mi.summary(self, input_size)


class Seq2Mat_Attn2DNet(nn.Layer):
    """   """
    def __init__(self, args):
        super(Seq2Mat_Attn2DNet, self).__init__()

        self.data_fmt = args.input_fmt.upper()
        # self.data_ndim = int(args.input_ndim)
        self.in_channels = int(args.input_dim)

        in_channels = self.in_channels # keep record of the running channel dim

        self.Embed = MyEmbdxDLayer(args, in_channels=in_channels, out_fmt='NLC')
        in_channels = self.Embed.out_channels

        self.InLinear = MyInitTower(args, in_channels=in_channels, data_fmt='NLC')
        in_channels = self.InLinear.out_channels

        self.Seq2Mat = Seq2MatCastxD(method='concat', in_fmt='NLC', out_fmt='NHWC')
        in_channels = in_channels # due to concatenation

        # self.Linear = MyLinearTower(args, in_channels=in_channels, data_fmt='NHWC')
        # in_channels = self.Linear.out_channels

        # Reshape data to NLC before feeding
        self.AttnEncoder = MyAttnxDTower(args, in_channels=in_channels, data_fmt='NHWC')
        in_channels = self.AttnEncoder.out_channels

        self.OutLinear = MyReturnTower(args, in_channels=in_channels, data_fmt='NHWC')

    def forward(self, x, seqs_len=None):

        x, seqs_len = self.Embed(x, seqs_len=seqs_len)
        x = self.InLinear(x, seqs_len=seqs_len)

        # x = self.Seq2Mat(x, x)
        x = self.Seq2Mat(*mi.chunk(x, 2, axis=-1))

        # x = self.Linear(x, seqs_len=seqs_len)

        x = self.AttnEncoder(x, seqs_len=seqs_len ** 2)

        x = (x + mi.transpose(x, perm=[0, 2, 1, 3])) / 2.0

        x = self.OutLinear(x, seqs_len=seqs_len)

        return x # mi.squeeze(x[:,:,1], axis=-1)

    def summary(self, input_size=None):
        if input_size is None:
            input_size = self.Embed.in_shapes
        return mi.summary(self, input_size)


class Seq2Mat_Attn1DConv2DNet(nn.Layer):
    """   """
    def __init__(self, args):
        super(Seq2Mat_Attn1DConv2DNet, self).__init__()

        self.data_fmt = args.input_fmt.upper()
        self.in_channels = int(args.input_dim)
        self.seq2mat_method = args.seq2mat_method.upper()

        in_channels = self.in_channels # keep record of the running channel dim

        self.Embed = MyEmbdxDLayer(args, in_channels=in_channels, out_fmt='NLC')
        in_channels = self.Embed.out_channels

        self.InLinear = MyInitTower(args, in_channels=in_channels, data_fmt='NLC')
        in_channels = self.InLinear.out_channels

        self.Attn = MyAttnxDTower(args, in_channels=in_channels, data_fmt='NLC')
        in_channels = self.Attn.out_channels

        # May consider to incorporate the following in the Seq2MatCastxD() layer
        if self.seq2mat_method in ['CONCAT', 'CONCATENATE']:
            self.Seq2MatPreLinear1 = nn.Linear(in_channels, args.conv2d_dim[0] // 2)
            self.Seq2MatPreLinear2 = nn.Linear(in_channels, args.conv2d_dim[0] // 2)
            in_channels = args.conv2d_dim[0] // 2
        else:
            self.Seq2MatPreLinear1 = nn.Linear(in_channels, args.conv2d_dim[0])
            self.Seq2MatPreLinear2 = nn.Linear(in_channels, args.conv2d_dim[0])
            in_channels = args.conv2d_dim[0]

        self.Seq2Mat = Seq2MatCastxD(method=self.seq2mat_method, in_fmt='NLC', out_fmt='NHWC')
        in_channels2d = in_channels * self.Seq2Mat.mul_channels

        # self.Linear = MyLinearTower(args, in_channels=in_channels, data_fmt='NHWC')
        # in_channels = self.Linear.out_channels

        # Add an fc layer after transforming to matrix
        out_channels2d = args.conv2d_dim[0] # in_channels / self.Seq2Mat.mul_channels
        self.PostSeq2Mat = nn.Sequential(
            nn.Linear(in_channels2d, out_channels2d, bias_attr=False),
            nn.LayerNorm(out_channels2d),
            nn.Swish(),
        )
        in_channels2d = out_channels2d

        self.Conv2D = MyConv2DTower(args, in_channels=in_channels2d, data_fmt='NHWC')
        in_channels2d = self.Conv2D.out_channels

        # if self.Conv2D.resnet:
        #     self.PostConv2d = nn.Sequential(
        #         nn.Swish(),
        #         nn.LayerNorm(in_channels2d),
        #         )
        # elif self.Conv2D.pre_act_norm:
        #     self.PostConv2d = nn.Sequential(
        #         nn.Swish(),
        #         nn.LayerNorm(in_channels2d),
        #         )
        # else:
        #     self.PostConv2d = lambda x: x

        self.Out2DLinear = MyReturnTower(args, in_channels=in_channels2d, data_fmt='NHWC')

    def forward(self, x, seqs_len=None):

        x, seqs_len = self.Embed(x, seqs_len=seqs_len)

        x = self.InLinear(x, seqs_len=seqs_len)

        x = self.Attn(x, seqs_len=seqs_len)

        x = self.Seq2Mat(self.Seq2MatPreLinear1(x), self.Seq2MatPreLinear2(x))
        x = self.PostSeq2Mat(x)
        # x = self.Linear(x, seqs_len=seqs_len)

        x = self.Conv2D(x, seqs_len=seqs_len)
        # x = self.PostConv2d(x)

        x = (x + mi.transpose(x, perm=[0, 2, 1, 3])) / 2.0
        x = self.Out2DLinear(x, seqs_len=seqs_len)

        if x.shape[-1] == 1:
            return mi.squeeze(x, -1)
        else:
            return x

    def summary(self, input_size=None):
        if input_size is None:
            input_size = self.Embed.in_shapes
        return mi.summary(self, input_size)


class Seq2MatNum_Attn1DConv2DLSTMNet(nn.Layer):
    """   """
    def __init__(self, args):
        super(Seq2MatNum_Attn1DConv2DLSTMNet, self).__init__()

        self.data_fmt = args.input_fmt.upper()
        self.in_channels = int(args.input_dim)
        self.seq2mat_method = args.seq2mat_method.upper()

        in_channels = self.in_channels # keep record of the running channel dim

        self.Embed = MyEmbdxDLayer(args, in_channels=in_channels, out_fmt='NLC')
        in_channels = self.Embed.out_channels

        self.InLinear = MyInitTower(args, in_channels=in_channels, data_fmt='NLC')
        in_channels = self.InLinear.out_channels

        self.Attn = MyAttnxDTower(args, in_channels=in_channels, data_fmt='NLC')
        in_channels = self.Attn.out_channels

        if self.seq2mat_method in ['CONCAT', 'CONCATENATE']:
            self.Seq2MatPreLinear1 = nn.Linear(in_channels, args.conv2d_dim[0] // 2)
            self.Seq2MatPreLinear2 = nn.Linear(in_channels, args.conv2d_dim[0] // 2)
            in_channels = args.conv2d_dim[0] // 2
        else:
            self.Seq2MatPreLinear1 = nn.Linear(in_channels, args.conv2d_dim[0])
            self.Seq2MatPreLinear2 = nn.Linear(in_channels, args.conv2d_dim[0])
            in_channels = args.conv2d_dim[0]

        self.Seq2Mat = Seq2MatCastxD(method=self.seq2mat_method, in_fmt='NLC', out_fmt='NHWC')
        in_channels2d = in_channels * self.Seq2Mat.mul_channels

        # self.Linear = MyLinearTower(args, in_channels=in_channels2d, data_fmt='NHWC')
        # in_channels2d = self.Linear.out_channels

        # Add an fc layer after transforming to matrix
        out_channels2d = args.conv2d_dim[0] # in_channels2d / self.Seq2Mat.mul_channels
        self.MatLinear = nn.Sequential(
            nn.Linear(in_channels2d, out_channels2d),
            nn.Swish(),
            nn.LayerNorm(out_channels2d),
            )
        in_channels2d = out_channels2d

        self.Conv2D = MyConv2DTower(args, in_channels=in_channels2d, data_fmt='NHWC')
        in_channels2d = self.Conv2D.out_channels

        # add act/norm if pre_act_norm or resnet for conv2d
        if self.Conv2D.pre_act_norm or self.Conv2D.resnet:
            self.PostConv2d = nn.Sequential(
                nn.Swish(),
                nn.LayerNorm(in_channels2d),
                )

        self.OutLinear = MyReturnTower(args, in_channels=in_channels2d, data_fmt='NHWC')

        # add LSTM tower to predict F1 score

        f1lstm_num = min([args.lstm_num, 3])
        f1lstm_dim = min(args.lstm_dim + [32])

        self.LSTM1D = MyLSTM1DTower(args, in_channels=in_channels, data_fmt='NLC',
            lstm_num=f1lstm_num, lstm_dim=f1lstm_dim, return_states=True)
        in_channels = self.LSTM1D.out_channels

        self.F1Linear = MyReturnTower(args, in_channels=in_channels, data_fmt='NLC',
            linear_dim=[32, 32, 1],
            linear_num=1,
            is_return=True,
            )

    def forward(self, x, seqs_len=None):

        x, seqs_len = self.Embed(x, seqs_len=seqs_len)

        x = self.InLinear(x, seqs_len=seqs_len)

        x = self.Attn(x, seqs_len=seqs_len)

        x2d = self.Seq2Mat(self.Seq2MatPreLinear1(x), self.Seq2MatPreLinear2(x))

        x2d = self.MatLinear(x2d)
        # x = self.Linear(x, seqs_len=seqs_len)
        x2d = self.Conv2D(x2d, seqs_len=seqs_len)
        if self.Conv2D.pre_act_norm or self.Conv2D.resnet:
            x2d = self.PostConv2d(x2d)

        x2d = (x2d + mi.transpose(x2d, perm=[0, 2, 1, 3])) / 2.0
        x2d = self.OutLinear(x2d, seqs_len=seqs_len)

        if x2d.shape[-1] == 1:
            x2d = mi.squeeze(x2d, -1)

        x, (h, c) = self.LSTM1D(x.detach(), seqs_len=seqs_len)
        # c: [num_layer*num_direction, batch_size, hidden_size]
        x = mi.reshape(mi.transpose(c, perm=[1, 0, 2]), [len(x), -1])
        f1 = F.sigmoid(self.F1Linear(x, seqs_len=seqs_len))

        return [x2d, f1]

    def summary(self, input_size=None):
        if input_size is None:
            input_size = self.Embed.in_shapes
        return mi.summary(self, input_size)


class Seq2MatSeq_Attn1DConv2DConv1DNet(nn.Layer):
    """ """
    def __init__(self, args):
        super(Seq2MatSeq_Attn1DConv2DConv1DNet, self).__init__()

        self.data_fmt = args.input_fmt.upper()
        self.in_channels = int(args.input_dim)
        self.seq2mat_method = args.seq2mat_method.upper()
        self.label_num = 1 if isinstance(args.label_genre, str) else len(args.label_genre)

        in_channels = self.in_channels # keep record of the running channel dim

        self.Embed = MyEmbdxDLayer(args, in_channels=in_channels, out_fmt='NLC')
        in_channels = self.Embed.out_channels

        self.InLinear = MyInitTower(args, in_channels=in_channels, data_fmt='NLC')
        in_channels = self.InLinear.out_channels

        self.Attn = MyAttnxDTower(args, in_channels=in_channels, data_fmt='NLC')
        in_channels = self.Attn.out_channels

        ####### Conv2D Block

        if self.seq2mat_method in ['CONCAT', 'CONCATENATE']:
            self.PreSeq2MatLinear1 = nn.Linear(in_channels, args.conv2d_dim[0] // 2)
            self.PreSeq2MatLinear2 = nn.Linear(in_channels, args.conv2d_dim[0] // 2)
            in_channels2d = args.conv2d_dim[0] // 2
        else:
            self.PreSeq2MatLinear1 = nn.Linear(in_channels, args.conv2d_dim[0])
            self.PreSeq2MatLinear2 = nn.Linear(in_channels, args.conv2d_dim[0])
            in_channels2d = args.conv2d_dim[0]

        self.Seq2Mat = Seq2MatCastxD(method=self.seq2mat_method, in_fmt='NLC', out_fmt='NHWC')
        in_channels2d = in_channels2d * self.Seq2Mat.mul_channels

        out_channels2d = args.conv2d_dim[0] # in_channels / self.Seq2Mat.mul_channels
        self.PostSq2Mat = nn.Sequential(
            nn.Linear(in_channels2d, out_channels2d, bias_attr=False),
            nn.LayerNorm(out_channels2d),
            nn.Swish(),
        )
        in_channels2d = out_channels2d

        self.Conv2D = MyConv2DTower(args, in_channels=in_channels2d, data_fmt='NHWC')
        in_channels2d = self.Conv2D.out_channels

        linear_dim2d = args.return_dim
        linear_dim2d[-1] = 2
        self.OutLinear2D = MyReturnTower(args, in_channels=in_channels2d,
            linear_dim=linear_dim2d, data_fmt='NHWC')

        ####### Conv1D blocks

        self.Conv1DSin = MyConv1DTower(args, in_channels=in_channels, data_fmt='NLC')

        self.Conv1DCos = MyConv1DTower(args, in_channels=in_channels, data_fmt='NLC')

        args.return_dim[-1] = 8
        in_channels = self.Conv1DSin.out_channels
        self.OutLinearSin = MyReturnTower(args, in_channels=in_channels, data_fmt='NLC')

        in_channels = self.Conv1DCos.out_channels
        self.OutLinearCos = MyReturnTower(args, in_channels=in_channels, data_fmt='NLC')

    def forward(self, x, seqs_len=None):

        x, seqs_len = self.Embed(x, seqs_len=seqs_len)

        x = self.InLinear(x, seqs_len=seqs_len)

        x = self.Attn(x, seqs_len=seqs_len)

        ####### Conv2D block
        if self.seq2mat_method in ['MULTIPLY']:
            x2d = self.Seq2Mat(self.Seq2MatPreLinear1(x), self.Seq2MatPreLinear2(x))
        else:
            x2d = self.Seq2Mat(x, x)

        x2d = self.PostSq2Mat(x2d)
        # x = self.Linear(x, seqs_len=seqs_len)
        x2d = self.Conv2D(x2d, seqs_len=seqs_len)

        # x = mi.transpose(x, perm=[0, 2, 3, 1]) # --> [N, L, L, 2*conv2d_dim[-1]]
        # x = (x + mi.transpose(x, perm=[0, 2, 1, 3])) / 2.0
        x2d = (x2d + mi.transpose(x2d, perm=[0, 2, 1, 3])) / 2.0

        x2d = self.OutLinear2D(x2d, seqs_len=seqs_len)
        if x2d.shape[-1] == 1:
            x2d = mi.squeeze(x, -1)

        # Conv1D blocks
        x_sin = self.Conv1DSin(x, seqs_len)
        x_sin = self.OutLinearSin(x_sin, seqs_len)
        x_sin = mi.sin(x_sin)
        # x_sin = F.softsign(x_sin)

        x_cos = self.Conv1DCos(x, seqs_len)
        x_cos = self.OutLinearCos(x_cos, seqs_len)
        x_cos = mi.cos(x_cos)

        x1d = mi.stack([x_sin, x_cos], axis=-1)

        return [x2d, x1d]

    def summary(self, input_size=None):
        if input_size is None:
            input_size = self.Embed.in_shapes
        return mi.summary(self, input_size)


class Seq2Mat_LSTMConv2DNet(nn.Layer):
    """ This ignores all inter-residue information  """
    def __init__(self, args):
        super(Seq2Mat_LSTMConv2DNet, self).__init__()

        self.data_fmt = args.input_fmt.upper()
        self.in_channels = int(args.input_dim)
        self.seq2mat_method = args.seq2mat_method.upper()

        in_channels = self.in_channels # keep record of the running channel dim

        self.Embed = MyEmbdxDLayer(args, in_channels=in_channels, out_fmt='NLC')
        in_channels = self.Embed.out_channels

        self.InLinear = MyInitTower(args, in_channels=in_channels, data_fmt='NLC')
        in_channels = self.InLinear.out_channels

        self.LSTM = MyLSTM1DTower(args, in_channels=in_channels, data_fmt='NLC')
        in_channels = self.LSTM.out_channels

        # if self.LSTM.resnet:
        #     self.PostLSTM = nn.Sequential(
        #         nn.Swish(),
        #         nn.LayerNorm(in_channels),
        #     )
        # elif self.LSTM.pre_act_norm:
        #     self.PostLSTM = nn.LayerNorm(in_channels)
        # else:
        #     self.PostLSTM = lambda x: x

        # May consider to incorporate the following in the Seq2MatCastxD() layer
        if self.seq2mat_method in ['CONCAT', 'CONCATENATE']:
            self.PreSeq2MatLinear1 = nn.Linear(in_channels, args.conv2d_dim[0] // 2)
            self.PreSeq2MatLinear2 = nn.Linear(in_channels, args.conv2d_dim[0] // 2)
            in_channels = args.conv2d_dim[0] // 2
        else:
            self.PreSeq2MatLinear1 = nn.Linear(in_channels, args.conv2d_dim[0])
            self.PreSeq2MatLinear2 = nn.Linear(in_channels, args.conv2d_dim[0])
            in_channels = args.conv2d_dim[0]

        self.Seq2Mat = Seq2MatCastxD(method=self.seq2mat_method, in_fmt='NLC', out_fmt='NHWC')
        in_channels2d = in_channels * self.Seq2Mat.mul_channels

        out_channels2d = args.conv2d_dim[0] # in_channels / self.Seq2Mat.mul_channels
        self.PostSeq2Mat = nn.Sequential(
            nn.Linear(in_channels2d, out_channels2d, bias_attr=False),
            nn.LayerNorm(out_channels2d),
            nn.Swish(),
            )
        in_channels2d = out_channels2d

        # MyLinearTower(args, in_channels=in_channels, data_fmt='NHWC')
        # in_channels = self.MatLinear.out_channels

        self.Conv2D = MyConv2DTower(args, in_channels=in_channels2d, data_fmt='NHWC')
        in_channels2d = self.Conv2D.out_channels

        # if self.Conv2D.resnet:
        #     self.PostConv2d = nn.Sequential(
        #         nn.Swish(),
        #         nn.LayerNorm(in_channels2d),
        #         )
        # elif self.Conv2D.pre_act_norm:
        #     self.PostConv2d = nn.Sequential(
        #         nn.Swish(),
        #         nn.LayerNorm(in_channels2d),
        #         )
        # else:
        #     self.PostConv2d = lambda x: x

        self.OutLinear = MyReturnTower(args, in_channels=in_channels2d, data_fmt='NHWC')

    def forward(self, x, seqs_len=None):

        x, seqs_len = self.Embed(x, seqs_len=seqs_len)
        x = self.InLinear(x, seqs_len=seqs_len)
        x = self.LSTM(x, seqs_len=seqs_len)
        # x = self.PostLSTM(x)

        x = self.Seq2Mat(self.PreSeq2MatLinear1(x), self.PreSeq2MatLinear2(x))
        x = self.PostSeq2Mat(x)

        x = self.Conv2D(x, seqs_len=seqs_len)
        # x = self.PostConv2d(x)

        x = (x + mi.transpose(x, perm=[0, 2, 1, 3])) / 2.0
        x = self.OutLinear(x, seqs_len=seqs_len)

        if x.shape[-1] == 1:
            return mi.squeeze(x, -1)
        else:
            return x

    def summary(self, input_size=None):
        if input_size is None:
            input_size = self.Embed.in_shapes
        return mi.summary(self, input_size)


class Seq2MatNum_LSTMConv2DLSTMNet(nn.Layer):
    """ This ignores all inter-residue information  """
    def __init__(self, args):
        super(Seq2MatNum_LSTMConv2DLSTMNet, self).__init__()

        self.data_fmt = args.input_fmt.upper()
        self.in_channels = int(args.input_dim)
        self.seq2mat_method = args.seq2mat_method.upper()

        in_channels = self.in_channels # keep record of the running channel dim

        self.Embed = MyEmbdxDLayer(args, in_channels=in_channels, out_fmt='NLC')
        in_channels = self.Embed.out_channels

        self.InLinear = MyInitTower(args, in_channels=in_channels, data_fmt='NLC')
        in_channels = self.InLinear.out_channels

        self.LSTM = MyLSTM1DTower(args, in_channels=in_channels, data_fmt='NLC')
        in_channels = self.LSTM.out_channels

        if self.seq2mat_method in ['CONCAT', 'CONCATENATE']:
            self.PreSeq2MatLinear1 = nn.Linear(in_channels, args.conv2d_dim[0] // 2)
            self.PreSeq2MatLinear2 = nn.Linear(in_channels, args.conv2d_dim[0] // 2)
            in_channels2d = args.conv2d_dim[0] // 2
        else:
            self.PreSeq2MatLinear1 = nn.Linear(in_channels, args.conv2d_dim[0])
            self.PreSeq2MatLinear2 = nn.Linear(in_channels, args.conv2d_dim[0])
            in_channels2d = args.conv2d_dim[0]

        self.Seq2Mat = Seq2MatCastxD(method=self.seq2mat_method, in_fmt='NLC', out_fmt='NHWC')
        in_channels2d = in_channels2d * self.Seq2Mat.mul_channels

        out_channels2d = args.conv2d_dim[0] # in_channels / self.Seq2Mat.mul_channels
        self.PostSeq2Mat = nn.Sequential(
            nn.Linear(in_channels2d, out_channels2d, bias_attr=False),
            nn.LayerNorm(out_channels2d),
            nn.Swish(),
            )
        in_channels2d = out_channels2d

        # MyLinearTower(args, in_channels=in_channels, data_fmt='NHWC')
        # in_channels = self.MatLinear.out_channels

        self.Conv2D = MyConv2DTower(args, in_channels=in_channels2d, data_fmt='NHWC')
        in_channels2d = self.Conv2D.out_channels

        self.OutLinear = MyReturnTower(args, in_channels=in_channels2d, data_fmt='NHWC')

        # add LSTM tower to predict F1 score

        f1lstm_num = min([args.lstm_num, 3])
        f1lstm_dim = min(args.lstm_dim + [32])

        self.F1LSTM = MyLSTM1DTower(args, in_channels=in_channels, data_fmt='NLC',
            lstm_num=f1lstm_num, lstm_dim=f1lstm_dim, train_initial=True, return_states=True)
        in_channels = self.F1LSTM.out_channels

        self.F1Linear = MyReturnTower(args, in_channels=in_channels, data_fmt='NLC',
            linear_dim=[32, 32, 1],
            linear_num=1,
            is_return=True,
            )

    def forward(self, x, seqs_len=None):

        x, seqs_len = self.Embed(x, seqs_len=seqs_len)
        x = self.InLinear(x, seqs_len=seqs_len)
        x = self.LSTM(x, seqs_len=seqs_len)

        x2d = self.Seq2Mat(self.PreSeq2MatLinear1(x), self.PreSeq2MatLinear2(x))

        x2d = self.PostSeq2Mat(x2d)
        x2d = self.Conv2D(x2d, seqs_len=seqs_len)

        x2d = (x2d + mi.transpose(x2d, perm=[0, 2, 1, 3])) / 2.0
        x2d = self.OutLinear(x2d, seqs_len=seqs_len)

        if x2d.shape[-1] == 1:
            x2d = mi.squeeze(x2d, -1)

        x, (h, c) = self.F1LSTM(x.detach(), seqs_len=seqs_len)
        # c: [num_layer*num_direction, batch_size, hidden_size]
        x = mi.reshape(mi.transpose(c, perm=[1, 0, 2]), [len(x), -1])
        f1 = F.sigmoid(self.F1Linear(x, seqs_len=seqs_len))

        return [x2d, f1]


    def summary(self, input_size=None):
        if input_size is None:
            input_size = self.Embed.in_shapes
        return mi.summary(self, input_size)


class Seq2MatSeq_LSTMConv2DConv1DNet(nn.Layer):
    """ """
    def __init__(self, args):
        super(Seq2MatSeq_LSTMConv2DConv1DNet, self).__init__()

        self.data_fmt = args.input_fmt.upper()
        self.in_channels = int(args.input_dim)
        self.seq2mat_method = args.seq2mat_method.upper()
        self.label_num = 1 if isinstance(args.label_genre, str) else len(args.label_genre)

        in_channels = self.in_channels # keep record of the running channel dim

        self.Embed = MyEmbdxDLayer(args, in_channels=in_channels, out_fmt='NLC')
        in_channels = self.Embed.out_channels

        self.InLinear = MyInitTower(args, in_channels=in_channels, data_fmt='NLC')
        in_channels = self.InLinear.out_channels

        self.LSTM = MyLSTM1DTower(args, in_channels=in_channels, data_fmt='NLC')
        in_channels = self.LSTM.out_channels

        if self.seq2mat_method in ['CONCAT', 'CONCATENATE']:
            self.PreSeq2MatLinear1 = nn.Linear(in_channels, args.conv2d_dim[0] // 2)
            self.PreSeq2MatLinear2 = nn.Linear(in_channels, args.conv2d_dim[0] // 2)
            in_channels2d = args.conv2d_dim[0] // 2
        else:
            self.PreSeq2MatLinear1 = nn.Linear(in_channels, args.conv2d_dim[0])
            self.PreSeq2MatLinear2 = nn.Linear(in_channels, args.conv2d_dim[0])
            in_channels2d = args.conv2d_dim[0]

        self.Seq2Mat = Seq2MatCastxD(method=self.seq2mat_method, in_fmt='NLC', out_fmt='NHWC')
        in_channels2d = in_channels2d * self.Seq2Mat.mul_channels

        out_channels2d = args.conv2d_dim[0] # in_channels / self.Seq2Mat.mul_channels
        self.PostSeq2Mat = nn.Sequential(
            nn.Linear(in_channels2d, out_channels2d, bias_attr=False),
            nn.LayerNorm(out_channels2d),
            nn.Swish(),
            )
        in_channels2d = out_channels2d

        # MyLinearTower(args, in_channels=in_channels, data_fmt='NHWC')
        # in_channels = self.MatLinear.out_channels

        self.Conv2D = MyConv2DTower(args, in_channels=in_channels2d, data_fmt='NHWC')
        in_channels2d = self.Conv2D.out_channels

        linear_dim2d = [in_channels2d] * 3 + [2]
        self.OutLinear2D = MyReturnTower(args, in_channels=in_channels2d, 
            linear_dim=linear_dim2d, data_fmt='NHWC')

        # add Conv1D towers to predict Seq (the torsion angles for now)

        out_channels1d = args.conv1d_dim[0]
        self.PreConv1D = nn.Sequential(
            nn.Linear(in_channels, out_channels1d, bias_attr=False),
            nn.LayerNorm(out_channels1d),
            nn.Swish(),
            )
        in_channels = out_channels1d
        
        self.Conv1D = MyConv1DTower(args, in_channels=in_channels, data_fmt='NLC')
        in_channels = self.Conv1D.out_channels

        linear_dim1d = [in_channels] * 3 + [16]
        self.OutLinear1D = MyReturnTower(args, in_channels=in_channels,
            linear_dim=linear_dim1d, data_fmt='NLC')

    def forward(self, x, seqs_len=None):

        x, seqs_len = self.Embed(x, seqs_len=seqs_len)
        x = self.InLinear(x, seqs_len=seqs_len)
        x = self.LSTM(x, seqs_len=seqs_len)

        x2d = self.Seq2Mat(self.PreSeq2MatLinear1(x), self.PreSeq2MatLinear2(x))

        x2d = self.PostSeq2Mat(x2d)
        x2d = self.Conv2D(x2d, seqs_len=seqs_len)

        x2d = (x2d + mi.transpose(x2d, perm=[0, 2, 1, 3])) / 2.0
        x2d = self.OutLinear2D(x2d, seqs_len=seqs_len)

        if x2d.shape[-1] == 1:
            x2d = mi.squeeze(x2d, -1)

        if self.label_num == 1:
            return x2d
        else:
            x = self.PreConv1D(x)
            x = self.Conv1D(x, seqs_len=seqs_len)
            x = self.OutLinear1D(x, seqs_len=seqs_len)

            return [x2d, mi.stack(mi.chunk(x, 2, axis=-1), axis=-1)] # x.reshape(x.shape[:-1] + [x.shape[-1] // 2, 2])]


    def summary(self, input_size=None):
        if input_size is None:
            input_size = self.Embed.in_shapes
        return mi.summary(self, input_size)


class Seq2Mat_LSTMConv2DCRFNet(nn.Layer):
    """ This ignores all inter-residue information  """
    def __init__(self, args):
        super(Seq2Mat_LSTMConv2DCRFNet, self).__init__()

        self.data_fmt = args.input_fmt.upper()
        # self.data_ndim = int(args.input_ndim)
        self.in_channels = int(args.input_dim)
        self.seq2mat_method = args.seq2mat_method.upper()

        in_channels = self.in_channels # keep record of the running channel dim

        self.Embed = MyEmbdxDLayer(args, in_channels=in_channels, out_fmt='NLC')
        in_channels = self.Embed.out_channels

        self.InLinear = MyInitTower(args, in_channels=in_channels, data_fmt='NLC')
        in_channels = self.InLinear.out_channels

        self.LSTM = MyLSTM1DTower(args, in_channels=in_channels, data_fmt='NLC')
        in_channels = self.LSTM.out_channels

        self.Seq2Mat = Seq2MatCastxD(method=self.seq2mat_method, in_fmt='NLC', out_fmt='NHWC')
        in_channels = in_channels * self.Seq2Mat.mul_channels

        out_channels = args.conv2d_dim[0] # in_channels / self.Seq2Mat.mul_channels
        self.MatLinear = nn.Sequential(
            nn.Linear(in_channels, out_channels),
            nn.Swish(),
            nn.LayerNorm(out_channels),
        )
        in_channels = out_channels

        # MyLinearTower(args, in_channels=in_channels, data_fmt='NHWC')
        # in_channels = self.MatLinear.out_channels

        self.Conv2D = MyConv2DTower(args, in_channels=in_channels, data_fmt='NHWC')
        in_channels = self.Conv2D.out_channels

        if self.Conv2D.pre_act_norm or self.Conv2D.resnet:
            self.PostConv2d = nn.Sequential(
                nn.Swish(),
                nn.LayerNorm(in_channels),
                )

        self.OutLinear = MyReturnTower(args, in_channels=in_channels, data_fmt='NHWC')
        in_channels = self.OutLinear.out_channels

        dropout = 0.1
        self.CRF = nn.Sequential(
            nn.Sigmoid(),
            nn.Dropout(dropout),
            nn.Conv2D(in_channels, in_channels, 11, padding='SAME', data_format='NHWC'),
            nn.LeakyReLU(),
            nn.Dropout(dropout),
            nn.Conv2D(in_channels, in_channels, 7, padding='SAME', data_format='NHWC'),
            nn.LeakyReLU(),
            nn.Dropout(dropout),
            nn.Conv2D(in_channels, in_channels, 5, padding='SAME', data_format='NHWC'),
        )

    def forward(self, x, seqs_len=None):

        x, seqs_len = self.Embed(x, seqs_len=seqs_len)
        x = self.InLinear(x, seqs_len=seqs_len)
        x = self.LSTM(x, seqs_len=seqs_len)
        x = self.Seq2Mat(x, x)
        x = self.MatLinear(x)
        x = self.Conv2D(x, seqs_len=seqs_len)
        if self.Conv2D.pre_act_norm or self.Conv2D.resnet:
            x = self.PostConv2d(x)
        x = self.OutLinear(x, seqs_len=seqs_len)
        x = self.CRF(x)

        x = (x + mi.transpose(x, perm=[0, 2, 1, 3])) / 2.0

        if x.shape[-1] == 1:
            return mi.squeeze(x, -1)
        else:
            return x

    def summary(self, input_size=None):
        if input_size is None:
            input_size = self.Embed.in_shapes
        return mi.summary(self, input_size)


class Seq2Mat_Attn1DAttn2DNet(nn.Layer):
    """   """
    def __init__(self, args):
        super(Seq2Mat_Attn1DAttn2DNet, self).__init__()

        self.data_fmt = args.input_fmt.upper()
        # self.data_ndim = int(args.input_ndim)
        self.in_channels = int(args.input_dim)
        self.seq2mat_method = args.seq2mat_method.upper()

        in_channels = self.in_channels # keep record of the running channel dim

        self.Embed = MyEmbdxDLayer(args, in_channels=in_channels, out_fmt='NLC')
        in_channels = self.Embed.out_channels

        self.InLinear = MyInitTower(args, in_channels=in_channels, data_fmt='NLC')
        in_channels = self.InLinear.out_channels

        self.Attn1D = MyAttnxDTower(args, in_channels=in_channels, data_fmt='NLC')
        in_channels = self.Attn1D.out_channels

        self.Seq2Mat = Seq2MatCastxD(method=self.seq2mat_method, in_fmt='NLC', out_fmt='NHWC')
        in_channels = in_channels * self.Seq2Mat.mul_channels

        out_channels = in_channels // self.Seq2Mat.mul_channels
        self.MatLinear = [
            nn.Linear(in_channels, out_channels),
            nn.LeakyReLU(),
            nn.LayerNorm(out_channels),
            ]
        in_channels = out_channels

        self.MatLinear = nn.Sequential(*self.MatLinear)

        # self.Linear = MyLinearTower(args, in_channels=in_channels, data_fmt='NHWC')
        # in_channels = self.Linear.out_channels

        self.Attn2D = MyAttnxDTower(misc.Struct(vars(args)).update(
            attn_axis=[1,2], attn_method=args.attn_method),
            in_channels=in_channels, data_fmt='NHWC')
        in_channels = self.Attn2D.out_channels

        # self.Conv2D = MyConv2DTower(args, in_channels=in_channels, data_fmt='NHWC')
        # in_channels = self.Conv2D.out_channels

        self.OutLinear = MyReturnTower(args, in_channels=in_channels, data_fmt='NHWC')

    def forward(self, x, seqs_len=None):

        x, seqs_len = self.Embed(x, seqs_len=seqs_len)

        x = self.InLinear(x, seqs_len=seqs_len)

        x = self.Attn1D(x, seqs_len=seqs_len)

        x = self.Seq2Mat(x, x)
        x = self.MatLinear(x)

        x = self.Attn2D(x, seqs_len=seqs_len)

        # x = self.Linear(x, seqs_len=seqs_len)
        # x = self.Conv2D(x, seqs_len=seqs_len)

        if self.seq2mat_method in ['ADD', 'MULTIPLY']:
            pass
        else:
            x = (x + mi.transpose(x, perm=[0, 2, 1, 3])) / 2.0

        x = self.OutLinear(x, seqs_len=seqs_len)

        return x # mi.squeeze(x[:,:,1], axis=-1)

    def summary(self, input_size=None):
        if input_size is None:
            input_size = self.Embed.in_shapes
        return mi.summary(self, input_size)


class Seq2Mat_Attn1DAttn2DConv2DNet(nn.Layer):
    """   """
    def __init__(self, args):
        super(Seq2Mat_Attn1DAttn2DConv2DNet, self).__init__()

        self.data_fmt = args.input_fmt.upper()
        # self.data_ndim = int(args.input_ndim)
        self.in_channels = int(args.input_dim)

        in_channels = self.in_channels # keep record of the running channel dim

        self.Embed = MyEmbdxDLayer(args, in_channels=in_channels, out_fmt='NLC')
        in_channels = self.Embed.out_channels

        self.InLinear = MyInitTower(args, in_channels=in_channels, data_fmt='NLC')
        in_channels = self.InLinear.out_channels

        self.Attn1D = MyAttnxDTower(args, in_channels=in_channels, data_fmt='NLC')
        in_channels = self.Attn1D.out_channels

        self.Seq2Mat = Seq2MatCastxD(method='multiply', in_fmt='NLC', out_fmt='NHWC')
        in_channels = in_channels * 1 # due to concatenation

        # self.Linear = MyLinearTower(args, in_channels=in_channels, data_fmt='NHWC')
        # in_channels = self.Linear.out_channels

        self.Attn2D = MyAttnxDTower(misc.Struct(vars(args)).update(
            attn_axis=[1,2], attn_method='dotproduct'),
            in_channels=in_channels, data_fmt='NHWC')
        in_channels = self.Attn2D.out_channels

        self.Conv2D = MyConv2DTower(args, in_channels=in_channels, data_fmt='NHWC')
        in_channels = self.Conv2D.out_channels

        self.OutLinear = MyReturnTower(args, in_channels=in_channels, data_fmt='NHWC')

    def forward(self, x, seqs_len=None):

        x, seqs_len = self.Embed(x, seqs_len=seqs_len)

        x = self.InLinear(x, seqs_len=seqs_len)

        x = self.Attn1D(x, seqs_len=seqs_len)

        x = self.Seq2Mat(x, x)

        x = self.Attn2D(x, seqs_len=seqs_len)

        # x = self.Linear(x, seqs_len=seqs_len)
        x = self.Conv2D(x, seqs_len=seqs_len)

        # x = mi.transpose(x, perm=[0, 2, 3, 1]) # --> [N, L, L, 2*conv2d_dim[-1]]
        # x = (x + mi.transpose(x, perm=[0, 2, 1, 3])) / 2.0

        x = self.OutLinear(x, seqs_len=seqs_len)

        return x # mi.squeeze(x[:,:,1], axis=-1)

    def summary(self, input_size=None):
        if input_size is None:
            input_size = self.Embed.in_shapes
        return mi.summary(self, input_size)


class Seq2Mat_LSTMAttn2DConv2DNet(nn.Layer):
    """ This ignores all inter-residue information  """
    def __init__(self, args):
        super(Seq2Mat_LSTMAttn2DConv2DNet, self).__init__()

        self.data_fmt = args.input_fmt.upper()
        # self.data_ndim = int(args.input_ndim)
        self.in_channels = int(args.input_dim)

        in_channels = self.in_channels # keep record of the running channel dim

        self.Embed = MyEmbdxDLayer(args, in_channels=in_channels, out_fmt='NLC')
        in_channels = self.Embed.out_channels

        self.InLinear = MyInitTower(args, in_channels=in_channels, data_fmt='NLC')
        in_channels = self.InLinear.out_channels

        self.LSTM = MyLSTM1DTower(args, in_channels=in_channels, data_fmt='NLC')
        in_channels = self.LSTM.out_channels

        self.Seq2Mat = Seq2MatCastxD(method='concat', in_fmt='NLC', out_fmt='NHWC')
        in_channels = in_channels * 2 # due to outer concatenation

        self.Linear = MyLinearTower(args, in_channels=in_channels, data_fmt='NHWC')
        in_channels = self.Linear.out_channels

        self.Attn2D = MyAttnxDTower(args, in_channels=in_channels, data_fmt='NHWC')
        in_channels = self.Attn2D.out_channels

        self.Conv2D = MyConv2DTower(args, in_channels=in_channels, data_fmt='NHWC')
        in_channels = self.Conv2D.out_channels

        self.OutLinear = MyReturnTower(args, in_channels=in_channels, data_fmt='NHWC')

    def forward(self, x, seqs_len):

        x, seqs_len = self.Embed(x, seqs_len=seqs_len)
        x = self.InLinear(x, seqs_len=seqs_len)
        x = self.LSTM(x, seqs_len)

        x = self.Seq2Mat(x, x)

        x = self.Linear(x, seqs_len=seqs_len)

        x = self.Attn2D(x, seqs_len=seqs_len)

        x = self.Conv2D(x, seqs_len=seqs_len)
        x = self.OutLinear(x, seqs_len=seqs_len)

        x = (x + mi.transpose(x, perm=[0, 2, 1, 3])) / 2

        return x # mi.squeeze(x[:,:,1], axis=-1)

    def summary(self, input_size=None):
        if input_size is None:
            input_size = self.Embed.in_shapes
        return mi.summary(self, input_size)


class Seq2Mat_Conv1DLSTMConv2DNet(nn.Layer):
    """  """
    def __init__(self, args):
        super(Seq2Mat_Conv1DLSTMConv2DNet, self).__init__()

        self.data_fmt = args.input_fmt.upper()
        # self.data_ndim = int(args.input_ndim) # 0 or 1
        self.in_channels = int(args.input_dim)

        in_channels = self.in_channels # keep record of the running channel dim

        self.Embed = MyEmbdxDLayer(args, in_channels=in_channels, out_fmt='NLC')
        in_channels = self.Embed.out_channels

        self.InLinear = MyInitTower(args, in_channels=in_channels, data_fmt='NLC')
        in_channels = self.InLinear.out_channels

        self.Conv1D = MyConv1DTower(args, in_channels=in_channels, data_fmt='NLC')
        in_channels = self.Conv1D.out_channels

        self.LSTM = MyLSTM1DTower(args, in_channels=in_channels, data_fmt='NLC')
        in_channels = self.LSTM.out_channels

        self.Seq2Mat = Seq2MatCastxD(method='concat', in_fmt='NLC', out_fmt='NHWC')
        in_channels = in_channels * 1 # due to outer concatenation

        # self.Linear = MyLinearTower(args, in_channels=in_channels, data_fmt='NHWC')
        # in_channels = self.Linear.out_channels

        self.Conv2D = MyConv2DTower(args, in_channels=in_channels, data_fmt='NHWC')
        in_channels = self.Conv2D.out_channels

        self.OutLinear = MyReturnTower(args, in_channels=in_channels, data_fmt='NHWC')

    # @mi.jit.to_static
    def forward(self, x, seqs_len):

        x, seqs_len = self.Embed(x, seqs_len)
        x = self.InLinear(x, seqs_len)
        x = self.Conv1D(x, seqs_len)
        x = self.LSTM(x, seqs_len)

        x = self.Seq2Mat(*mi.chunk(x, 2, axis=-1))

        # x = self.Linear(x, seqs_len)
        x = self.Conv2D(x, seqs_len)

        # x = (x + mi.transpose(x, perm=[0, 2, 1, 3])) / 2
        x = self.OutLinear(x, seqs_len)

        return x # mi.squeeze(x[:,:,1], axis=-1)

    def summary(self, input_size=None):
        if input_size is None:
            input_size = self.Embed.in_shapes
        return mi.summary(self, input_size)


class Seq2Mat_Conv1DLSTMAttn2DNet(nn.Layer):
    """  """
    def __init__(self, args):
        super(Seq2Mat_Conv1DLSTMAttn2DNet, self).__init__()

        self.data_fmt = args.input_fmt.upper()
        # self.data_ndim = int(args.input_ndim) # 0 or 1
        self.in_channels = int(args.input_dim)

        in_channels = self.in_channels # keep record of the running channel dim

        # set_global_initializer(args.param_init)
        self.Embed = MyEmbdxDLayer(args, in_channels=in_channels, out_fmt='NLC')
        in_channels = self.Embed.out_channels

        self.InLinear = MyInitTower(args, in_channels=in_channels, data_fmt='NLC')
        in_channels = self.InLinear.out_channels

        self.Conv1D = MyConv1DTower(args, in_channels=in_channels, data_fmt='NLC')
        in_channels = self.Conv1D.out_channels

        self.LSTM = MyLSTM1DTower(args, in_channels=in_channels, data_fmt='NLC')
        in_channels = self.LSTM.out_channels

        self.Seq2Mat = Seq2MatCastxD(method='multiply', in_fmt='NLC', out_fmt='NHWC')
        in_channels = in_channels * 1 # due to outer concatenation

        # self.Linear = MyLinearTower(args, in_channels=in_channels, data_fmt='NHWC')
        # in_channels = self.Linear.out_channels

        self.Attn2D = MyAttnxDTower(args, in_channels=in_channels, data_fmt='NHWC')
        in_channels = self.Attn2D.out_channels

        self.OutLinear = MyReturnTower(args, in_channels=in_channels, data_fmt='NHWC')

    # @mi.jit.to_static
    def forward(self, x, seqs_len):

        x, seqs_len = self.Embed(x, seqs_len)
        x = self.InLinear(x, seqs_len)
        x = self.Conv1D(x, seqs_len)
        x = self.LSTM(x, seqs_len)

        x = self.Seq2Mat(x, x)
        # x = self.Seq2Mat(*mi.chunk(x, 2, axis=-1))

        # x = self.Linear(x, seqs_len)
        x = self.Attn2D(x, seqs_len)

        x = (x + mi.transpose(x, perm=[0, 2, 1, 3])) / 2
        x = self.OutLinear(x, seqs_len)

        return x # mi.squeeze(x[:,:,1], axis=-1)

    def summary(self, input_size=None):
        if input_size is None:
            input_size = self.Embed.in_shapes
        return mi.summary(self, input_size)


class Seq2Mat_Conv1DLSTMAttn1DConv2DNet(nn.Layer):
    """  """
    def __init__(self, args):
        super(Seq2Mat_Conv1DLSTMAttn1DConv2DNet, self).__init__()

        self.data_fmt = args.input_fmt.upper()
        # self.data_ndim = int(args.input_ndim) # 0 or 1
        self.in_channels = int(args.input_dim)

        in_channels = self.in_channels # keep record of the running channel dim

        # set_global_initializer(args.param_init)
        self.Embed = MyEmbdxDLayer(args, in_channels=in_channels, out_fmt='NLC')
        in_channels = self.Embed.out_channels

        self.InLinear = MyInitTower(args, in_channels=in_channels, data_fmt='NLC')
        in_channels = self.InLinear.out_channels

        self.Conv1D = MyConv1DTower(args, in_channels=in_channels, data_fmt='NLC')
        in_channels = self.Conv1D.out_channels

        self.LSTM = MyLSTM1DTower(args, in_channels=in_channels, data_fmt='NLC')
        in_channels = self.LSTM.out_channels

        self.Attn1D = MyAttnxDTower(args, in_channels=in_channels, data_fmt='NLC')
        in_channels = self.Attn1D.out_channels

        self.Seq2Mat = Seq2MatCastxD(method='multiply', in_fmt='NLC', out_fmt='NHWC')
        in_channels = in_channels * 1 # due to outer concatenation

        # self.Linear = MyLinearTower(args, in_channels=in_channels, data_fmt='NHWC')
        # in_channels = self.Linear.out_channels
        self.Conv2D = MyConv2DTower(args, in_channels=in_channels, data_fmt='NHWC')
        in_channels = self.Conv2D.out_channels

        self.OutLinear = MyReturnTower(args, in_channels=in_channels, data_fmt='NHWC')

    # @mi.jit.to_static
    def forward(self, x, seqs_len):

        x, seqs_len = self.Embed(x, seqs_len)
        x = self.InLinear(x, seqs_len)
        x = self.Conv1D(x, seqs_len)
        x = self.LSTM(x, seqs_len)

        x = self.Attn1D(x, seqs_len)

        x = self.Seq2Mat(x, x)
        # x = self.Seq2Mat(*mi.chunk(x, 2, axis=-1))

        # x = self.Linear(x, seqs_len)
        x = self.Conv2D(x, seqs_len)

        x = (x + mi.transpose(x, perm=[0, 2, 1, 3])) / 2
        x = self.OutLinear(x, seqs_len)

        return x # mi.squeeze(x[:,:,1], axis=-1)

    def summary(self, input_size=None):
        if input_size is None:
            input_size = self.Embed.in_shapes
        return mi.summary(self, input_size)


class Seq2Mat_AttnLSTMConv2DNet(nn.Layer):
    """  """
    def __init__(self, args):
        super(Seq2Mat_AttnLSTMConv2DNet, self).__init__()

        self.data_fmt = args.input_fmt.upper()
        # self.data_ndim = int(args.input_ndim) # 0 or 1
        self.in_channels = int(args.input_dim)

        in_channels = self.in_channels # keep record of the running channel dim

        self.Embed = MyEmbdxDLayer(args, in_channels=in_channels, out_fmt='NLC')
        in_channels = self.Embed.out_channels

        self.InLinear = MyInitTower(args, in_channels=in_channels, data_fmt='NLC')
        in_channels = self.InLinear.out_channels

        self.Attn = MyAttnxDTower(args, in_channels=in_channels, data_fmt='NLC')
        in_channels = self.Attn.out_channels

        self.LSTM = MyLSTM1DTower(args, in_channels=in_channels, data_fmt='NLC')
        in_channels = self.LSTM.out_channels

        self.Seq2Mat = Seq2MatCastxD(method='concat', in_fmt='NLC', out_fmt='NHWC')
        in_channels = in_channels * 2 # due to outer concatenation

        self.Linear = MyLinearTower(args, in_channels=in_channels, data_fmt='NHWC')
        in_channels = self.Linear.out_channels

        self.Conv2D = MyConv2DTower(args, in_channels=in_channels, data_fmt='NHWC')
        in_channels = self.Conv2D.out_channels

        self.OutLinear = MyReturnTower(args, in_channels=in_channels, data_fmt='NHWC')

    def forward(self, x, seqs_len=None):

        x, seqs_len = self.Embed(x, seqs_len=seqs_len)
        x = self.InLinear(x, seqs_len=seqs_len)
        x = self.Attn(x, seqs_len=seqs_len)
        x = self.LSTM(x, seqs_len=seqs_len)

        x = self.Seq2Mat(x, x)

        x = self.Linear(x, seqs_len=seqs_len)
        x = self.Conv2D(x, seqs_len=seqs_len)

        x = (x + mi.transpose(x, perm=[0, 2, 1, 3])) / 2
        x = self.OutLinear(x, seqs_len=seqs_len)

        return x # mi.squeeze(x[:,:,1], axis=-1)

    def summary(self, input_size=None):
        if input_size is None:
            input_size = self.Embed.in_shapes
        return mi.summary(self, input_size)


class Seq2Mat_MXFoldNet(nn.Layer):
    """  """
    def __init__(self, args):
        super(Seq2Mat_MXFoldNet, self).__init__()

        self.data_fmt = args.input_fmt.upper()
        # self.data_ndim = int(args.input_ndim) # 0 or 1
        self.in_channels = int(args.input_dim)

        in_channels = self.in_channels # keep record of the running channel dim

        # set_global_initializer(args.param_init)
        self.Embed = MyEmbdxDLayer(args, in_channels=in_channels, out_fmt='NLC')
        in_channels = self.Embed.out_channels

        self.InLinear = MyInitTower(args, in_channels=in_channels, data_fmt='NLC')
        in_channels = self.InLinear.out_channels

        self.Conv1D = MyConv1DTower(args, in_channels=in_channels, data_fmt='NLC')
        in_channels = self.Conv1D.out_channels

        self.LSTM = MyLSTM1DTower(args, in_channels=in_channels, data_fmt='NLC')
        in_channels = self.LSTM.out_channels

        self.Seq2Mat = Seq2MatCastxD(method='concat', in_fmt='NLC', out_fmt='NHWC')
        in_channels = in_channels * 1 # due to outer concatenation

        # self.Linear = MyLinearTower(args, in_channels=in_channels, data_fmt='NHWC')
        # in_channels = self.Linear.out_channels

        self.Conv2D = MyConv2DTower(args, in_channels=in_channels, data_fmt='NHWC')
        in_channels = self.Conv2D.out_channels

        self.OutLinear = MyReturnTower(args, in_channels=in_channels, data_fmt='NHWC')

    # @mi.jit.to_static
    def forward(self, x, seqs_len):

        x, seqs_len = self.Embed(x, seqs_len)
        x = self.InLinear(x, seqs_len)
        x = self.Conv1D(x, seqs_len)
        x = self.LSTM(x, seqs_len)

        x = self.Seq2Mat(*mi.chunk(x, 2, axis=-1))

        x = self.Conv2D(x, seqs_len)

        x = (x + mi.transpose(x, perm=[0, 2, 1, 3])) / 2
        x = self.OutLinear(x, seqs_len)

        return x # mi.squeeze(x[:,:,1], axis=-1)

    def summary(self, input_size=None):
        if input_size is None:
            input_size = self.Embed.in_shapes
        return mi.summary(self, input_size)


class Seq2Mat_EvoFormerNet(nn.Layer):
    """  just get started, then thought to go with pytorch """
    def __init__(self, args):
        super(Seq2Mat_EvoFormerNet, self).__init__()

        self.data_fmt = args.input_fmt.upper()
        # self.data_ndim = int(args.input_ndim)
        self.in_channels = int(args.input_dim)

        # 1) Sequence pre-layers, starts with [N, L, C]
        in_channels = self.in_channels # keep record of the running channel dim

        self.Embed = MyEmbdxDLayer(args, in_channels=in_channels, out_fmt='NLC')
        in_channels = self.Embed.out_channels

        self.InLinear = MyInitTower(args, in_channels=in_channels, data_fmt='NLC')
        in_channels = self.InLinear.out_channels

        self.Seq2Mat = Seq2MatCastxD(method='concat', in_fmt='NLC', out_fmt='NHWC')
        mat_in_channels = in_channels * 2 # due to outer concatenation

        self.SeqAttn = MyAttnxDTower(args, in_channels=in_channels, data_fmt='NLC')
        in_channels = self.SeqAttn.out_channels

        # self.
        MyMultiHeadAttention(q_dim=in_channels, k_dim=in_channels,
                    v_dim=in_channels, join_method='add')


        self.seq_in = MyLinearTower(args, in_channels=in_channels)
        in_channels = self.seq_in.out_channels
        # 2) Mat pre-layers, starts with [N, L, L C]
        mat_in_channels = 8
        self.mat_in = MyLinearTower(args, in_channels=mat_in_channels)
        mat_in_channels = self.mat_in.out_channels

        self.mat_self_attn = MyMultiHeadAttention(q_dim=mat_in_channels, k_dim=mat_in_channels,
                    v_dim=mat_in_channels, join_method='add')
        mat_in_channels = self.mat_self_attn.out_channels

        # 3) Mutual attention between seq and mat
        self.seq_mat_attn = MyMultiHeadAttention(q_dim=in_channels, k_dim=mat_in_channels,
                    v_dim=mat_in_channels, join_method='concat')
        in_channels = self.seq_mat_attn.out_channels

        self.mat_seq_attn = MyMultiHeadAttention(q_dim=mat_in_channels, k_dim=in_channels,
                    v_dim=in_channels, join_method='concat')
        mat_in_channels = self.mat_seq_attn.out_channels

        # 4) Seq post-layers

        self.seq_lstm = MyLSTM1DTower(args, in_channels=in_channels)
        in_channels = self.seq_lstm.out_channels

        # self.seq2mat = Seq2MatTransform(method='concat', in_fmt='NLC', out_fmt='NCHW')
        # seq_in_channels = seq_in_channels * 2 # due to outer concatenation
        # self.conv2d = MyConv2DTower(args, in_channels=seq_in_channels)
        # seq_in_channels = self.conv2d.out_channels

        self.seq_out = MyReturnTower(args, in_channels=in_channels, data_fmt='NLC')

        # 5) Mat post-layers
        self.mat_conv2d = MyConv2DTower(args, in_channels=mat_in_channels)
        mat_in_channels = self.mat_conv2d.out_channels

        self.mat_out = MyLinearTower(misc.Struct(vars(args)).update(
                data_fmt = 'NLLC',
                linear_dim = args.return_dim,
                linear_num = args.return_num,
                linear_resnet = False,
                ),
                in_channels = in_channels,
                is_return = True,
        )

    def forward(self, seq, mat, seqs_len=None):

        seq, seqs_len = self.Embed(seq, seqs_len=seqs_len)
        seq = self.seq_in(seq, seqs_len=seqs_len)
        seq = self.SeqAttn(seq, seqs_len=seqs_len)

        # mat = self.mat_embed(mat, seqs_len=seqs_len)
        mat = self.mat_in(mat, seqs_len=seqs_len)
        mat = self.mat_self_attn(mat, seqs_len=seqs_len)

        seq = self.seq_mat_attn(seq, mat.reshape([mat.shape[0], -1, mat.shape[-1]]),
                    mat.reshape([mat.shape[0], -1, mat.shape[-1]]))
        seq = self.seq_lstm(seq, seqs_len=seqs_len)

        seq = self.seq2mat(seq, seq)
        seq = self.conv2d(seq, seqs_len=seqs_len)
        seq = mi.transpose(seq, perm=[0, 3, 2, 1]) # --> [N, L, L, 2*conv2d_dim[-1]]

        seq = self.out(seq)

        seq = (seq + mi.transpose(seq, perm=[0, 2, 1, 3])) / 2

        return seq # mi.squeeze(x[:,:,1], axis=-1)

    def summary(self, input_size=None):
        if input_size is None:
            input_size = self.Embed.in_shapes
        return mi.summary(self, input_size)


class SeqMat2Mat_Attn1DConv2DNet(nn.Layer):
    """   """
    def __init__(self, args):
        super(SeqMat2Mat_Attn1DConv2DNet, self).__init__()

        self.data_fmt = args.input_fmt.upper()
        self.in_channels = int(args.input_dim)
        self.seq2mat_method = args.seq2mat_method.upper()

        in_channels = self.in_channels # keep record of the running channel dim

        self.Embed = MyEmbdxDLayer(args, in_channels=in_channels, out_fmt='NLC')
        in_channels = self.Embed.out_channels

        self.InLinear = MyInitTower(args, in_channels=in_channels, data_fmt='NLC')
        in_channels = self.InLinear.out_channels

        self.Attn = MyAttnxDTower(args, in_channels=in_channels, data_fmt='NLC')
        in_channels = self.Attn.out_channels

        if self.seq2mat_method in ['CONCAT', 'CONCATENATE']:
            self.PreSeq2MatLinear1 = nn.Linear(in_channels, args.conv2d_dim[0] // 2)
            self.PreSeq2MatLinear2 = nn.Linear(in_channels, args.conv2d_dim[0] // 2)
            in_channels = args.conv2d_dim[0] // 2
        else:
            self.PreSeq2MatLinear1 = nn.Linear(in_channels, args.conv2d_dim[0])
            self.PreSeq2MatLinear2 = nn.Linear(in_channels, args.conv2d_dim[0])
            in_channels = args.conv2d_dim[0]

        self.Seq2Mat = Seq2MatCastxD(method=self.seq2mat_method, in_fmt='NLC', out_fmt='NHWC')
        in_channels2d = in_channels * self.Seq2Mat.mul_channels

        in_channels2d += 4 # add the bpmat [LL4] here!!!

        out_channels2d = args.conv2d_dim[0] # (in_channels - 4) // self.Seq2Mat.mul_channels
        self.PostSeq2Mat = nn.Sequential(
            nn.Linear(in_channels2d, out_channels2d, bias_attr=False),
            nn.LayerNorm(out_channels2d),
            nn.Swish(),
        )
        in_channels2d = out_channels2d

        self.Conv2D = MyConv2DTower(args, in_channels=in_channels2d, data_fmt='NHWC')
        in_channels2d = self.Conv2D.out_channels

        self.Out2DLinear = MyReturnTower(args, in_channels=in_channels2d, data_fmt='NHWC')

    def forward(self, x, bpmat, seqs_len):

        x, seqs_len = self.Embed(x, seqs_len=seqs_len)

        x = self.InLinear(x, seqs_len=seqs_len)

        x = self.Attn(x, seqs_len=seqs_len)

        x = self.Seq2Mat(self.PreSeq2MatLinear1(x), self.PreSeq2MatLinear2(x))

        x = mi.concat([x, bpmat], axis=-1)

        x = self.PostSeq2Mat(x)
        x = self.Conv2D(x, seqs_len=seqs_len)

        x = (x + mi.transpose(x, perm=[0, 2, 1, 3])) / 2.0
        x = self.Out2DLinear(x, seqs_len=seqs_len)

        if x.shape[-1] == 1:
            return mi.squeeze(x, -1)
        else:
            return x

    def summary(self, input_size=None):
        if input_size is None:
            input_size = self.Embed.in_shapes
        return mi.summary(self, input_size)


class SeqMat2MatNum_Attn1DConv2DLSTMNet(nn.Layer):
    """   """
    def __init__(self, args):
        super(SeqMat2MatNum_Attn1DConv2DLSTMNet, self).__init__()

        self.data_fmt = args.input_fmt.upper()
        self.in_channels = int(args.input_dim)
        self.seq2mat_method = args.seq2mat_method.upper()

        in_channels = self.in_channels # keep record of the running channel dim

        self.Embed = MyEmbdxDLayer(args, in_channels=in_channels, out_fmt='NLC')
        in_channels = self.Embed.out_channels

        self.InLinear = MyInitTower(args, in_channels=in_channels, data_fmt='NLC')
        in_channels = self.InLinear.out_channels

        self.Attn = MyAttnxDTower(args, in_channels=in_channels, data_fmt='NLC')
        in_channels = self.Attn.out_channels

        if self.seq2mat_method in ['CONCAT', 'CONCATENATE']:
            self.Seq2MatPreLinear1 = nn.Linear(in_channels, args.conv2d_dim[0] // 2)
            self.Seq2MatPreLinear2 = nn.Linear(in_channels, args.conv2d_dim[0] // 2)
            in_channels = args.conv2d_dim[0] // 2
        else:
            self.Seq2MatPreLinear1 = nn.Linear(in_channels, args.conv2d_dim[0])
            self.Seq2MatPreLinear2 = nn.Linear(in_channels, args.conv2d_dim[0])
            in_channels = args.conv2d_dim[0]

        self.Seq2Mat = Seq2MatCastxD(method=self.seq2mat_method, in_fmt='NLC', out_fmt='NHWC')
        in_channels2d = in_channels * self.Seq2Mat.mul_channels

        in_channels2d += 4 # add the bpmat [LL4] here!!!

        out_channels2d = args.conv2d_dim[0] # (in_channels - 4) // self.Seq2Mat.mul_channels
        self.Mat2DLinear = nn.Sequential(
            nn.Linear(in_channels2d, out_channels2d),
            nn.Swish(),
            nn.LayerNorm(out_channels2d),
            )
        in_channels2d = out_channels2d

        self.Conv2D = MyConv2DTower(args, in_channels=in_channels2d, data_fmt='NHWC')
        in_channels2d = self.Conv2D.out_channels

        if self.Conv2D.pre_act_norm or self.Conv2D.resnet:
            self.PostConv2d = nn.Sequential(
                nn.Swish(),
                nn.LayerNorm(in_channels2d),
                )

        self.Out2DLinear = MyReturnTower(args, in_channels=in_channels2d, data_fmt='NHWC')

        # add LSTM tower to predict F1 score

        f1lstm_num = min([args.lstm_num, 3])
        f1lstm_dim = min(args.lstm_dim + [32])

        self.LSTM1D = MyLSTM1DTower(args, in_channels=in_channels, data_fmt='NLC',
            lstm_num=f1lstm_num, lstm_dim=f1lstm_dim, return_states=True)
        in_channels = self.LSTM1D.out_channels

        self.F1Linear = MyReturnTower(args, in_channels=in_channels, data_fmt='NLC',
            linear_dim=[32, 32, 1],
            linear_num=1,
            is_return=True,
            )


    def forward(self, x, bpmat, seqs_len):

        x, seqs_len = self.Embed(x, seqs_len=seqs_len)

        x = self.InLinear(x, seqs_len=seqs_len)

        x = self.Attn(x, seqs_len=seqs_len)

        x2d = self.Seq2Mat(self.Seq2MatPreLinear1(x), self.Seq2MatPreLinear2(x))

        x2d = mi.concat([x2d, bpmat], axis=-1)
        x2d = self.Mat2DLinear(x2d)
        x2d = self.Conv2D(x2d, seqs_len=seqs_len)

        if self.Conv2D.pre_act_norm or self.Conv2D.resnet:
            x2d = self.PostConv2d(x2d)

        x2d = (x2d + mi.transpose(x2d, perm=[0, 2, 1, 3])) / 2.0
        x2d = self.Out2DLinear(x2d, seqs_len=seqs_len)

        if x2d.shape[-1] == 1:
            x2d = mi.squeeze(x2d, -1)

        x, (h, c) = self.LSTM1D(x.detach(), seqs_len=seqs_len)
        # c: [num_layer*num_direction, batch_size, hidden_size]
        x = mi.reshape(mi.transpose(c, perm=[1, 0, 2]), [len(x), -1])
        f1 = F.sigmoid(self.F1Linear(x, seqs_len=seqs_len))

        return [x2d, f1]


    def summary(self, input_size=None):
        if input_size is None:
            input_size = self.Embed.in_shapes
        return mi.summary(self, input_size)


class Seq2Mat_XCodeConv2DNet(nn.Layer):
    """   """
    def __init__(self, args, seq2mat_method='multiply'):
        super(Seq2Mat_XCodeConv2DNet, self).__init__()

        self.data_fmt = args.input_fmt.upper()
        # self.data_ndim = int(args.input_ndim)
        self.in_channels = int(args.input_dim)
        self.seq2mat_method = seq2mat_method.upper()

        in_channels = self.in_channels # keep record of the running channel dim

        self.Embed = MyEmbdxDLayer(args, in_channels=in_channels, out_fmt='NLC')
        in_channels = self.Embed.out_channels

        self.InLinear = MyInitTower(args, in_channels=in_channels, data_fmt='NLC')
        in_channels = self.InLinear.out_channels

        self.Attn = MyAttnxDTower(args, in_channels=in_channels, data_fmt='NLC',
                        attn_dual=True,
                        # attn_method='efficient',
                        attn_axis=2)
        in_channels = self.Attn.out_channels

        self.Seq2Mat = Seq2MatCastxD(method=self.seq2mat_method, in_fmt='NLC', out_fmt='NHWC')
        in_channels = in_channels * self.Seq2Mat.mul_channels

        # in_channels += 4 # add the bpmat [LL4]
        # self.Linear = nn.Sequential(
        #     nn.Linear(in_channels, in_channels - 4),
        #     nn.Sigmoid(),
        #     nn.LayerNorm(in_channels - 4),
        # )
        # in_channels -= 4

        self.Conv2D = MyConv2DTower(args, in_channels=in_channels, data_fmt='NHWC')
        in_channels = self.Conv2D.out_channels

        self.OutLinear = MyReturnTower(args, in_channels=in_channels, data_fmt='NHWC')

    def forward(self, x, seqs_len):

        x, seqs_len = self.Embed(x, seqs_len=seqs_len)

        x = self.InLinear(x, seqs_len=seqs_len)

        x, x_mat = self.Attn(x, seqs_len=seqs_len)

        x = self.Seq2Mat(x, x)

        # one can in principle concat [x, x_mat] (or multiply, add?) and
        # then do a linear transformation

        # x = mi.concat([x, bpmat], axis=-1)
        # x = self.Linear(x)

        x = self.Conv2D(x, seqs_len=seqs_len)

        if self.seq2mat_method in ['ADD', 'MULTIPLY']:
            pass
        else:
            x = (x + mi.transpose(x, perm=[0, 2, 1, 3])) / 2.0

        x = self.OutLinear(x, seqs_len=seqs_len)

        return x # mi.squeeze(x[:,:,1], axis=-1)

    def summary(self, input_size=None):
        if input_size is None:
            input_size = self.Embed.in_shapes
        return mi.summary(self, input_size)


class SeqMat2Mat_AttnLSTMConv2DNet(nn.Layer):
    """   """
    def __init__(self, args):
        super(SeqMat2Mat_AttnLSTMConv2DNet, self).__init__()

        self.data_fmt = args.input_fmt.upper()
        # self.data_ndim = int(args.input_ndim)
        self.in_channels = int(args.input_dim)

        # 1) Sequence pre-layers, starts with [N, L, C]
        seq_in_channels = self.in_channels # keep record of the running channel dim

        self.seq_embed = MyEmbdxDLayer(args, in_channels=seq_in_channels)
        seq_in_channels = self.seq_embed.out_channels

        self.seq_in = MyLinearTower(args, in_channels=seq_in_channels)
        seq_in_channels = self.seq_in.out_channels

        self.seq_self_attn = MyMultiHeadAttention(q_dim=seq_in_channels, k_dim=seq_in_channels,
                    v_dim=seq_in_channels, join_method='add')

        # 2) Mat pre-layers, starts with [N, L, L C]
        mat_in_channels = 8
        self.mat_in = MyLinearTower(args, in_channels=mat_in_channels)
        mat_in_channels = self.mat_in.out_channels

        self.mat_self_attn = MyMultiHeadAttention(q_dim=mat_in_channels, k_dim=mat_in_channels,
                    v_dim=mat_in_channels, join_method='add')
        mat_in_channels = self.mat_self_attn.out_channels

        # 3) Mutual attention between seq and mat
        self.seq_mat_attn = MyMultiHeadAttention(q_dim=seq_in_channels, k_dim=mat_in_channels,
                    v_dim=mat_in_channels, join_method='concat')
        seq_in_channels = self.seq_mat_attn.out_channels

        self.mat_seq_attn = MyMultiHeadAttention(q_dim=mat_in_channels, k_dim=seq_in_channels,
                    v_dim=seq_in_channels, join_method='concat')
        mat_in_channels = self.mat_seq_attn.out_channels

        # 4) Seq post-layers

        self.seq_lstm = MyLSTM1DTower(args, in_channels=seq_in_channels)
        seq_in_channels = self.seq_lstm.out_channels

        # self.seq2mat = Seq2MatTransform(method='concat', in_fmt='NLC', out_fmt='NCHW')
        # seq_in_channels = seq_in_channels * 2 # due to outer concatenation
        # self.conv2d = MyConv2DTower(args, in_channels=seq_in_channels)
        # seq_in_channels = self.conv2d.out_channels

        self.seq_out = MyReturnTower(args, in_channels=seq_in_channels, data_fmt='NLC')

        # 5) Mat post-layers
        self.mat_conv2d = MyConv2DTower(args, in_channels=mat_in_channels)
        mat_in_channels = self.mat_conv2d.out_channels

        self.mat_out = MyLinearTower(misc.Struct(vars(args)).update(
                data_fmt = 'NLLC',
                linear_dim = args.return_dim,
                linear_num = args.return_num,
                linear_resnet = False,
                ),
                in_channels = seq_in_channels,
                is_return = True,
        )

    def forward(self, seq, mat, seqs_len=None):

        seq = self.seq_embed(seq, seqs_len=seqs_len)
        seq = self.seq_in(seq, seqs_len=seqs_len)
        seq = self.seq_self_attn(seq, seqs_len=seqs_len)

        # mat = self.mat_embed(mat, seqs_len=seqs_len)
        mat = self.mat_in(mat, seqs_len=seqs_len)
        mat = self.mat_self_attn(mat, seqs_len=seqs_len)

        seq = self.seq_mat_attn(seq, mat.reshape([mat.shape[0], -1, mat.shape[-1]]),
                    mat.reshape([mat.shape[0], -1, mat.shape[-1]]))
        seq = self.seq_lstm(seq, seqs_len=seqs_len)

        seq = self.seq2mat(seq, seq)
        seq = self.conv2d(seq, seqs_len=seqs_len)
        seq = mi.transpose(seq, perm=[0, 3, 2, 1]) # --> [N, L, L, 2*conv2d_dim[-1]]

        seq = self.out(seq)

        seq = (seq + mi.transpose(seq, perm=[0, 2, 1, 3])) / 2

        return seq # mi.squeeze(x[:,:,1], axis=-1)

    def summary(self, input_size=None):
        if input_size is None:
            input_size = self.Embed.in_shapes
        return mi.summary(self, input_size)


class SeqSeq2Seq_AttnLSTMNet(nn.Layer):
    """   """
    def __init__(self, args):
        super(SeqSeq2Seq_AttnLSTMNet, self).__init__()

        self.data_fmt = args.input_fmt.upper()
        # self.data_ndim = int(args.input_ndim)
        self.in_channels = int(args.input_dim)

        # 1) Sequence pre-layers, starts with [N, L, C]
        in_channels = self.in_channels # keep record of the running channel dim

        self.seq1_embed = MyEmbdxDLayer(args, in_channels=in_channels)
        self.seq2_embed = MyEmbdxDLayer(args, in_channels=in_channels)
        in_channels = self.seq1_embed.out_channels

        self.seq1_in = MyLinearTower(args, in_channels=in_channels)
        in_channels = self.seq1_in.out_channels

        self.seq_self_attn = MyMultiHeadAttention(q_dim=in_channels, k_dim=in_channels,
                    v_dim=in_channels, join_method='add')

        # 2) Mat pre-layers, starts with [N, L, L C]
        mat_in_channels = 8
        self.mat_in = MyLinearTower(args, in_channels=mat_in_channels)
        mat_in_channels = self.mat_in.out_channels

        self.mat_self_attn = MyMultiHeadAttention(q_dim=mat_in_channels, k_dim=mat_in_channels,
                    v_dim=mat_in_channels, join_method='add')
        mat_in_channels = self.mat_self_attn.out_channels

        # 3) Mutual attention between seq and mat
        self.seq_mat_attn = MyMultiHeadAttention(q_dim=in_channels, k_dim=mat_in_channels,
                    v_dim=mat_in_channels, join_method='concat')
        in_channels = self.seq_mat_attn.out_channels

        self.mat_seq_attn = MyMultiHeadAttention(q_dim=mat_in_channels, k_dim=in_channels,
                    v_dim=in_channels, join_method='concat')
        mat_in_channels = self.mat_seq_attn.out_channels

        # 4) Seq post-layers

        self.seq_lstm = MyLSTM1DTower(args, in_channels=in_channels)
        in_channels = self.seq_lstm.out_channels

        # self.seq2mat = Seq2MatTransform(method='concat', in_fmt='NLC', out_fmt='NCHW')
        # seq_in_channels = seq_in_channels * 2 # due to outer concatenation
        # self.conv2d = MyConv2DTower(args, in_channels=seq_in_channels)
        # seq_in_channels = self.conv2d.out_channels

        self.seq_out = MyReturnTower(args, in_channels=in_channels, data_fmt='NLC')

        # 5) Mat post-layers
        self.mat_conv2d = MyConv2DTower(args, in_channels=mat_in_channels)
        mat_in_channels = self.mat_conv2d.out_channels

        self.mat_out = MyLinearTower(misc.Struct(vars(args)).update(
                data_fmt = 'NLLC',
                linear_dim = args.return_dim,
                linear_num = args.return_num,
                linear_resnet = False,
                ),
                in_channels = in_channels,
                is_return = True,
        )

    def forward(self, seq, mat, seqs_len=None):

        seq = self.seq_embed(seq, seqs_len=seqs_len)
        seq = self.seq1_in(seq, seqs_len=seqs_len)
        seq = self.seq_self_attn(seq, seqs_len=seqs_len)

        # mat = self.mat_embed(mat, seqs_len=seqs_len)
        mat = self.mat_in(mat, seqs_len=seqs_len)
        mat = self.mat_self_attn(mat, seqs_len=seqs_len)

        seq = self.seq_mat_attn(seq, mat.reshape([mat.shape[0], -1, mat.shape[-1]]),
                    mat.reshape([mat.shape[0], -1, mat.shape[-1]]))
        seq = self.seq_lstm(seq, seqs_len=seqs_len)

        seq = self.seq2mat(seq, seq)
        seq = self.conv2d(seq, seqs_len=seqs_len)
        seq = mi.transpose(seq, perm=[0, 3, 2, 1]) # --> [N, L, L, 2*conv2d_dim[-1]]

        seq = self.out(seq)

        seq = (seq + mi.transpose(seq, perm=[0, 2, 1, 3])) / 2

        return seq # mi.squeeze(x[:,:,1], axis=-1)

    def summary(self, input_size=None):
        if input_size is None:
            if len(self.data_fmt) == 2:
                input_size = (4, 512)
            else:
                input_size = (4, 512, self.in_channels)
        return mi.summary(self, input_size)


class Seq2Num_LinearNet(nn.Layer):
    """   """
    def __init__(self, args):
        super(Seq2Num_LinearNet, self).__init__()

        self.data_fmt = args.input_fmt.upper()
        # self.data_ndim = int(args.input_ndim)
        self.in_channels = int(args.input_dim)

        self.seq_len = 512
        if args.input_genre in ['esm_mean', 'mean_esm', 'esmmean', 'meanesm']:
            self.seq_len = 1280
        elif args.input_genre in ['domain_feat', 'domainfeat']:
            self.seq_len = 18
        elif args.input_genre in ['esm']:
            pass
        else:
            logger.error(f'Unrecognized data_genre: {args.input_genre}!!!')

        in_channels = self.in_channels # keep record of the running channel dim

        self.Embed = MyEmbdxDLayer(args, in_channels=in_channels)
        in_channels = self.Embed.out_channels

        self.InLinear = MyInitTower(args, in_channels=in_channels)
        in_channels = self.InLinear.out_channels

        self.Linear = MyLinearTower(args, in_channels=in_channels, data_fmt='NHWC')
        in_channels = self.Linear.out_channels

        in_channels = self.seq_len
        self.OutLinear = MyReturnTower(args, in_channels=in_channels, data_fmt='NHWC')

    def forward(self, x, seqs_len=None):

        x, seqs_len = self.Embed(x, seqs_len=seqs_len)
        x = self.InLinear(x, seqs_len=seqs_len)

        x = self.Linear(x)

        x = mi.squeeze(x, -1)

        x = self.OutLinear(x, seqs_len=seqs_len)

        return x # mi.squeeze(x[:,:,1], axis=-1)

    def summary(self, input_size=None):
        if input_size is None:
            input_size = self.Embed.in_shapes
        return mi.summary(self, input_size)


class Seq2Num_LSTMNet(nn.Layer):
    """   """
    def __init__(self, args):
        super(Seq2Num_LSTMNet, self).__init__()

        self.data_fmt = args.input_fmt.upper()
        # self.data_ndim = int(args.input_ndim)
        self.in_channels = int(args.input_dim)

        self.seq_len = 512
        if args.input_genre in ['esm_mean', 'mean_esm', 'esmmean', 'meanesm']:
            self.seq_len = 1280
        elif args.input_genre in ['domain_feat', 'domain_prop']:
            self.seq_len = 18
        elif args.input_genre in ['aa_feat', 'aa_prop']:
            self.seq_len = 512
        elif args.input_genre in ['esm', 'esm_mat', 'esm_file']:
            pass
        else:
            logger.error(f'Unrecognized data_genre: {args.input_genre}!!!')

        in_channels = self.in_channels # keep record of the running channel dim

        self.Embed = MyEmbdxDLayer(args, in_channels=in_channels)
        in_channels = self.Embed.out_channels

        self.InLinear = MyInitTower(args, in_channels=in_channels)
        in_channels = self.InLinear.out_channels

        self.LSTM = MyLSTM1DTower(args, in_channels=in_channels)
        in_channels = self.LSTM.out_channels

        # only take the cell state
        self.OutLinear = MyReturnTower(args, in_channels=in_channels)

    def forward(self, x, seqs_len=None):

        x, seqs_len = self.Embed(x, seqs_len=seqs_len)
        x = self.InLinear(x, seqs_len=seqs_len)

        x, (h, c) = self.LSTM(x, return_states=True)#, seqs_len=seqs_len)

        x = x[:, 0, :]
        # x = mi.squeeze(x[:,-1,:])
        # x = (c[-1] + c[-2]) / 2.0

        x = self.OutLinear(x, seqs_len=seqs_len)

        return x # mi.squeeze(x[:,:,1], axis=-1)

    def summary(self, input_size=None):
        # if input_size is None:
        #     if # self.data_ndim == 0:
        #         input_size = (1, self.seq_len)
        #     else:
        #         input_size = (1, self.seq_len, self.in_channels)
        # return mi.summary(self, input_size)
        if input_size is None:
            input_size = self.Embed.in_shapes
        return mi.summary(self, input_size)


class Seq2Num_AttnNet(nn.Layer):
    """   """
    def __init__(self, args):
        super(Seq2Num_AttnNet, self).__init__()

        self.data_fmt = args.input_fmt.upper()
        # self.data_ndim = int(args.input_ndim)
        self.in_channels = int(args.input_dim)

        self.seq_len = 512
        if args.input_genre in ['esm_mean', 'mean_esm', 'esmmean', 'meanesm']:
            self.seq_len = 1280
        elif args.input_genre in ['domain_feat', 'domainfeat']:
            self.seq_len = 18
        elif args.input_genre in ['esm']:
            pass
        else:
            logger.error(f'Unrecognized data_genre: {args.input_genre}!!!')

        in_channels = self.in_channels # keep record of the running channel dim

        self.Embed = MyEmbdxDLayer(args, in_channels=in_channels)
        in_channels = self.Embed.out_channels

        self.InLinear = MyInitTower(args, in_channels=in_channels)
        in_channels = self.InLinear.out_channels

        self.Attn = MyAttnxDTower(args, in_channels=in_channels)
        in_channels = self.Attn.out_channels

        self.Linear = MyLinearTower(args, in_channels=in_channels)
        in_channels = self.Linear.out_channels

        in_channels = self.seq_len
        self.OutLinear = MyReturnTower(args, in_channels=in_channels, data_fmt='NL')

    def forward(self, x, seqs_len=None):

        x, seqs_len = self.Embed(x, seqs_len=seqs_len)
        x = self.InLinear(x, seqs_len=seqs_len)

        x = self.Attn(x, seqs_len=seqs_len)

        x = self.Linear(x, seqs_len=seqs_len)

        x = mi.squeeze(x, -1)

        x = self.OutLinear(x, seqs_len=seqs_len)

        return x # mi.squeeze(x[:,:,1], axis=-1)

    def summary(self, input_size=None):
        if input_size is None:
            input_size = self.Embed.in_shapes
        return mi.summary(self, input_size)


class Mat2Mat_Conv2DNet(nn.Layer):
    """   """
    def __init__(self, args):
        super(Mat2Mat_Conv2DNet, self).__init__()

        self.data_fmt = args.input_fmt.upper()
        # self.data_ndim = int(args.input_ndim)
        self.in_channels = int(args.input_dim)

        in_channels = self.in_channels # keep record of the running channel dim

        self.Embed = MyEmbdxDLayer(args, in_channels=in_channels, out_fmt='NHWC')
        in_channels = self.Embed.out_channels

        self.InLinear = MyInitTower(args, in_channels=in_channels, data_fmt='NHWC')
        in_channels = self.InLinear.out_channels

        self.Conv2D = MyConv2DTower(args, in_channels=in_channels, data_fmt='NHWC')
        in_channels = self.Conv2D.out_channels

        self.OutLinear = MyReturnTower(args, in_channels=in_channels, data_fmt='NHWC')

    def forward(self, x, seqs_len=None):

        x, seqs_len = self.Embed(x, seqs_len=seqs_len)
        x = self.InLinear(x, seqs_len=seqs_len)

        x = self.Conv2D(x, seqs_len=seqs_len)

        x = (x + mi.transpose(x, perm=[0, 2, 1, 3])) / 2.0

        x = self.OutLinear(x, seqs_len=seqs_len)

        return x # mi.squeeze(x[:,:,1], axis=-1)

    def summary(self, input_size=None):
        if input_size is None:
            input_size = self.Embed.in_shapes
        return mi.summary(self, input_size)
