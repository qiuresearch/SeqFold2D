# %%
import os
import functools
import logging
from io import StringIO
import json
import math
import random
import numpy as np
from scipy.signal import convolve2d
import pandas as pd
import multiprocessing as mpi
from collections import defaultdict, namedtuple, Counter
from pathlib import Path
from tqdm import tqdm

# homebrew
import misc
import molstru_config
import geometry as geom
from gwio import get_file_lines

logger = logging.getLogger(__name__)

reslet_lookup = defaultdict(lambda: set(random.choice('*')))
reslet_lookup.update(
    A=set('AI'), I=set('AI'), G=set('G'), C=set('C'), 
    U=set(['U', 'T']),
    T=set(['T', 'U']),
    R=set(['G', 'A']), # puRine
    Y=set(['C', 'T']), # pYrimidine
    K=set(['G', 'T']),
    M=set(['A', 'C']),
    S=set(['G', 'C']),
    W=set(['A', 'U', 'T']),
    B=set(['G', 'C', 'T', 'U']),
    D=set(['A', 'G', 'U', 'T']),
    H=set(['A', 'C', 'T', 'U']),
    V=set(['A', 'G', 'C']),
    X=set(['A', 'U', 'G', 'C', 'T']),
    N=set(['A', 'U', 'G', 'C', 'T']),
    P=set(['A', 'U', 'G', 'C', 'T']),
)


def seq_fuzzy_match(seq1, seq2):
    assert len(seq1) == len(seq2), 'two sequences must be of the same length!'

    for i, res1 in enumerate(seq1):
        if len(reslet_lookup[res1].intersection(reslet_lookup[seq2[i]])) == 0:
            return False
    return True

# ======= convert structure into scalars/quants  =======
# defaultdict ADD a field if not present, undesired!
res2num_dict = dict() # defaultdict(lambda: '-')
res2num_dict.update({
    '-': 0,
    'A': 1,
    'U': 2,
    'C': 3,
    'G': 4,
    'N': 5,
})

num2res_dict = dict() # defaultdict(lambda: 0)
num2res_dict.update(dict([(_v, _k) for _k, _v in num2res_dict.items()]))

num2dbn_dict = dict() # defaultdict(lambda: '-')
num2dbn_dict.update({
    0: '-',
    1: '.',
    2: '(',
    3: ')',
})
dbn2num_dict = dict() # defaultdict(lambda: 0)
dbn2num_dict.update(dict([(_v, _k) for _k, _v in num2dbn_dict.items()]))

def quant2seq():
    pass


def seq2quant(seq, dbn=None, use_nn=0, use_dbn=False, length=None):
    """ use a single number for each seq/dbn
    0 denotes padding/start/end
    Note: all special residues are assigned as "N=5"s
    """
    seq_len = len(seq)
    if not length:
        length = seq_len
    if not use_nn:
        use_nn = 0
    else:
        use_nn = int(use_nn)

    # convert to capital letters and replace X by N, T by U
    seq = seq.upper().replace('X', 'N').replace('T', 'U')

    # deal with nearest neighbor inclusions
    seq = ('-' * use_nn) + seq + ('-' * use_nn)
    if use_dbn and dbn is not None:
        dbn = ('-' * use_nn) + dbn + ('-' * use_nn)
        assert len(dbn) == len(seq), 'seq and dbn must have the same length!'

    # first get the list of idx for each residue
    seq_emb = []
    nn_indices = list(range(-use_nn, use_nn + 1))

    # this is the shape/len for each dim
    dim_sizes = [len(res2num_dict)] * (2 * use_nn + 1)
    if use_dbn and dbn is not None:
        dim_sizes.extend([len(dbn2num_dict)] * (2 * use_nn + 1))

    for i in range(use_nn, min([length, seq_len]) + use_nn):
        # first each residue is represented by a list of indices for seq/dbn/etc
        res_emb = [res2num_dict.get(seq[i + _i], 5) for _i in nn_indices]

        if use_dbn and dbn is not None:
            res_emb.extend([dbn2num_dict.get(dbn[i + _i], 0) for _i in nn_indices])

        # if use_attr: (not necessary for using attributes)
            # res_embedded.append(res2onehot_attr_dict[seq[i]])
        seq_emb.append(np.array(res_emb, dtype=np.int32))

    seq_emb = np.stack(seq_emb, axis=0)
    dim_sizes = np.array(dim_sizes, dtype=np.int32)
    # the multiplier for each dim, 1 for the last dim
    dim_multiplier = np.concatenate((np.flip(np.cumprod(np.flip(dim_sizes[1:]))),
                                      np.ones((1), dtype=np.int32)), axis=0)

    # quantize the vector for each residue
    seq_emb = np.matmul(seq_emb, dim_multiplier)

    if len(seq_emb) < length:
        seq_emb = np.concatenate((seq_emb, np.zeros((length - len(seq_emb),), dtype=np.int32)))

    return seq_emb


def seq2quant_old(seq, dbn=None, use_nn=0, use_dbn=False, length=None):
    """ use a single number for each seq/dbn """
    if not length:
        length = len(seq)

    if use_dbn and dbn is not None:
        num_dbn_idx = len(dbn2num_dict)
        seq_quant = [(res2num_dict[_s]-1) * num_dbn_idx +
                      dbn2num_dict[dbn[_i]] if res2num_dict[_s] else 0
                      for _i, _s in enumerate(seq[:length])]
    else:
        seq_quant = [res2num_dict[_s] for _s in seq[:length]]

    if len(seq) < length:
        seq_quant.extend([0] * (length - len(seq)))

    return np.array(seq_quant, dtype=np.int32)

# ======= convert structure into vectors/embeddings =======
# The following dict was modified from e2efold code
res2onehot_dict = dict() # defaultdict(lambda: np.array([0, 0, 0, 0, 0]))
res2onehot_dict.update({  # A? U? C? G?
    '-': np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float32), # required for residue_nn feature
    'A': np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32),
    'U': np.array([0.0, 1.0, 0.0, 0.0], dtype=np.float32),
    'T': np.array([0.0, 1.0, 0.0, 0.0], dtype=np.float32),
    'C': np.array([0.0, 0.0, 1.0, 0.0], dtype=np.float32),
    'G': np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32),
    'I': np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32),
    'N': np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float32)/4,
    'X': np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float32)/4,
    'P': np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float32)/4,  # Phosphate only
    'M': np.array([1.0, 0.0, 1.0, 0.0], dtype=np.float32)/2,  # aMide
    'K': np.array([0.0, 1.0, 0.0, 1.0], dtype=np.float32)/2,  # Keto
    'R': np.array([1.0, 0.0, 0.0, 1.0], dtype=np.float32)/2,  # puRine
    'Y': np.array([0.0, 1.0, 1.0, 0.0], dtype=np.float32)/2,  # pYrimidine
    'W': np.array([1.0, 1.0, 0.0, 0.0], dtype=np.float32)/2,  # Weak
    'S': np.array([0.0, 0.0, 1.0, 1.0], dtype=np.float32)/2,  # Strong
    'V': np.array([1.0, 0.0, 1.0, 1.0], dtype=np.float32)/3,  # not U/T
    'D': np.array([1.0, 1.0, 0.0, 1.0], dtype=np.float32)/3,  # not C
    'B': np.array([0.0, 1.0, 1.0, 1.0], dtype=np.float32)/3,  # not A
    'H': np.array([1.0, 1.0, 1.0, 0.0], dtype=np.float32)/3,  # not G
})

# Additional features for five bases: A, U, C, G, -
res2onehot_attr_dict = dict() # defaultdict(lambda: np.array([0] * 8))
res2onehot_attr_dict.update({
    #    AU GC  GU M  K  S  R  Y more to add?
    '-': np.array([0, 0, 0, 0, 0, 0, 0, 0], dtype=np.float32),
    'A': np.array([1, 0, 0, 1, 0, 0, 1, 0], dtype=np.float32),
    'U': np.array([1, 0, 1, 0, 1, 1, 0, 1], dtype=np.float32),
    'C': np.array([0, 1, 0, 1, 0, 0, 0, 1], dtype=np.float32),
    'G': np.array([0, 1, 1, 0, 1, 1, 1, 0], dtype=np.float32),
    'R': np.array([1, 1, 1, 1, 1, 1, 2, 0], dtype=np.float32)/2, # average of A, G
    'Y': np.array([1, 1, 1, 0, 1, 1, 0, 2], dtype=np.float32)/2, # average of U, C
})
# seq_trait_mat = np.array([
#     # AU? GC? GU? M? K? S?   more to add?
#     [1,   0,  0, 1, 0, 0],
#     [1,   0,  1, 0, 1, 1],
#     [0,   1,  0, 1, 0, 0],
#     [0,   1,  1, 0, 1, 1],
#     [0,   0,  0, 0, 0, 0],
# ], dtype=np.int32)

dbn2onehot_dict = dict() # defaultdict(lambda: np.array([0, 0, 0]))
dbn2onehot_dict.update({
    '-': np.array([0, 0, 0], dtype=np.int32),  # gap or padding
    '.': np.array([1, 0, 0], dtype=np.int32),
    '(': np.array([0, 1, 0], dtype=np.int32),
    ')': np.array([0, 0, 1], dtype=np.int32),
})

def onehot2seq(seq_vec, vocab=np.array(list('AUCG'))):
    """ vocab can be a string or np array of letters """
    if not isinstance(vocab, np.ndarray):
        vocab = np.array(list(vocab))
    return ''.join(vocab[seq_vec.argmax(axis=1)])


def seq2onehot(seq, dbn=None, bpp=None, length=None,
               use_nn=0, use_attr=False, use_dbn=False, use_bpp=False,
               res2vec_dict=res2onehot_dict,
               dbn2vec_dict=dbn2onehot_dict,
               attr2vec_dict=res2onehot_attr_dict):
    """ embed each seq/dbn with a vector """
    seq_len = len(seq)
    if not length:
        length = seq_len
    if not use_nn:
        use_nn = 0
    else:
        use_nn = int(use_nn)

    # convert to capital letters
    seq = seq.upper()

    # pad for nearest neighbor inclusion
    seq = ('-' * use_nn) + seq + ('-' * use_nn)
    if use_dbn is not None and dbn is not None:
        dbn = ('-' * use_nn) + dbn + ('-' * use_nn)
        assert len(dbn) == len(seq), 'seq and dbn must have the same length!'

    seq_emb = []
    nn_indices = list(range(-use_nn, use_nn + 1))

    for i in range(use_nn, min([length, seq_len]) + use_nn):
        res_emb = [res2vec_dict[seq[i + _i]] for _i in nn_indices]

        if use_dbn and dbn is not None:
            res_emb.extend([dbn2vec_dict[dbn[i + _i]] for _i in nn_indices])

        if use_attr: # no nn for trait yet
            res_emb.append(attr2vec_dict[seq[i]])

        if use_bpp: # the base pair probability from linear_partition
            res_emb.append(bpp[i - use_nn])

        if isinstance(res_emb[0], np.ndarray):
            seq_emb.append(np.concatenate(res_emb))
        else:
            seq_emb.append(np.array(res_emb, dtype=np.int32))

    # for i in range(len(seq_emb), length):
        # seq_emb.append(np.zeros((seq_emb[0].shape[0]), dtype=np.int16))
    if length > len(seq_emb):
        seq_emb.extend(np.zeros((length - len(seq_emb), seq_emb[0].shape[0]), dtype=np.int32))

    seq_emb = np.stack(seq_emb, axis=0)

    # if use_attr and use_dbn and dbn is not None:
    #     seq_embedded = [np.concatenate((
    #         res2onehot_dict[_s],
    #         res2onehot_attr_dict[_s],
    #         dbn2onehot_dict[dbn[_i]],
    #     )) for _i, _s in enumerate(seq)]
    # elif use_attr and not use_dbn:
    #     seq_embedded = [np.concatenate((
    #         res2onehot_dict[_s],
    #         res2onehot_attr_dict[_s],
    #     )) for _s in seq]
    # elif use_dbn and dbn is not None:
    #     seq_embedded = [np.concatenate((
    #         res2onehot_dict[_s],
    #         dbn2onehot_dict[dbn[_i]],
    #     )) for _i, _s in enumerate(seq)]
    # else:
    #     seq_embedded = [res2onehot_dict[_s] for _s in seq]

    # if use_dbn and dbn is not None:
    #     for _i, _s in enumerate(dbn):
    #         seq_embedded[_i].extend(dbn2onehot_dict[_s])
    # rna_embedded = [np.concatenate((
    #                     res2onehot_dict[_s],
    #                     res2onehot_attr_dict[_s] if use_attr else [],
    #                     dbn2onehot_dict[dbn[_i]] if use_dbn else [],
    #                     )) for _i, _s in enumerate(seq[:length])]
    return seq_emb


def seq2RNA(
        seq_in,
        upper=True,
        vocab_set=set('AUCG'),
        pseudo_random=False,
        trans_fn_dict=molstru_config.get_res2RNA_trans_dict(return_fn=True),
        trans_dict=None, # str.maketrans({'T':'U', 'I':'A'}),
        verify=False):
    """ convert a sequence to RNA sequence
        pseudo_random=True will translate all N to the same random
            letter chosen from AUCG per call.
            It may be faster for long seqs.
    """
    if upper:
        seq_in = seq_in.upper()

    if pseudo_random:
        if trans_dict is None:
            trans_dict = str.maketrans(molstru_config.get_res2RNA_trans_dict())
        seq_out = seq_in.translate(trans_dict)
    else:
        seq_out = ''.join([trans_fn_dict[_s]() for _s in seq_in])

    if verify and len(set(seq_out) - vocab_set):
        logger.warning(f'Check sequence for untranslated nts: {set(seq_out)}')
        print(seq_in)

    return seq_out


def seq2DNA(
        seq_in,
        upper=True,
        vocab_set=set('ATCG'),
        pseudo_random=False,
        trans_fn_dict=molstru_config.get_res2DNA_trans_dict(return_fn=True),
        trans_dict=None, # str.maketrans({'U':'T', 'I':'A'}),
        verify=False):
    """ convert a sequence to DNA sequence
        Can make this a wrapper to call seq2RNA()
    """
    if upper:
        seq_in = seq_in.upper()

    if pseudo_random:
        if trans_dict is None:
            trans_dict = str.maketrans(molstru_config.get_res2DNA_trans_dict())
        seq_out = seq_in.translate(trans_dict)
    else:
        seq_out = ''.join([trans_fn_dict[_s]() for _s in seq_in])

    if verify and len(set(seq_out) - vocab_set):
        logger.warning(f'Check sequence for untranslated nts: {set(seq_out)}')
        print(seq_in)

    return seq_out


def seq2cDNA(seq, reverse=True, verify=False,
        vocab_set=set('ACGT'),
        trans_dict=str.maketrans({'A':'T', 'C':'G', 'G':'C', 'T':'A', 'U':'A'}),
        ):
    """ convert a sequence to its cDNA sequence """
    if reverse:
        seq_cdna = seq.translate(trans_dict)[::-1]
    else:
        seq_cdna = seq.translate(trans_dict)

    if verify and len(set(seq_cdna) - vocab_set):
        logger.warning(f'Check sequence for untranslated nts: {set(seq_cdna)}')
        print(seq)

    return seq_cdna


def seq2kmer(seq, k=5, padding=None, delimiter=' '):
    """
    Convert original sequence to kmers (adopted from DNABERT)

    Arguments:
    seq -- str, original sequence.
    k -- int, kmer of length k specified.

    Returns:
    kmer_seq -- a sequence of kmers separated by delimiter

    """
    if padding:
        if k % 2 != 1:
            logger.error('k must be odd with padding!!!')
            k += 1
        nn = k // 2

        seq = padding * nn + seq + padding * nn
        kmers_list = [seq[i - nn:i + nn + 1] for i in range(nn, len(seq) - nn)]
    else:
        kmers_list = [seq[i:i + k] for i in range(len(seq) - k + 1)]

    return delimiter.join(kmers_list)


def kmer2seq(kmer, padding=None):
    """
    Convert kmers to original sequence (adopted from DNABERT)

    Arguments:
    kmers -- str, kmers separated by space.

    Returns:
    seq -- str, original sequence.

    """
    if type(kmer) in (list, tuple):
        k = len(kmer[0])
        if padding:
            if k % 2 == 0:
                logger.warning(f'k={k} should be an odd number with padding={padding}!!!')
            seq_len = len(kmer)
            i = k // 2
            seq = [_s[i] for _s in kmer]
        else:
            seq_len = len(kmer) + k - 1
            seq = [_s[0] for _s in kmer[:-1]]
            seq.append(kmer[-1])
    elif isinstance(kmer, str):
        # perhaps better to split first
        try:
            k = kmer.index(' ')
        except:
            k = len(kmer)
        if padding:
            if k % 2 == 0:
                logger.warning(f'k={k} should be an odd number with padding={padding}!!!')
            seq_len = (len(kmer) + 1) // (k + 1)
            seq = [kmer[_i] for _i in range(k // 2, len(kmer), k + 1)]
        else:
            seq_len = len(kmer) // (k + 1) + k
            # exclude the last kmer
            seq = [kmer[_i] for _i in range(0, len(kmer) - k, k + 1)]
            seq.append(kmer[-k:])
    else:
        logger.error(f'Unknown data type for kmer: {type(kmer)} !!!')

    seq = ''.join(seq)
    assert len(seq) == seq_len, f'sequence length mismatch: {len(seq)}!={seq_len}'

    return seq


def seq2chunks_contig(seq, min_len=30, max_len=510, drop_last=False, bias=0.0):
    """ the last chunk may be shorter than min_len if max_len < 2 * min_len """

    seq_len = len(seq)
    seq_chunks = []
    chunk_sizes = np.random.randint(min_len, max_len + 1, seq_len // min_len + 1)
    # check whether all elements are used
    # chunk_total = chunk_sizes.cumsum()
    # ilast = np.nonzero(chunk_total >= len(seq))[0][0]

    i = 0
    istart = 0
    max_start = seq_len - min_len
    while istart <= max_start:
        seq_chunks.append(seq[istart:istart + chunk_sizes[i]])
        istart += chunk_sizes[i]
        i += 1

    # deal with the leftovers at the end
    num_left = seq_len - istart
    if num_left <= 0:
        assert seq_len == sum([len(_s) for _s in seq_chunks]), 'total chunk length differs from input!!!'
        return seq_chunks

    if num_left >= min_len: # the simplest case: num_left within [min_len, max_len)
        seq_chunks.append(seq[istart:])
    elif num_left + len(seq_chunks[-1]) <= max_len: # add to the last one
        seq_chunks[-1] += seq[istart:]
    else:
        seq_extra = seq_chunks.pop() + seq[istart:]
        seq_chunks.extend([seq_extra[:min_len], seq_extra[min_len:]])
        if len(seq_chunks[-1]) < min_len:
            if drop_last:
                seq_chunks.pop()
            else:
                logger.warning(f'the last chunk (len: {len(seq_chunks[-1])} < min_len: {min_len}!!!')

    assert drop_last or seq_len == sum([len(_s) for _s in seq_chunks]), \
        'total chunk length differs from input!!!'
    return seq_chunks


def seq2chunks_poll(seq, min_len=30, max_len=510, coverage=2.0):
    """ the last chunk may be shorter than min_len if max_len < 2 * min_len """

    seq_len = len(seq)
    num_chunks = int(coverage * seq_len // min_len + 1)
    chunk_sizes = np.random.randint(min_len, max_len + 1, num_chunks)
    chunk_starts = np.random.randint(0, seq_len - min_len + 1, num_chunks )
    # cut off chunk ends by seq_len
    chunk_ends = np.minimum(chunk_starts + chunk_sizes, seq_len)

    # cut off the overestimate in num_chunks
    iend = np.argmin(np.abs((chunk_ends - chunk_starts).cumsum() - coverage * seq_len))

    seq_chunks = []
    for i in range(0, iend + 1):
        seq_chunks.append(seq[chunk_starts[i]:chunk_ends[i]])

    # print(f'seq_len: {seq_len} vs chunks_len: {sum([len(_s) for _s in seq_chunks])}')

    return seq_chunks


def seq2chunks(seq_in, min_len=30, max_len=510,  drop_last=False,
        method='contig', coverage=2.0, vocab_set=set('ACGT')):
    """ basically a wrapper for seq2chunks_poll and seq2chunks_contig  """

    assert isinstance(seq_in, str), 'seq must be a string'
    seq_list = [seq_in]

    # divide the seq at the positions of unknown residues
    if vocab_set is not None and len(vocab_set):
        unk_nts_set = set(seq_in) - vocab_set
        if len(unk_nts_set):
            logger.info(f'Found unknown nucleotides: {unk_nts_set} which will be skipped!!!')
            unk_trans_dict = str.maketrans(dict.fromkeys(unk_nts_set, ' '))
            seq_list = seq_in.translate(unk_trans_dict).split(' ')

    seq_chunks = []
    # convert to chunks and then kmerize
    for seq in seq_list:
        if len(seq) < min_len:
            logger.info(f'Skipping sequences shorter than min_len={min_len}: {seq} ...')
        elif len(seq) <= max_len:
            seq_chunks.append(seq)
        else: # need to chunk the seq
            method = method.lower()
            if method in ['contig', 'non-overlap', 'nonoverlap']:
                seq_chunks.extend(seq2chunks_contig(seq, min_len=min_len, max_len=max_len, drop_last=drop_last))
            elif method in ['poll', 'random', 'sample', 'sampling']:
                seq_chunks.extend(seq2chunks_poll(seq, min_len=min_len, max_len=max_len, coverage=coverage))
            else:
                logger.error(f'Unsupported chunk method: {method}')

    # remove empty strings
    # seq_chunks = [_s for _s in seq_chunks if len(_s)] # filter(None, bert_seqs)
    return seq_chunks


def seq2bert(seq, k=3, padding=None, min_len=12, max_len=510,  drop_last=False,
            chunk_method='contig', coverage=1.0, nts_set=set('ACGT')):
    """ basically a wrapper for seq2chunks and seq2kmer """
    # first check for unrecognized sequences
    unk_nts_set = set(seq) - nts_set
    if len(unk_nts_set):
        logger.info(f'Found unknown nucleotides: {unk_nts_set} which will be skipped!!!')
        unk_trans_dict = str.maketrans(dict.fromkeys(unk_nts_set, ' '))
        seq_list = seq.translate(unk_trans_dict).split(' ')
    else:
        seq_list = [seq]

    # convert to chunks and then kmerize
    bert_seqs = []
    for seq in seq_list:
        if len(seq) < k and not padding:
            logger.warning(f'Skipping sequences shorter than k={k}: {seq} ...')
        elif len(seq) <= max_len:
            bert_seqs.append(' '.join(seq2kmer(seq, k=k, padding=padding)))
        else: # need to chunk the seq
            chunk_method = chunk_method.lower()
            if chunk_method in ['contig', 'non-overlap', 'nonoverlap']:
                seq_chunks = seq2chunks_contig(seq, min_len=min_len, max_len=max_len, drop_last=drop_last)
            elif chunk_method in ['poll', 'random', 'sample', 'sampling']:
                seq_chunks = seq2chunks_poll(seq, min_len=min_len, max_len=max_len, coverage=coverage)
            else:
                logger.error(f'Unsupported chunk method: {chunk_method}')

            for seq in seq_chunks:
                bert_seqs.append(' '.join(seq2kmer(seq, k=k, padding=padding)))

    # remove empty strings
    bert_seqs = [_s for _s in bert_seqs if len(_s)] # filter(None, bert_seqs)
    return bert_seqs


def mutate_residues(seq_npa, rate=0.2, replace=True,
        alphabet=np.array(list('AUGC'), dtype='U1'),
        rng=np.random.default_rng(),
        return_list=False):
    """ np.ndarray is returned!!! """
    assert 0. <= rate <= 1.0, 'rate must be between 0 and 1!!!'
    seq_len = len(seq_npa)
    if type(seq_npa) is not np.ndarray:
        seq_npa = np.array(list(seq_npa), dtype='U1')

    mut_idx = np.random.choice(np.arange(seq_len), size=int(seq_len * rate), replace=False)
    if len(mut_idx) == 0:
        return seq_npa

    # seq_ori = seq_npa[mut_idx]
    if replace:
        seq_new = alphabet[rng.integers(0, len(alphabet), size=len(mut_idx))]
    else:
        logger.warning(f'Not yet implemented, it is the same as replace=True!')
        seq_new = alphabet[rng.integers(0, len(alphabet), size=len(mut_idx))]

    seq_npa[mut_idx] = seq_new

    if return_list:
        return ''.join(seq_npa)
    else:
        return seq_npa


def mutate_basepairs(seq_npa, ct, rate=0.2, replace=True,
        alphabet=np.array(['AU', 'UA', 'GC', 'CG', 'GU', 'UG'], dtype='U2'),
        rng=np.random.default_rng(), return_list=False):
    """ np.ndarray is returned!!! """
    assert 0. <= rate <= 1.0, 'rate must be between 0 and 1!!!'

    ct_len = len(ct)
    mut_idx = np.random.choice(np.arange(ct_len), size=int(ct_len * rate), replace=False)
    if len(mut_idx) == 0:
        return seq_npa

    if replace:
        bps_new = alphabet[rng.integers(0, len(alphabet), size=len(mut_idx))]
    else:
        logger.warning(f'Not yet implemented, it is the same as replace=True!')
        bps_new = alphabet[rng.integers(0, len(alphabet), size=len(mut_idx))]

    if type(seq_npa) is not np.ndarray:
        seq_npa = np.array(list(seq_npa), dtype='U1')
    seq_npa[ct[mut_idx,0] - 1] = bps_new.view('<U1')[::2]  # add .reshape(bps_new.shape + (-1,))
    seq_npa[ct[mut_idx,1] - 1] = bps_new.view('<U1')[1::2]

    if return_list:
        return ''.join(seq_npa)
    else:
        return seq_npa


def mutate_stems_loops(seq_npa, ct, replace=True,
        rate=0.2, loop_rate=None, stem_rate=None, return_list=True):
    """ basically a wrapper for mutate_basepairs and mutate_residues """

    seq_len = len(seq_npa)
    if type(seq_npa) is not np.ndarray:
        seq_npa = np.array(list(seq_npa), dtype='U1')

    if ct is not None and len(ct):
        loop_idx = np.delete(np.arange(seq_len), ct.flatten() - 1)
        stem_rate = rate if stem_rate is None else stem_rate
        if stem_rate > 0:
            seq_npa = mutate_basepairs(seq_npa, ct, replace=replace, rate=stem_rate)
    else:
        loop_idx = np.arange(seq_len)

    if len(loop_idx):
        loop_rate = rate if loop_rate is None else loop_rate
        if loop_rate > 0:
            seq_npa[loop_idx] = mutate_residues(seq_npa[loop_idx], replace=replace, rate=loop_rate)

    if return_list:
        return ''.join(seq_npa)
    else:
        return seq_npa


def DEBUG_encoding():
    seq_debug = 'AUCG'*3 + '-'*3
    # seq_debug = np.random.randint(0, 5, size=50)
    dbn_debug = '.()'*4 + '-'*3
    print('Quant and Embed encoding examples:\n')
    print(f'SEQ: {seq_debug}')
    print(f'DBN: {dbn_debug}')
    print('\nQuant/scalar encoding')
    print(seq2quant(seq_debug))
    print('\nQuant/scalar encoding fixed length=23')
    print(seq2quant(seq_debug, length=23))
    print('\nQuant/scalar encoding with dbn')
    print(seq2quant(seq_debug, dbn_debug, use_dbn=True, length=23))

    print('\nEmbedding sequence only')
    print(seq2onehot(seq_debug, use_attr=False))
    print('\nEmbedding sequence with dbn')
    print(seq2onehot(seq_debug, dbn=dbn_debug, use_dbn=True, use_attr=False))
    print('\nEmbedding sequence with dbn and trait')
    print(seq2onehot(seq_debug, dbn=dbn_debug, use_dbn=True, length=23, use_attr=True))


def pair_energy(na_pair):
    """ adopted from e2efold code """
    set_pair = set(na_pair)
    if set_pair == {'A', 'U'}:
        E = 2
    elif set_pair == {'G', 'C'}:
        E = 3
    elif set_pair == {'G', 'U'}:
        E = 0.8
    else:
        E = 0
    return E


def ct_clip_delta_ij(ct, min_delta_ij=3):
    """ remove ct pairs with delta_ij < min_delta_ij, i and j are not sorted """
    if ct.ndim == 2 and ct.shape[0]:
        return np.delete(ct, np.nonzero(abs(np.diff(ct, axis=-1)) < min_delta_ij)[0], axis=0)
    else:
        return ct


def bpmat_clip_delta_ij(bpmat, min_delta_ij=3):
    """ remove bps with delta_ij < min_delta_ij in the bpmat/ctmat """
    x_len = bpmat.shape[0]
    y_len = bpmat.shape[1]

    # a matrix of [x_len, y_len] with each element storing the row idx
    row_idx_mat = np.broadcast_to(
            np.expand_dims(np.linspace(1, x_len, x_len, dtype=int), -1),
            (x_len, y_len))

    if x_len == y_len:
        bpmat *= (np.abs(row_idx_mat - row_idx_mat.T) >= min_delta_ij).astype(int)
    else:
        col_idx_mat = np.broadcast_to(
            np.expand_dims(np.linspace(1, y_len, y_len, dtype=int), 0),
            (x_len, y_len))

        bpmat *= (np.abs(row_idx_mat - col_idx_mat) >= min_delta_ij).astype(int)

    return bpmat


def bpmat_clip_stem_len(bpmat, min_stem_len=2, gap_len=0):
    """ remove stems with length < min_stem_len in the bpmat/ctmat

    The recipe is to first pad bpmat along -i, +i, -j, +j,
    then count the continuous ones along the diagonal direction.
    Counting is done along both upper-right and lower-left directions,
    and the two counts are added

    """
    # pair_mat = np.random.random((20,20)).round()

    x_len = bpmat.shape[0]
    y_len = bpmat.shape[1]

    # pad zeros to both ends of both dimensions
    big_mat = np.pad(bpmat, min_stem_len - 1)

    # count continuous ones towards the upper right
    gate_mat = np.ones_like(bpmat)
    up_right = np.zeros_like(bpmat)
    for i in range(1, min_stem_len):
        x_start = min_stem_len - 1 - i
        y_start = min_stem_len - 1 + i
        up_right += gate_mat * big_mat[x_start : x_start + x_len, y_start: y_start + y_len]
        gate_mat *= big_mat[x_start : x_start + x_len, y_start: y_start + y_len]

    # count continuous ones towards the lower left
    gate_mat[:] = 1
    low_left = np.zeros_like(bpmat)
    for i in range(1, min_stem_len):
        x_start = min_stem_len - 1 + i
        y_start = min_stem_len - 1 - i
        low_left += gate_mat * big_mat[x_start : x_start + x_len, y_start: y_start + y_len]
        gate_mat *= big_mat[x_start : x_start + x_len, y_start: y_start + y_len]

    # stemlen_mat[i,j] gives the length of the stem involving the [i,j] pair (0 if unpaired)
    # (note that the length is capped at 2 * min_stem_len + 1 to save time)
    stemlen_mat = bpmat * (bpmat + low_left + up_right)

    bpmat[stemlen_mat < min_stem_len] = 0

    # need to think about how to allow gaps between stems
    # Only need to deal with
    if gap_len > 0:
        kernel = np.full((2 * gap_len + 1, 2 * gap_len + 1), 1, dtype=int)
        neighbor_counts = convolve2d(bpmat, kernel, mode='same')
        bpmat[np.logical_and(neighbor_counts > 0, stemlen_mat > 0)] = 1

    return bpmat


def bpmat_clip_multi_bp(bpmat, weights=None,  ):
    pass


def seq2bpmat_turner_neighbors(
        seq,
        bp_energies=misc.dict_add_reversed_keys({'AU': 2.0, 'CG': 3.0, 'GU': 0.8}),
        nn_energies=molstru_config.Turner_nnEnergy_Params,
        nn=1,
        nn_left=1,
        nn_right=1,
        min_delta_ij=3, # better to set min_delta_ij smaller than physically allowed
        dtype='float32'):
    """ note that nn stacking energies are double-counted because both left and right
        neighbors of ij pair are added
    """
    seq_len = len(seq)
    bpmat = np.zeros((seq_len, seq_len), dtype=dtype)

    for i in range(seq_len - min_delta_ij):
        for j in range(i + min_delta_ij, seq_len):

            # no need to look up Turner Energies if this is not a pair
            if bp_energies.get(f'{seq[i]}{seq[j]}', 0.0) == 0.0:
                continue

            # stacking energy with its left bp
            for delta in range(1, min([nn_left, i, seq_len - j - 1]) + 1):
                # 1st key: 5->3, 2nd key: 3->5
                bpmat[i, j] += nn_energies[f'{seq[i - delta]}{seq[i]}'][f'{seq[j + delta]}{seq[j]}'][0]

            # stacking energy with its right bp
            for delta in range(1, min([nn_right, (j - i - min_delta_ij + 1) // 2]) + 1):
                bpmat[i, j] += nn_energies[f'{seq[i]}{seq[i + delta]}'][f'{seq[j]}{seq[j - delta]}'][0]

    bpmat *= -1.6 # convert to positive and kT/mol
    bpmat += bpmat.T

    return bpmat


def seq2bpmat_gauss_neighbors(
        seq,
        bp_energies=misc.dict_add_reversed_keys({'AU': 2.0, 'CG': 3.0, 'GU': 0.8}),
        decay_len=2,    # decay lenght of the nn "stacking" effect
        nn=12,          # the number of nearest neighbors
        nn_left=None,
        nn_right=None,
        min_delta_ij=3, # better to set min_delta_ij smaller than physically allowed
        min_stem_len=1, #
        dtype='float32'):
    """  """
    # add the reversed pair energies
    reversed_pairs = dict()
    for key, val in bp_energies.items():
        reversed_pairs[key[::-1]] = val
    bp_energies.update(reversed_pairs)

    seq_len = len(seq)
    if nn_left is None: nn_left = nn
    if nn_right is None: nn_right = nn

    bpmat = np.zeros((seq_len, seq_len), dtype=dtype)
    gauss_coeff = -0.5 / decay_len / decay_len

    for i in range(seq_len - min_delta_ij):
        for j in range(i + min_delta_ij, seq_len):
            e_ij = bp_energies.get(f'{seq[i]}{seq[j]}', 0.0)

            if e_ij == 0.0:
                continue

            stem_len = 1

            e_left = 0.0 # bp energies from its left bps if exists
            for delta in range(1, min([nn_left, i, seq_len - j - 1]) + 1):
                e_nn = bp_energies.get(f'{seq[i - delta]}{seq[j + delta]}', 0.0)
                if e_nn == 0.0:
                    break
                else:
                    stem_len += 1
                    e_left += e_nn * math.exp(gauss_coeff * delta * delta)

            e_right = 0.0 # bp energies from its right bps if exists
            for delta in range(1, min([nn_right, (j - i - min_delta_ij + 1) // 2]) + 1):
                e_nn = bp_energies.get(f'{seq[i + delta]}{seq[j - delta]}', 0.0)
                if e_nn == 0.0:
                    break
                else:
                    stem_len += 1
                    e_right += e_nn * math.exp(gauss_coeff * delta * delta)

            # add all energies if over the min_stem_len
            if stem_len >= min_stem_len:
                bpmat[i, j] = e_ij + e_left + e_right

    bpmat += bpmat.T

    return bpmat


def seq2bpmat_just_pairs(
        seq,
        min_delta_ij=None,
        min_stem_len=None,
        cn_residues=[['A', 'U', 2.0], ['C', 'G', 3.0], ['G', 'U', 0.8]],
        nc_residues=[['A', 'A', 0.1], ['A', 'C', 0.1], ['A', 'G', 0.1], ['A', 'U', 2.0],
                     ['C', 'C', 0.1], ['C', 'G', 3.0], ['C', 'U', 0.1],
                     ['G', 'G', 0.1], ['G', 'U', 0.8], ['U', 'U', 0.1]],
        canonical=True,
        return_energy=True,
        dtype=None,
        ):
    """ get the constraint matrix for pairing
        the default mode is to output the pairing energy
    """
    if dtype is None:
        dtype = 'float32' if return_energy else 'int16'

    seq_len = len(seq)
    seq_vec = seq2onehot(seq)
    seq_kron_prod = np.kron(seq_vec, seq_vec).reshape((seq_len, seq_len, -1)).astype(dtype)

    num_channels = seq_vec.shape[-1]
    bpmat = np.zeros((seq_len, seq_len), dtype=dtype)

    for bp_info in cn_residues if canonical else nc_residues:
        # find the idx along the channel dimension of seq_vec
        i = np.nonzero(res2onehot_dict[bp_info[0]])[0][0]
        j = np.nonzero(res2onehot_dict[bp_info[1]])[0][0]

        ikron = i + j * num_channels
        if return_energy:
            bpmat += seq_kron_prod[:,:,ikron] * bp_info[2]
        else:
            bpmat += seq_kron_prod[:,:,ikron]

        #     if binary:
        #         pair_mat += np.matmul(seq_vec[:, i:i+1], seq_vec[:, j].T)
        #     else:
        #         pair_mat += np.matmul(seq_vec[:, i:i+1], seq_vec[:, j].T) * pair[2]

    # include the transpose
    bpmat += bpmat.T

    # apply min_delta_ij
    if min_delta_ij is not None and min_delta_ij > 0:
        row_mat = np.broadcast_to(
            np.expand_dims(np.linspace(1, seq_len, seq_len, dtype=dtype), -1),
            (seq_len, seq_len))

        bpmat *= (np.abs(row_mat - row_mat.T) >= min_delta_ij).astype(dtype)

    if min_stem_len is not None and min_stem_len > 1:
        bpmat = bpmat_clip_stem_len(bpmat, min_stem_len)

    return bpmat


moltype_mapping = {
    '16s': '16S-rRNA', # archiveII
    "16S_rRNA": "16S-rRNA", # Stralign
    '23s': '23S-rRNA', # archiveII
    "23S_rRNA": "23S-rRNA", # Stralign
    '5s': '5S-rRNA',    # archiveII
    "5S_rRNA": "5S-rRNA", # Stralign
    'antisense RNA': 'antisense-RNA', # rnacentral
    'autocatalytically spliced intron': 'ss-intron', # rnacentral
    'CRW': 'CRW', # bpRNA: the Comparative RNA Web site
    'grp1': 'gpI-intron',   # archiveII
    "group_I_intron": 'gpI-intron', # Stralign
    'grp2': 'gpII-intron',  # archiveII
    "group_II_intron": 'gpII-intron', # Stralign
    'guide RNA': 'gRNA', # rnacentral
    'hammerhead ribozyme': 'hh-ribozyme', # rnacentral
    'lncRNA': 'lncRNA', # rnacentral
    'miRNA': 'miRNA', # rnacentral
    'misc RNA': 'misc_RNA', # rnacentral
    'ncRNA': 'ncRNA', # rnacentral
    'other': 'other', # rnacentral
    'PDB': 'PDB',    # bpRNA
    'piRNA': 'piRNA', # rnacentral
    'pre miRNA': 'pre_miRNA', # rnacentral
    'precursor RNA': 'pre_mRNA', # rnacentral
    'rRNA': 'rRNA', # rnacentral
    'ribozyme': 'ribozyme', # rnacentral
    'RFAM': 'RFAM',  # bpRNA
    'RNase MRP RNA': 'RNaseMRP', # rnacentral (for mitocondria RNA processing)
    'RNase P RNA': 'RNaseP', # rnacentral
    'RNaseP': 'RNaseP', # archiveII
    'RNP': 'RNaseP',    # bpRNA
    'scaRNA': 'scaRNA', # rnacentral ()
    'scRNA': 'scRNA', # rnacentral (small conditional RNA)
    'snoRNA': 'snoRNA', # rnacentral
    'snRNA': 'snRNA', # rnacentral
    'siRNA': 'siRNA', # rnacentral
    'SPR': 'tRNA', # bpRNA: Sprinzl tRNA Database
    'srp': 'SRP',   # archiveII
    'SRP': 'SRP',   # bpRNA
    'SRP RNA': 'SRP', # rnacentral
    'sRNA': 'sRNA', # rnacentral (bacterial small RNA)
    'tmRNA': 'tmRNA', # archiveII
    'tRNA': 'tRNA', # archiveII
    'telomerase': 'TERC', # archiveII
    'telomerase RNA': 'TERC', # rnacentral
    'UNK': 'UNK',
    'vault RNA': 'vRNA', # rnacentral (associated with vault protein)
    'Y RNA': 'YRNA', # rnacentral
    }
def parse_moltype(fname, database=None):
    """ fname is either file name or moltype; db should be in lower case """
    if database is None: return ''

    if database == 'archiveii':
        moltype = Path(fname).stem.split('_')[0]
    elif database == 'stralign':
        moltype = Path(fname).parent
        if moltype.stem.endswith('_database'):
            moltype = moltype.stem[:-9]
        elif moltype.parent.stem.endswith('_database'):
            moltype = moltype.parent.stem[:-9]
    elif database == 'bprna':
        moltype = Path(fname).stem.split('_')[1]
    elif database == 'rnastrand':
        moltype = Path(fname).stem.split('_')[0]
    elif database == 'rnacentral':
        moltype = fname.replace('_', ' ')
    else:
        moltype = 'UNK'

    return moltype_mapping.get(moltype, 'UNK')


def parse_seqres_id(seqres_id, fmt='pdblib'):
    """ parse the id field of seqres and return a dict """
    tokens = seqres_id.split()
    return {
        'id': f'{tokens[0][:4]}_{tokens[0][5:]}',
        'moltype': tokens[1][4:],
        'len': int(tokens[2][7:]),
        'title': ' '.join(tokens[3:]),
    }


def parse_json_lines(json_info, fmt='rnacentral', return_list=True):
    """ each line of bpseq contains 3 columns:
        returns the same as ct file: len, seq, ct, resnum
    """
    if (isinstance(json_info, str) and os.path.exists(json_info)) or isinstance(json_info, Path):
        fname = str(json_info)
        with open(fname, 'r') as iofile:
            seqs_list = json.load(iofile)
    else:
        fname = ''
        seqs_list = json.loads(json_info)

    keys_mapping = {'rnacentral_id': 'id',
                    'description': 'desc',
                    'sequence': 'seq',
                    'md5': 'md5',
                    'rna_type': 'moltype',
                    'taxon_id': 'taxon',
                    'xrefs': 'refs',
                    }

    if isinstance(seqs_list, dict): seqs_list = [seqs_list]

    seqs_list = [{keys_mapping[k]: v for k, v in seq_dict.items()} for seq_dict in seqs_list]

    for i in range(len(seqs_list)):
        # the ID is RNAcentraID_Taxon
        # seqs_list[i]['id'] = seqs_list[i]['id'].split('_')[:-1]
        seqs_list[i]['fname'] = fname
        seqs_list[i]['db'] = 'rnacentral'
        seqs_list[i]['moltype'] = moltype_mapping[seqs_list[i]['moltype'].replace('_', ' ')]

    if return_list:
        return seqs_list
    else:
        return seqs_list


def ct2bpseq_lines(ct, seq, id=None, return_list=True):
    """ ct should not contain zeros
        bpseq fmt: resnum resname resnum_bp (0 if unpaired)
    """

    seq_len = len(seq)
    resnums = np.linspace(1, seq_len, seq_len, dtype=int)
    resnums_bp = np.zeros(seq_len, dtype=int)
    if len(ct):
        resnums_bp[ct[:,0] - 1] = ct[:,1]
        resnums_bp[ct[:,1] - 1] = ct[:,0]

    bpseq_lines = np.char.add(np.char.add(
                resnums.astype(str),
                np.char.center(np.array(list(seq)), 3)),
                resnums_bp.astype(str))

    if isinstance(id, str):
        bpseq_lines = np.concatenate(([f'#{id}'], bpseq_lines))

    if return_list:
        return bpseq_lines.tolist()
    else:
        return bpseq_lines


def ctmat2bpseq_lines(ctmat, seq, id=None, return_list=True):
    """ bpseq fmt: resnum resname resnum_bp (0 if unpaired)
    """

    seq_len = len(seq)
    resnums = np.linspace(1, seq_len, seq_len, dtype=int)
    resnums_bp = np.where(ctmat.any(axis=1), ctmat.argmax(axis=1) + 1, 0)

    bpseq_lines = np.char.add(np.char.add(
                resnums.astype(str),
                np.char.center(np.array(list(seq)), 3)),
                resnums_bp.astype(str))

    if isinstance(id, str):
        bpseq_lines = np.concatenate(([f'#{id}'], bpseq_lines))

    if return_list:
        return bpseq_lines.tolist()
    else:
        return bpseq_lines


def parse_bpseq_lines(bpseq_info, comment='#', return_list=False):
    """ each line of bpseq contains 3 columns:
            resnum resname bp_resnum (or 0)
    returns the same as ct file: len, seq, ct, resnum
    """
    if isinstance(bpseq_info, str) or isinstance(bpseq_info, Path):
        fname = str(bpseq_info)
        file_lines = get_file_lines(fname, strip=True, comment=comment,
                    keep_empty=False, keep_comment=True)
    else:
        fname = ''
        file_lines = bpseq_info

    # find comments to be used as id
    comment_lines, bpseq_lines = [], []
    len_comment = len(comment)
    for _s in file_lines:
        if _s.startswith(comment):
            comment_lines.append(_s[len_comment:])
        else:
            bpseq_lines.append(_s)

    id = ' '.join(comment_lines)

    # parse bpseq_lines
    try:
        df = pd.read_csv(StringIO('\n'.join(bpseq_lines)), sep=r'\s+', skiprows=0,
                header=None, engine='c', names=['resnum', 'resname', 'resnum_bp'])
    except:
        print(f'Error in parsing {bpseq_lines}')

    resnum = df.resnum.to_numpy(dtype="int32")

    seqlen_ct = max([df.shape[0], resnum.max()])

    if seqlen_ct < df.resnum_bp.max():
        logger.warning(f'Max ct resnum: {df.resnum_bp.max()} > seqlen: {seqlen_ct}, file: {fname}')

    seq = np.array(['N'] * seqlen_ct)
    seq[resnum - 1] = df.resname.to_list()
    seq = ''.join(seq)

    # generate the list of contacts
    ct = df[['resnum', 'resnum_bp']].to_numpy(dtype="int32")

    # remove lines with resnum_bp = 0
    ct = ct[ct[:, 1] > 0]

    # sort and removes redundant [j, i] pairs
    # ct = ct[ct[:, 1] > ct[:, 0]]
    ct.sort(axis=1) # sort inplace
    ct = np.unique(ct, axis=0) # this keeps pseudoknots

    # collect data into a dict
    seq_dict = dict(ct=ct, len=seqlen_ct, seq=seq, resnum=resnum, id=id, fname=fname)

    # for i in range(seq_dict['len']):
    #     tokens = file_lines[i].split()
    #     if int(tokens[0]) != i + 1:
    #         logger.warning(f'residue number is corrupted: {file_lines[i]}')
    #         continue

    #     seq += tokens[1]

    #     j = int(tokens[2])
    #     if j == 0: continue

    #     bp.append([i + 1, j])
    #     bpmat[[i, j - 1], [j - 1, i]] = 1

    if return_list:
        return ct, id, seq
    else:
        return seq_dict


def parse_upp_lines(upp_lines, l=None):
    """ Format: resnum upp
    Only parse up to the length l, if passed """

    num = l if l else len(upp_lines)
    if num < len(upp_lines):
        logger.warning(f'Passed length: {l} < # of upp lines: {len(upp_lines)}')

    upp_idx = np.zeros((num,), dtype=np.int32)
    upp_val = np.zeros((num,), dtype=np.float32)

    for i, s in enumerate(upp_lines[:num]):
        tokens = s.split()
        upp_idx[i] = int(tokens[0])
        upp_val[i] = float(tokens[1])

    # check continuity of upper_idx
    if upp_idx[0] != 1 or any(upp_idx[1:] - upp_idx[:-1] != 1):
        logger.critical("Indices are not continuous in UPP data!")

    return upp_val


def parse_seq_lines(seq_info, fmt='fasta', has_id=True, has_seq=True, has_dbn=None, has_upp=None,
        return_namedtuple=False, return_list=False):
    """ Designed to parse all sequences in ONE file only, return a list of dict by default
    Requires:
            1) stripped at the ends
            2) no additional comment lines
    """
    if isinstance(seq_info, Path) or isinstance(seq_info, str):
        fname = str(seq_info)
        seq_lines = get_file_lines(fname, keep_empty=False, strip=True, comment='#', keep_comment=False)
    else:
        fname = ''
        seq_lines = seq_info

    fmt = fmt.lower()
    if fmt in ['fasta', 'dbn', 'seqres']:
        id_tag = '>'
    elif fmt == 'seq':
        id_tag = ';'
    else:
        id_tag = None
        logger.critical(f'Unknow sequence file format: {fmt}')

    num_lines = len(seq_lines)
    # logger.debug(f'Total number of lines: {num_lines}')

    # find all lines with id tags ------ [num_lines] is added!!!
    seq_starts = np.array([_i for _i in range(num_lines) if seq_lines[_i][0] == id_tag] \
                    + [num_lines], dtype=int)

    # find the index for real sequences only (>=2 lines for fasta, >=3 for seq)
    # e.g., multiple ";" lines are possible in SEQ format, only need the last one
    if len(seq_starts) == 1:
        logger.critical(f'No lines with id tag: {id_tag} found for file: {fname} !!!')
        print(seq_lines)
        idx_valid_seq = []
    else:
        # remove consecutive lines with the id tag (only applicable for SEQ format)
        nlines_per_seq = seq_starts[1:] - seq_starts[:-1]
        idx_valid_seq = np.where(nlines_per_seq > (2 if fmt == 'seq' else 1))[0]

    seqs_dict = []

    # a slow way..., but nice to put SEQ and FASTA together
    for i, iline in enumerate(seq_starts[:-1]):
        if i not in idx_valid_seq: continue

        iend = seq_starts[i + 1]

        seq_dict = dict(fname=fname, id='', seq='', dbn='', upp=np.empty(0, dtype=np.float32))

        if has_id:
            if fmt in ['seq']:
                iline += 1 # ID is one line after the last ";"
                seq_dict['id'] = seq_lines[iline]
            else:
                seq_dict['id'] = seq_lines[iline][1:]
            # id = id.replace(',', ':')
            iline += 1

        # assume both the sequence and dbn are one-liners
        if has_seq:
            if not has_dbn and not has_upp: # use all remaining lines
                seq_dict['seq'] = ''.join(seq_lines[iline:iend])
            else:
                seq_dict['seq'] = seq_lines[iline]

            seq_dict['seq'] = misc.str_deblank(seq_dict['seq'])
            if fmt == 'seq':
                seq_dict['seq'] = seq_dict['seq'][:-1] # remove 1 at the end

            iline += 1

        if has_dbn:
            tokens = misc.str_deblank(seq_lines[iline]).split()
            seq_dict['dbn'] = tokens[0]
            seq_dict['ct'] = dbn2ct(seq_dict['dbn'])
            if len(tokens) > 1 and tokens[1][0] == '(' and tokens[1][-1] == ')':
                seq_dict['energy'] = float(tokens[1][1:-1])
            iline += 1

        if has_upp:
            seq_dict['upp'] = parse_upp_lines(seq_lines[iline:iend])

        seqs_dict.append(seq_dict)

    if return_namedtuple:
        SeqInfo = namedtuple('SeqInfo', ['fname', 'id', 'seq', 'dbn', 'upp'], defaults=[None]*4)
        return [SeqInfo(**_d) for _d in seqs_dict]
    elif return_list:
        return [list(_d.values()) for _d in seqs_dict]
    else:
        return seqs_dict


def parse_st_lines(seq_info, fmt='sta', seq=True, dbn=True, ssn=True, nkn=True,
            comment='#', return_namedtuple=False, return_list=False):
    """ Designed to parse ONE structure type (array) file as defined by bpRNA
        fmt: st (structure type) or sta (structure type array)
        ssn: H (hairpin loop), I (interior loop), B (bulge), M (multi-loop), S (stack), E (external)
        nkn: N-regular, K-pseudoknot
    """
    if isinstance(seq_info, Path) or isinstance(seq_info, str):
        fname = str(seq_info)
        seq_lines = get_file_lines(fname, keep_empty=False, strip=True, keep_comment=True)
    else:
        fname = ''
        seq_lines = seq_info

    fmt = fmt.lower()
    num_lines = len(seq_lines)
    # logger.debug(f'Total number of lines: {num_lines}')

    # find the first among the #... lines
    id_tag = '#'
    seq_starts = ([0] if seq_lines[0][0] == id_tag else []) + \
                [_i for _i in range(1, num_lines) if seq_lines[_i][0] == id_tag and seq_lines[_i - 1][0] != id_tag] + \
                [num_lines]
    seq_starts = np.array(seq_starts, dtype=np.int32)

    # find the index for real sequences only (>=2 lines for fasta, >=3 for seq)
    # e.g., multiple ";" lines are possible in SEQ format, only need the last one
    if len(seq_starts) == 1:
        logger.critical(f'No lines with id tag: {id_tag} found!!!')
        valid_iseqs = []
    else:
        # remove consecutive lines with the id tag (only applicable for SEQ format)
        nlines_per_seq = seq_starts[1:] - seq_starts[:-1]
        valid_iseqs = np.where(nlines_per_seq > 1)[0]

    seqs_list = []

    # a slow way..., but nice to put SEQ and FASTA together
    for iseq, iline in enumerate(seq_starts[:-1]):
        if iseq not in valid_iseqs: continue

        iend = seq_starts[iseq + 1]

        seq_dict = dict(fname=fname, id='', seq='', dbn='', ssn='', nkn='')

        # get the lines starting with #
        while seq_lines[iline][0] == id_tag:
            tokens = seq_lines[iline][1:].split(':')
            iline += 1
            if len(tokens) != 2:
                logger.warning(f'One and only one : is expected in {seq_lines[iline]}!!!')
                continue
            if tokens[0] == 'Name':
                seq_dict['id'] = tokens[1]
            elif tokens[0] == 'Length':
                seq_dict['len'] = int(misc.numbers_first(tokens[1]))
            elif tokens[0] in ['Warning']:
                seq_dict[tokens[0]] = tokens[1]
            elif tokens[0] in ['Length', 'PageNumber']:
                seq_dict[tokens[0]] = int(misc.numbers_first(tokens[1]))
            else:
                logger.warning(f'Unrecognized fields: {tokens}')

        # assume both the sequence and dbn are one-liners
        if seq:
            seq_dict['seq'] = misc.str_deblank(seq_lines[iline])
            iline += 1

        if dbn:
            seq_dict['dbn'] = misc.str_deblank(seq_lines[iline])
            seq_dict['ct'] = dbn2ct(seq_dict['dbn'])
            iline += 1

        if ssn:
            seq_dict['ssn'] = misc.str_deblank(seq_lines[iline])
            iline += 1

        if nkn:
            seq_dict['nkn'] = misc.str_deblank(seq_lines[iline])
            iline += 1

        if fmt == 'sta': # sta format doesnot have the list of structure motifs at the end
            if iline != iend:
                logger.warning(f'Next line: {iline} should be equal to iend: {iend}')

        # final check
        if not seq_dict['len'] == len(seq_dict['seq']) == len(seq_dict['dbn']) == \
                len(seq_dict['ssn']) == len(seq_dict['nkn']):
            logger.warning(f'Inconsistent lengths between len/seq/dbn/ssn/nkn for fname: {fname}!!!')

        seq_dict['resnum'] = np.linspace(1, len(seq_dict['seq']), len(seq_dict['seq']), dtype=int)

        seqs_list.append(seq_dict)

    if return_namedtuple:
        SeqInfo = namedtuple('SeqInfo', ['fname', 'id', 'seq', 'dbn', 'ct', 'ssn', 'nkn'],
                    defaults=[None]*4)
        return [SeqInfo(**_d) for _d in seqs_list]
    elif return_list:
        return [list(_d.values()) for _d in seqs_list]
    else:
        return seqs_list


def parse_tangle_lines(tangle_info, return_list=False):
    """ this is the self-defined dbn/torsion angle csv data
    current columns are:
        resnum | nt | dbn | alpha | beta | gamma | delta | epsilon | zeta | chi | phase_angle |
    """
    if isinstance(tangle_info, str) or isinstance(tangle_info, Path):
        fname = str(tangle_info)
        tangle_lines = get_file_lines(fname, keep_empty=False, comment='#', keep_comment=False)
    else:
        fname = ''
        tangle_lines = tangle_info

    try:
        df = pd.read_csv(StringIO('\n'.join(tangle_lines)), sep=',', skiprows=None,
                header='infer', engine='c')
    except:
        print(f'Error in parsing {tangle_lines}')

    seq_dict = dict(fname=fname, id='', seq='', dbn='', ssn='', nkn='')

    resnums = df.resnum.to_numpy(dtype=np.int32)
    seq_dict = {'fname':fname, 'seq':df.nt.sum(), 'dbn':df.dbn.sum()}
    seq_dict['ct'] = dbn2ct(seq_dict['dbn'])
    seq_dict['tangle_labels'] = df.columns.to_list()
    for _name in ['resnum', 'nt', 'dbn']:
        seq_dict['tangle_labels'].remove(_name)
    seq_dict['tangle'] = df[seq_dict['tangle_labels']].to_numpy(dtype=np.float32)

    if return_list:
        return seq_dict['seq'], seq_dict['dbn'], seq_dict['ct']
    else:
        return seq_dict


def parse_ct_lines(ct_info, return_list=False):
    """  """
    if isinstance(ct_info, str) or isinstance(ct_info, Path):
        fname = str(ct_info)
        ct_lines = get_file_lines(fname, keep_empty=False, comment='#', keep_comment=False)
    else:
        fname = ''
        ct_lines = ct_info

    tokens_header = ct_lines[0].split()
    seqlen_header = int(tokens_header[0])
    if tokens_header[1].upper().startswith('ENERGY'):
        energy = float(tokens_header[3])
        id = ' '.join(tokens_header[4:])
    else:
        energy = 0.0
        id = ' '.join(tokens_header[1:])

    # id = id.replace(',', ':')

    try:
        df = pd.read_csv(StringIO('\n'.join(ct_lines[1:])), sep=r'\s+', skiprows=0,
                header=None, engine='c',
                names=['resnum', 'resname', 'nn_before', 'nn_after', 'resnum_bp', 'resnum2'])
    except:
        print(f'Error in parsing {ct_lines}')

    df_len = len(df)
    resnum = df.resnum.to_numpy(dtype=np.int32)
    resnum_max = resnum[-1]

    # resnum.max()=resnum[-1] should be the same as len(df)
    if resnum_max != df_len:
        logger.warning(f'resnum max: {resnum[-1]} != len(df): {df_len}, file: {fname}')

    if seqlen_header != resnum_max:
        logger.warning(f'resnum max: {resnum_max} != header len: {seqlen_header}, file: {fname}')

    resnum_bp_max = df.resnum_bp.max()
    if resnum_max < resnum_bp_max:
        logger.warning(f'resnum max: {resnum_max} < resnum_bp max: {resnum_bp_max}, file: {fname}')
        seq = np.array(['N'] * resnum_bp_max)
    else:
        seq = np.array(['N'] * resnum_max)

    seq[resnum - 1] = df.resname.to_list()
    seq = ''.join(seq)

    # generate the list of contacts
    ct = df[['resnum', 'resnum_bp']].to_numpy(dtype=np.int32)

    # remove lines with resnum_bp = 0
    ct = ct[ct[:, 1] > 0]

    # sort and removes redundant [j, i] pairs
    # ct = ct[ct[:, 1] > ct[:, 0]]
    ct.sort(axis=1) # sort in place
    ct = np.unique(ct, axis=0) # this keeps pseudoknots

    # collect data into a dict
    seq_dict = dict(ct=ct, len=len(seq), energy=energy, seq=seq,
                resnum=resnum, id=id, fname=fname)

    if return_list:
        return ct, id, seq, len(seq), energy
    else:
        return seq_dict


def parse_bps_lines(bps_info):
    """ bps is the format from SPOT-RNA datasets """
    if isinstance(bps_info, str) or isinstance(bps_info, Path):
        ct_lines = get_file_lines(bps_info, keep_empty=False, comment='#', strip=True, keep_comment=False)
    else:
        ct_lines = bps_info

    tokens_header = ct_lines[0].split()
    assert tokens_header == ['i', 'j'], "the header of spot-rna bps must be i j!!!"

    if len(ct_lines) < 2:
        ct = np.empty((0, 2), dtype=int)
    else:
    # try:
        ct = np.loadtxt(StringIO('\n'.join(ct_lines[1:])), dtype=int)
    # except:
        # print(f'Error in parsing {ct_lines}')

    return np.expand_dims(ct, 0) if ct.ndim == 1 else ct


def dbn2ct(dbn, check_closure=False):
    """ https://www.tbi.univie.ac.at/RNA/ViennaRNA/doc/html/rna_structure_notations.html

    Unsupported symbols:
        ~: insertions

    When check_closure=True, return True if all pairs are matched else False
    """
    letters = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
    open_sym = '([{<' + letters
    shut_sym = ')]}>' + letters.lower()

    # initialize for open and shut stacks to share the SAME list
    open_stack = {}
    shut_stack = {}
    for i in range(len(open_sym)):
        open_stack[open_sym[i]] = shut_stack[shut_sym[i]] = []

    # get contact table
    ct = []
    for i, s in enumerate(dbn):
        # all loops: hairpin(_) buldge&interior(-) multibranch(,) exterior(:)
        if s in '._-,:':
            pass
        elif s in open_stack:
            open_stack[s].append(i + 1)
        elif s in shut_stack:
            # will raise exception if stack is empty (just don't want to deal with it now)
            if len(shut_stack[s]):
                ct.append([shut_stack[s].pop(), i + 1])
            elif check_closure:
                return False
            else:
                logger.error(f'Found umatching symbols for {s}!')
        else:
            logger.error(f'Unrecognized dbn symbol: {s}')

    # check for unpaired residues
    for key, pos in open_stack.items():
        if len(pos) == 0: continue
        if check_closure:
            return False
        else:
            logger.error(f'Umatched key: {key} at pos: {pos} for dbn: {dbn}!!!')

    if check_closure:
        return True

    # return
    if len(ct):
        ct = np.stack(ct)
        ct = ct[ct[:,0].argsort()] # sort by the first column
        return ct
    else:
        logger.warning(f'No contact table generated for dbn: {dbn}')
        return np.empty((0,0), dtype=np.int)


def dbn2upp(dbn):

    dbn = np.array(list(dbn))
    upp = np.zeros_like(dbn, dtype=float)
    upp[dbn == '.'] = 1.0
    return upp


def bpp2upp(bpp, l=None):
    """ bpp is the base-pair probabilities from linear_partition, [i, j, p]
        upp is the unpaired probabilities
    """
    if len(bpp) == 0:
        return np.zeros(l, dtype=np.float32)
    if l is None: l = int(bpp.max())

    upp = np.zeros(l, dtype=np.float32)

    for i in [0, 1]:
        idx_sorted = np.argsort(bpp[:, i])
        bpp_sorted = bpp[idx_sorted]
        resnums_unique, idx_unique, resnum_counts = \
            np.unique(bpp_sorted[:, i].astype(int) - 1, return_index=True, return_counts=True)

        # add the first occurrence of the resum
        upp[resnums_unique] += bpp_sorted[idx_unique, 2]

        idx_unique = np.append(idx_unique, [bpp_sorted.shape[0]], axis=0)
        # additional base pairing for some bases
        for j in np.where(resnum_counts > 1)[0]:
            upp[resnums_unique[j]] += bpp_sorted[(idx_unique[j] + 1):idx_unique[j+1], 2].sum()

    return 1.0 - upp


def ct2bpnum(ct, l=None):
    """ return a [l] vector indicating the number of pairing base at each i
        ct should be nx2 numpy array
    """
    if l is None: l = ct.max()

    # method 1:
    bpnum = np.zeros(l, dtype=int)
    np.add.at(bpnum, ct[:] - 1, 1)

    return bpnum

    # method 2:
    return np.bincount(ct[:] - 1, minlength=l)

    if ct.ndim == 2:
        bpnum[ct[:] - 1] += 1
    elif len(ct):
        logger.warning(f'ct is not empty ({ct}) but ct.ndim != 2')

    return bpnum


def ct2dbn(ct, l=None):
    """ ct should be nx2 numpy array """
    if l is None: l = ct.max()

    dbn = np.array(['.'] * l)

    if ct.ndim == 2:
        dbn[ct[:, 0] - 1] = '('
        dbn[ct[:, 1] - 1] = ')'
    elif len(ct):
        logger.warning(f'ct is not empty ({ct}) but ct.ndim != 2')

    return ''.join(dbn)


def ct2upp(ct, l=None, dtype=np.float32):
    """ ct should be nx2 numpy array """
    if l is None: l = ct.max()

    upp = np.ones(l, dtype=dtype)

    if ct.ndim == 2:
        upp[ct[:] - 1] = 0.0
    elif len(ct):
        logger.warning(f'ct is not empty ({ct}) but ct.ndim != 2')

    return upp


def ct2ctmat(ct, l=None, dtype=np.int32):
    """ ct should be nx2 numpy array and idx starts from 1 """
    if l is None: l = ct.max()

    ctmat = np.zeros((l, l), dtype=dtype)

    if ct.ndim == 2 and ct.shape[0]:
        ict = ct -1 # ct starts from 1 instead of 0

        ctmat[ict[:, 0], ict[:, 1]] = 1
        ctmat[ict[:, 1], ict[:, 0]] = 1
    elif len(ct):
        logger.warning(f'ct is not empty ({ct}) but ct.ndim != 2')

    return ctmat


def ppmat2ctmat(ppmat, threshold=0.5, dtype=np.int32):
    """ ppmat is continuous, return ct as nx2 numpy array """

    ctmat = np.zeros(ppmat.shape, dtype=dtype)

    idx_max = (np.arange(len(ppmat)), ppmat.argmax(axis=0))
    if threshold is None:
        ctmat[idx_max] = 1
    else:
        ctmat[idx_max] = (ppmat[idx_max] >= threshold).astype(ctmat.dtype)

    return ctmat


def ctmat2ct(ctmat):
    """ cmat is discrete (0 or 1), return ct as nx2 numpy array """

    ct = np.stack(np.where(ctmat == 1), axis=1) + 1
    ct = ct[ct[:, 1] > ct[:, 0]]

    return ct


def count_pseudoknot(bp, sorted=True, return_resnums=False):
    """ bp/ct is nx2 numpy array (default: the first column already sorted)
        condition: i1 < i2 < j1 < j2
    """
    if len(bp) < 2: return 0
    if not sorted:
        bp = np.sort(bp, axis=1) # i < j for each bp
        bp = bp[bp[:, 0].argsort()] # i1 < i2 for all bps

    resnums = []
    num_pknots = 0
    in_a_pknot = False
    # for a sorted pair list, only need to check i2 < j1 < j2
    for _i, j1 in enumerate(bp[:-1, 1]):
        _i += 1
        if np.logical_and(bp[_i:, 0] < j1, j1 < bp[_i:, 1]).any():
            if not in_a_pknot:  # a new knot
                num_pknots += 1
                resnums_newknot = []
                in_a_pknot = True
            resnums_newknot.append(j1)
        else:
            if in_a_pknot: # was inside a pseudo knot
                resnums.append(resnums_newknot)
                in_a_pknot = False

    if return_resnums:
        return num_pknots, resnums
    else:
        return  num_pknots

    # # get the number of continuous segments
    # if len(resnums):
    #     resnums = np.sort(np.array(resnums))
    #     print(resnums)
    #     return (np.diff(resnums) > 1).astype(int).sum() + 1
    # else:
    #     return 0


def count_residue_names(seq, return_tuple=True):
    """ example: return_tuple --> (('A', 'G'), (3, 5)) """

    res_counter = Counter(seq)

    if return_tuple:
        res_names, res_counts = tuple(zip(*res_counter.most_common()))
        return res_names, res_counts
    else:
        return res_counter

    upper_count = dict()
    lower_count = dict()
    other_count = dict()

    for key, val in res_count.items():
        if key in ['A', 'U', 'G', 'C', 'T']:
            upper_count[key] = str(val)
        elif key in ['a', 'u', 'g', 'c', 't']:
            lower_count[key] = str(val)
        else:
            other_count[key] = val

    return ['|'.join(upper_count.keys()), list(upper_count.values()),
            '|'.join(lower_count.keys()), list(lower_count.values()),
            '|'.join(other_count.keys()), list(other_count.values()),
    ]


def count_ct_bpnums(ct, l=None, return_tuple=True):
    """ count how many neighbors/pairs each base has """
    num_counter = Counter(ct2bpnum(ct, l=l))
    if return_tuple:
        return tuple(zip(*num_counter.most_common()))
    else:
        return num_counter


def count_ct_bptypes(ct, seq, return_tuple=True):
    """ Count the types of paired bases in ct and seq

        renames are converted to upper case
        base pairs are ordered, e.g., UA and AU are assigned to AU
    """

    if len(ct) == 0 or len(seq) == 0:
        logger.warning('Empty ct and/or seq!')
        if return_tuple:
            return ((), ())
        else:
            return Counter()

    if ct.ndim == 1:
        ct = np.expand_dims(ct, 0)

    seq = np.array(list(seq.upper()))
    seq_ct = np.sort(seq[ct - 1], 1)
    bp_counter = Counter(np.char.add(seq_ct[:,0], seq_ct[:,1]))
    if return_tuple:
        return tuple(zip(*bp_counter.most_common()))
    else:
        return bp_counter


def count_ct_stemlens(ct, warn_single=False, return_tuple=False, sort='key'):
    """ count the length distribution of stems in the ct nx2 array
    [:, 0] of ct must be sorted from small to large

        The recipe is to find the INDEX of breaks in i+j is not constant or i is not
    continuous. Then, the differences between the break INDEX are stem lengths

    """
    stem_counter = Counter()

    if ct.ndim == 1:
        ct = np.expand_dims(ct, 0)

    ct_len = len(ct)

    if ct_len == 0:
        logger.debug('Empty ct!!!')
    elif ct_len == 1:
        logger.debug(f'Only one entry in ct: {ct}!')
        stem_counter.update([1]) # length of 1
    else:
        contig_breaks = np.nonzero(np.logical_or(
            np.diff(ct.sum(axis=1), prepend=0, append=0),   # i+j not constant
            np.diff(ct[:, 0], prepend=-2, append=-2) != 1)  # i not continuous
            )[0]
        stem_lengths = contig_breaks[1:] - contig_breaks[0:-1]

        if warn_single:
            idx_singles = np.nonzero(stem_lengths == 1)[0]
            if len(idx_singles):
                logger.warning('Found stem length of 1 in ct!!!')
                for _i in idx_singles:
                    print(ct[contig_breaks[_i]-1:contig_breaks[_i]+2])

        # stem_counter.update(np.char.add(stem_lengths.astype(str), 'S'))
        stem_counter.update(stem_lengths)

    if return_tuple:
        if len(stem_counter):
            sort = sort.lower()
            if sort in ['count', 'counts', 'val', 'vals']:
                return tuple(zip(*stem_counter.most_common()))
            elif sort in ['key', 'keys', 'numeric', 'numerical']:
                return tuple(zip(*sorted(stem_counter.items())))
            else:
                return tuple(zip(*stem_counter.items()))
        else:
            return ((), ())
    else:
        return stem_counter


def count_ct_deltaij(ct, warn_single=False, return_tuple=False, sort='key'):
    """ count the length distribution of stems in the ct nx2 array
    [:, 0] of ct must be sorted from small to large

        The recipe is to find the INDEX of breaks in i+j is not constant or i is not
    continuous. Then, the differences between the break INDEX are stem lengths

    """
    deltaij_counter = Counter()

    if ct.ndim == 1:
        ct = np.expand_dims(ct, 0)

    ct_len = len(ct)

    if ct_len == 0:
        logger.debug('Empty ct!!!')
    elif ct_len == 1:
        logger.debug(f'Only one entry in ct: {ct}!')
        deltaij_counter.update(ct[:, 1] - ct[:, 0])
    else:
        deltaij_counter.update(ct[:,1] - ct[:,0])

    if return_tuple:
        if len(deltaij_counter):
            sort = sort.lower()
            if sort in ['count', 'counts', 'val', 'vals']:
                return tuple(zip(*deltaij_counter.most_common()))
            elif sort in ['key', 'keys', 'numeric', 'numerical']:
                return tuple(zip(*sorted(deltaij_counter.items())))
            else:
                return tuple(zip(*deltaij_counter.items()))
        else:
            return ((), ())
    else:
        return deltaij_counter


class SeqsData():
    def __init__(self, fnames=None, **kwargs):
        self.db = [] # a list of string ('archiveII', 'strAlign', 'bpRNA', etc)
        self.file = []
        self.type = [] # a list of string (not used yet)
        self.title = []
        self.moltype = [] # a list of string ('16S', '5S', etc)
        self.taxon = np.empty(0, dtype=np.int32)
        self.desc = []
        self.md5 = []

        # 1D info
        self.id = []   # a list of string
        self.seq = []  # a list of string
        self.upp = []  # a list of np.array(1d) (up paird probability)
        self.len = np.empty(0, dtype=np.int32) # a 1D np array
        self.nbreaks = np.empty(0, dtype=np.int32)  # a 1D np array
        self.npknots = np.empty(0, dtype=np.int32)  # a 1D np array
        self.energy = np.empty(0, dtype=np.float32)

        self.tangle = [] # torsion/rotation/etc angles
        self.tangle_labels = []

        # 2D info
        self.dbn = []  # a list of string (dot bracket notation)
        self.ssn = []  # a list of string (secondary structure notation by bpRNA: SMIH.. )
        self.nkn = []  # a list of string (N or K notation by bpRNA, K for pseudoknots)
        self.ct = []   # a list of np.array(2d) (contact table) for both ct and bpseq files
        self.ctmat = [] # a list of 2D numpy array
        self.bp = []   # bpseq array? not used yet as
        self.bpmat = []

        if fnames is not None:
            self.parse_files(fnames, **kwargs)

    def get_subset(self, idx, keep=True):
        """  """
        seqs_subset = SeqsData()
        if not hasattr(idx, '__len__'): idx = [idx]

        # it can be expedited by using slices for continuous indices
        for i in idx:
            seqs_subset.db.append(self.db[i])
            seqs_subset.file.append(self.file[i])
            seqs_subset.type.append(self.type[i])
            seqs_subset.title.append(self.title[i])
            seqs_subset.moltype.append(self.moltype[i])
            seqs_subset.id.append(self.id[i])
            seqs_subset.desc.append(self.desc[i])
            seqs_subset.md5.append(self.md5[i])
            seqs_subset.upp.append(self.upp[i])
            seqs_subset.dbn.append(self.dbn[i])
            seqs_subset.ssn.append(self.ssn[i])
            seqs_subset.nkn.append(self.nkn[i])
            seqs_subset.ct.append(self.ct[i])
            seqs_subset.bp.append(self.bp[i])
            seqs_subset.ctmat.append(self.ctmat[i])
            seqs_subset.bpmat.append(self.bpmat[i])

        seqs_subset.taxon = self.taxon[idx]
        seqs_subset.len = self.len[idx]
        seqs_subset.nbreaks = self.nbreaks[idx]
        seqs_subset.npknots = self.npknots[idx]
        seqs_subset.energy = self.energy[idx]

        return seqs_subset

    def add_dummy_seq(self, num=1):
        """  """
        self.db.extend([''] * num)
        self.file.extend([''] * num)
        self.type.extend([''] * num)
        self.title.extend([''] * num)
        self.moltype.extend([''] * num)
        self.taxon = np.pad(self.taxon, (0, num), 'constant', constant_values=(0, 0))
        self.desc.extend([''] * num)
        self.md5.extend([''] * num)
        self.id.extend([''] * num)
        self.seq.extend([''] * num)
        self.upp.extend([np.empty(0, dtype=np.float32) for i in range(num)])
        self.len = np.pad(self.len, (0, num), 'constant', constant_values=(0, 0))
        self.nbreaks = np.pad(self.nbreaks, (0, num), 'constant', constant_values=(0, 0))
        self.npknots = np.pad(self.npknots, (0, num), 'constant', constant_values=(0, 0))
        self.energy = np.pad(self.energy, (0, num), 'constant', constant_values=(0., 0.))

        self.tangle.extend([np.empty((0, 0), dtype=np.float32) for _i in range(num)])

        self.dbn.extend([''] * num)
        self.ssn.extend([''] * num)
        self.nkn.extend([''] * num)
        self.ct.extend([np.empty((0,2), dtype=np.int32) for _i in range(num)])
        self.bp.extend([np.empty((0,2), dtype=np.int32) for _i in range(num)])
        self.ctmat.extend([np.empty((0,2), dtype=np.int32) for _i in range(num)])
        self.bpmat.extend([np.empty((0,2), dtype=np.int32) for _i in range(num)])

    def add_seq_from_dict(self, seqs_list, istart=None, fname=None):
        """ seqs_dict is a dict (or list of dict).
            each dict contains one sequence only.
        """
        if type(seqs_list) not in (list, tuple):
            seqs_list = [seqs_list]
        if istart is None:
            istart = len(self.seq) # index of self.id/seq/...
        if type(fname) not in (list, tuple):
            fname = [fname]

        num_seqs = len(seqs_list)
        if (istart + num_seqs) > len(self.seq):
            self.add_dummy_seq(istart + num_seqs - len(self.seq))

        fname += [fname[-1]] * (num_seqs - len(fname))

        for i in range(num_seqs):
            iself = istart + i

            self.db[iself] =  seqs_list[i].get('db', self.db[iself])
            self.file[iself] = seqs_list[i].get('fname', self.file[iself]) \
                                if fname[i] is None else fname[i]
            self.title[iself] = seqs_list[i].get('title', self.title[iself])
            self.moltype[iself] =  seqs_list[i].get('moltype', self.moltype[iself])
            self.desc[iself] =  seqs_list[i].get('desc', self.desc[iself])
            self.md5[iself] = seqs_list[i].get('md5', self.md5[iself])
            self.taxon[iself] = seqs_list[i].get('taxon', self.taxon[iself])
            self.id[iself] =  seqs_list[i].get('id', self.id[iself])
            self.seq[iself] = seqs_list[i].get('seq', self.seq[iself])
            self.len[iself] = len(self.seq[iself])
            self.dbn[iself] = seqs_list[i].get('dbn', self.dbn[iself])
            self.ssn[iself] = seqs_list[i].get('ssn', self.ssn[iself])
            self.nkn[iself] = seqs_list[i].get('nkn', self.nkn[iself])
            self.upp[iself] = seqs_list[i].get('upp', self.upp[iself])
            self.len[iself] = seqs_list[i].get('len', self.len[iself])
            self.energy[iself] = seqs_list[i].get('energy', self.energy[iself])

            self.tangle[iself] = seqs_list[i].get('tangle', self.tangle[iself])
            self.ct[iself] = seqs_list[i].get('ct', self.ct[iself])

        self.tangle_labels = seqs_list[i].get('tangle_labels', self.tangle_labels)

    def parse_sequence_file_old(self, fnames, fmt='fasta', id=True, seq=True, dbn=None, upp=None,
                istart=None, **kwargs):
        """ the default is fasta file with id and seq fields """

        # set up parameters
        if isinstance(fnames, str): fnames = [fnames]
        fmt = fmt.lower()
        if fmt == 'fasta':
            id_tag = '>'
        elif fmt == 'seq':
            id_tag = ';'
        else:
            id_tag = None
            logger.critical(f'Unknow sequence file format: {fmt}')

        # read the files
        str_lines = get_file_lines(fnames, strip=True, keep_empty=False)
        num_lines = len(str_lines)
        logger.debug(f'File <{str(fnames)}> has {num_lines} of lines')

        # find all lines with id tags ------ [num_lines] is added!!!
        idx_id = np.array([_i for _i in range(num_lines)
                if str_lines[_i][0] == id_tag] + [num_lines], dtype=np.int)

        # find the indices for real sequences only (>=2 lines for fasta, >=3 for seq)
        # e.g., multiple ";" lines are possible in SEQ format, only need the last one
        if len(idx_id) == 1:
            logger.critical(f'No lines with id tag: {id_tag} found!!!')
            idx_seqs = []
        else:
            # remove consecutive lines with the id tag (only applicable for SEQ format)
            nlines_per_seq = idx_id[1:] - idx_id[:-1]
            idx_seqs = np.where(nlines_per_seq > (2 if fmt == 'seq' else 1))[0]

        num_seqs = len(idx_seqs)
        logger.info(f'Total number of sequence files: {len(fnames)} with {num_lines} lines')
        logger.info(f'Found {len(idx_id) - 1} id tags, and {num_seqs} candidate sequences')

        # Group fasta_lines into groups for parallel processing
        lines_grpby_seq = []
        for i, iline in enumerate(idx_id[:-1]):
            # logger.debug(f'Processing sequence # {iseq}')
            if i not in idx_seqs: continue
            lines_grpby_seq.append(str_lines[iline:idx_id[i + 1]])

        # Ready to parse each sequence
        parse_func = functools.partial(parse_seq_lines, fmt=fmt, id=id_tag, seq=seq,
                    dbn=dbn, upp=upp, return_dict=True)

        if num_seqs > 7:
            num_cpus = round(mpi.cpu_count() * 0.8)
            logger.info(f'Using multiprocessing for parsing, cpu count: {num_cpus}')
            mpool = mpi.Pool(processes=num_cpus)

            parsed_seq_dicts = mpool.map(parse_func, lines_grpby_seq)
            mpool.close()
        else:
            parsed_seq_dicts = [parse_func(_seq_lines) for _seq_lines in lines_grpby_seq]

        # Assign into the structure
        self.add_seq_from_dict(parsed_seq_dicts, istart=istart, fname=fnames)

        return

    def parse_seq_files_old(self, fnames, fmt='fasta', id=True, seq=True, dbn=None, upp=None,
                    istart=None, **kwargs):
        """ the default is fasta file with id and seq fields """

        # set up parameters
        if isinstance(fnames, str): fnames = [fnames]

        logger.info(f'Total number of sequence files: {len(fnames)}')
        fmt = fmt.lower()
        # Ready to parse each sequence
        if fmt in ['fasta', 'seq']:
            parse_func = functools.partial(parse_seq_lines, fmt=fmt, id=id, seq=seq,
                    dbn=dbn, upp=upp)
        elif fmt in ['st', 'sta']:
            parse_func = functools.partial(parse_st_lines, fmt=fmt, seq=True,
                    dbn=True, ssn=True, nkn=True)
        else:
            logger.error(f'Unsupported fmt: {fmt} for this functionality!!!')

        if len(fnames) > 7:
            num_cpus = round(mpi.cpu_count() * 0.8)
            logger.info(f'Using multiprocessing for parsing, cpu count: {num_cpus}')
            mpool = mpi.Pool(processes=num_cpus)
            parsed_seq_dicts = mpool.map(parse_func, fnames)
            mpool.close()
        else:
            parsed_seq_dicts = [parse_func(_seq_lines) for _seq_lines in fnames]

        # Assign into the structure
        parsed_seq_dicts = misc.unpack_list_tuple(parsed_seq_dicts)
        logger.info(f'Found total number of sequences: {len(parsed_seq_dicts)}')
        self.add_seq_from_dict(parsed_seq_dicts, istart=istart)

        return

    def parse_files(self, fnames, fmt='fasta', istart=None,
                has_seq=True, has_dbn=None, has_upp=None, database=None):
        """ the default is fasta file with id and seq fields

        For ct and bpseq files:
            use pandas.read_csv() which is written in C
            Each line in ct files has 6 columns:
            resnum resname resnum_before resnum_after resnum_ct/0 resnum
            Each line in bpseq files has 3 columns:
            resnum resname resnum_bp
         """
        if type(fnames) not in (list, tuple): fnames = [fnames]

        num_files = len(fnames)
        logger.info(f'Parsing {num_files} {fmt} file(s)...')

        # select parser
        if fmt:
            fmt = fmt.lower()
        else:
            fmt = Path(fnames[0].lower()).suffix[1:]

        if fmt in ['fasta', 'seq', 'seqres']:
            parse_func = functools.partial(parse_seq_lines, fmt=fmt, has_seq=has_seq,
                    has_dbn=has_dbn, has_upp=has_upp)
        elif fmt in ['dbn']:
            parse_func = functools.partial(parse_seq_lines, fmt=fmt, has_seq=has_seq,
                    has_dbn=True, has_upp=has_upp)
        elif fmt in ['st', 'sta']: # bpRNA structure type (array) format
            parse_func = functools.partial(parse_st_lines, fmt=fmt, seq=True,
                    dbn=True, ssn=True, nkn=True)
        elif fmt == 'ct':
            parse_func = functools.partial(parse_ct_lines, return_list=False)
        elif fmt == 'bpseq':
            parse_func = functools.partial(parse_bpseq_lines, return_list=False)
        elif fmt in ['json', 'rnacentral']:
            parse_func = functools.partial(parse_json_lines, fmt='rnacentral', return_list=False)
        elif fmt in ['tangle']:
            parse_func = parse_tangle_lines
        else:
            logger.error(f'Unsupported fmt: {fmt} for this functionality!!!')

        # parse with optional multiprocessing
        moltypes = None
        if num_files > 7:
            num_cpus = round(mpi.cpu_count() * 0.8)
            logger.info(f'Using multiprocessing for parsing, cpu count: {num_cpus}')
            with mpi.Pool(processes=num_cpus) as mpool:
                parsed_seq_dicts = mpool.map(parse_func, tqdm(fnames, desc='Parsing files'))
                if database is not None:
                    database = database.lower()
                    moltypes = mpool.map(
                        functools.partial(parse_moltype, database=database),
                        tqdm(fnames))
        else:
            parsed_seq_dicts = [parse_func(_fname) for _fname in fnames]
            if database is not None:
                database = database.lower()
                moltypes = [parse_moltype(_fname, database=database) for _fname in fnames]

        # assign db and moltype to each dict of sequence
        if moltypes is not None:
            assert len(parsed_seq_dicts) == len(moltypes), 'The number of moltypes and dicts must be the same!'
            for i in range(len(moltypes)):
                if type(parsed_seq_dicts[i]) is dict:
                    parsed_seq_dicts[i]['db'] = database
                    parsed_seq_dicts[i]['moltype'] = moltypes[i]
                    continue
                # a list of dict, assign the same type
                for j in range(len(parsed_seq_dicts[i])):
                    parsed_seq_dicts[i][j]['db'] = database
                    parsed_seq_dicts[i][j]['moltype'] = moltypes[i]

        # parse_seq_lines return a list of dicts rather than a dict
        if type(parsed_seq_dicts[0]) in (list, tuple):
            parsed_seq_dicts = misc.unpack_list_tuple(parsed_seq_dicts)

        logger.info(f'Parsed {num_files} {fmt} file(s), found {len(parsed_seq_dicts)} sequence(s)')

        # parse information in the seqres id
        if fmt in ['seqres']:
            seqres_dicts = misc.mpi_map(parse_seqres_id, \
                [_seq_dict['id'] for _seq_dict in parsed_seq_dicts])
            assert len(seqres_dicts) == len(parsed_seq_dicts), 'seqres and seq dicts must have the same length!'
            for i in range(len(parsed_seq_dicts)):
                parsed_seq_dicts[i].update(seqres_dicts[i])

        # Assign into the structure
        self.add_seq_from_dict(parsed_seq_dicts, istart=istart)
        return

    def __str__(self):
        return f'num_seqs: {len(self.seq)}, min_len: {self.len.min()}, max_len: {self.len.max()}'

    # def pad_length(self, max_len):
    #     """ pad seqence, upp, and secondary structures to max_len """
    #     for i in range(self._len):
    #         if self.numResids[i] < max_len: # adding dummy residues
    #             if self.seq[i] is not None:
    #                 self.seq[i] += '-' * (max_len-self.numResids[i])
    #             if self.dbn[i] is not None:
    #                 self.dbn[i] += '-' * (max_len-self.numResids[i])
    #             if self.upp[i] is not None:
    #                 self.upp[i] = np.concatenate((self.upp[i],
    #                             np.ones((max_len - self.numResids[i],))), axis=0)
    #         elif self.numResids[i] == max_len:
    #             pass
    #         else: # removing residues from the end
    #             if self.seq[i] is not None:
    #                 self.seq[i] = self.seq[i][0:max_len]
    #             if self.dbn[i] is not None:
    #                 self.dbn[i] = self.dbn[i][0:max_len]
    #             if self.upp[i] is not None:
    #                 self.upp[i] = self.upp[i][0:max_len]

    @property
    def summary(self):
        """  """
        self.synopsis = dict(
            num_seq = len(self.seq),
            max_len = self.len.max(),
            min_len = self.len.min(),
            std_len = self.len.std()
        )
        return self.synopsis

    def to_df(self, seq=True, ct=False, tangle=False, has=False,
                res_count=False, bp_count=False, stem_count=False, pknot_count=False):
        """ recommended to collect all data into pd.DataFrame for ease of processing """
        num_seqs = len(self.seq)
        dict2df = dict(
            idx = list(range(1, num_seqs + 1)),
            file = self.file, # [str(_f).replace(',', ':') for _f in self.file],
            db = self.db,
            moltype = self.moltype,
            id = self.id,
            len = self.len,
            title = self.title,
        )
        # stdUPP = [_upp.std() if len(_upp) else None for _upp in self.upp],
        # meanUPP = [_upp.mean() if len(_upp) else None for _upp in self.upp],
        if seq:
            dict2df['seq'] = self.seq
            dict2df['lenSeq'] = [len(_s) for _s in self.seq]
        if ct:
            dict2df['ct'] = self.ct
            dict2df['lenCT'] = [len(_v) for _v in self.ct]
        if tangle:
            dict2df['tangle'] = self.tangle
        if has:
            dict2df['hasDBN'] = [bool(_dbn) for _dbn in self.dbn]
            dict2df['hasUPP'] = [len(_upp) > 0 for _upp in self.upp]
            dict2df['hasCT'] = [len(_ct) > 0 for _ct in self.ct]
        if res_count:
            res_names, res_counts = tuple(zip(*misc.mpi_map(
                functools.partial(count_residue_names, return_tuple=True),
                self.seq, desc='Counting residues')))
            dict2df['resNames'] = ['|'.join(_s) for _s in res_names]
            dict2df['resNameCounts'] = ['|'.join(map(str, _c)) for _c in res_counts]

            # residue_counts = zip(*misc.mpi_map(count_seq_bases, self.seq))
            # for col_name in ['resGroup1', 'resCount1', 'resGroup2', 'resCount2', \
            #                 'resGroup3', 'resCount3']:
            #     dict2df[col_name] = next(residue_counts)
        if bp_count:
            bp_names, bp_counts = tuple(zip(*misc.mpi_map(
                functools.partial(count_ct_bptypes, return_tuple=True),
                zip(self.ct, self.seq),
                starmap=True, desc='Counting bptypes',
                )))
            dict2df['bpTypes'] = ['|'.join(_s) for _s in bp_names]
            dict2df['bpTypeCounts'] = ['|'.join(map(str, _c)) for _c in bp_counts]

            bp_nums, num_counts = tuple(zip(*misc.mpi_map(
                functools.partial(count_ct_bpnums, return_tuple=True),
                zip(self.ct, self.len),
                starmap=True, desc='Counting bpnums',
                )))
            dict2df['bpNums'] = ['|'.join(map(str, _n)) for _n in bp_nums]
            dict2df['bpNumCounts'] = ['|'.join(map(str, _c)) for _c in num_counts]

        if stem_count:
            stem_lens, stem_counts = tuple(zip(*misc.mpi_map(
                functools.partial(count_ct_stemlens, return_tuple=True, sort='key'),
                self.ct, desc='Counting stem length',
            )))

            dict2df['stemLens'] = ['|'.join(map(str, _l)) for _l in stem_lens]
            dict2df['stemLenCounts'] = ['|'.join(map(str, _c)) for _c in stem_counts]

            delta_ijs, delta_ij_counts = tuple(zip(*misc.mpi_map(
                functools.partial(count_ct_deltaij, return_tuple=True, sort='key'),
                self.ct, desc='Counting delta_ij'
            )))

            dict2df['deltaijs'] = ['|'.join(map(str, _l)) for _l in delta_ijs]
            dict2df['deltaijCounts'] = ['|'.join(map(str, _c)) for _c in delta_ij_counts]

            dict2df['hasSharpTurns'] = [any([_dij < 4 for _dij in _l]) for _l in delta_ijs]

        if pknot_count:
            dict2df['numPKnots'] = misc.mpi_map(count_pseudoknot, self.ct, desc='Counting pseudoknots')

        return pd.DataFrame(dict2df, copy=False)

    def get_fasta_lines(self, idx=None, line_width=None, line_break=False,
                dbn=False, upp=False, **kwargs):
        if idx is None:
            idx = list(range(len(self.seq)))
        elif isinstance(idx, int) or isinstance(idx, np.integer):
            idx = [idx]

        fasta_lines = []
        if line_break:
            for i in idx:
                fasta_lines.append(f'>{self.id[i]}\n')
                fasta_lines.append(f'{self.seq[i]}\n')
                if dbn: fasta_lines.append(f'{self.dbn[i]}\n')
                if upp: fasta_lines.extend(f'{self.upp[i].astype(str)}\n')
        else:
            for i in idx:
                fasta_lines.append(f'>{self.id[i]}')
                fasta_lines.append(self.seq[i])
                if dbn: fasta_lines.append(self.dbn[i])
                if upp: fasta_lines.extend(self.upp[i].astype(str))

        return fasta_lines

    def write_sequence_file(self, fasta=None, dbn=None, bpseq=None, save_dir='', id=None, single=True):
        assert bool(fasta) + bool(dbn) + bool(bpseq) == 1, \
            "One and only one of the 1D/2D structure files must be provided!"

        if fasta is not None:
            with open(os.path.join(save_dir, fasta), 'w') as hfile:
                hfile.write('>' + (id if id else molstru_config.pdbinfo2filename('+'.join(self.id))) + '\n')
                hfile.write(''.join(self.seq))
        elif dbn is not None:
            pass

    def write_base_pair_matrix(self, fname, fsuffix='.bpm', fdir=''):
        """ this adds residue numbers as the first column and the first row """
        bpm_file = os.path.join(fdir, fname+fsuffix if '.' not in fname else fname)
        sav_mat = np.zeros((self.len[0]+1, self.len[0]+1), dtype=int)
        sav_mat[1:, 1:] = self.bpmat
        sav_mat[0, :] = np.linspace(0, self.len[0], num=self.len[0]+1, dtype=int)
        sav_mat[:, 0] = np.linspace(0, self.len[0], num=self.len[0]+1, dtype=int)
        sav_mat = sav_mat.astype(str)
        sav_mat[0, 0] = ''
        np.savetxt(bpm_file, sav_mat, fmt='%4s')


class AtomsData(object):
    def __init__(self, pdbfile=None, pdbdir=''):
        """ """
        self.id = ''
        self.dir = ''
        self.atomLines = []
        if pdbfile is None:
            return
        if isinstance(pdbfile, str):
            pdbfile = os.path.join(pdbdir, pdbfile)
            self.dir, self.fname = os.path.split(pdbfile)
            self.id, self.suffix = os.path.splitext(self.fname)

            if not os.path.exists(pdbfile):
                pdbfile = pdbfile + '.pdb'
            if not os.path.exists(pdbfile):
                print("Cannot find file: {}".format(pdbfile))
                atom_lines = []
            with open(pdbfile, 'r') as ofile:
                atom_lines = ofile.readlines()
            print('Successfully read pdb file: {}'.format(pdbfile))
            # No stripping, keeping \n at the end
            atom_lines = [strline for strline in atom_lines
                          if ('ATOM  ' == strline[0:6]) or ('HETATM' == strline[0:6])]
        elif isinstance(pdbfile, list):
            atom_lines = pdbfile.copy()
            self.id = 'TEMP'
            self.dir = ''
        else:
            print("Neither a filename or list of strings was passed, returning!!!")
            atom_lines = []

        self.atomLines = np.array(atom_lines, dtype='U81')
        self.parse_atom_lines()

    def parse_atom_lines(self, atom_lines=None):
        if atom_lines is None:
            atom_lines = self.atomLines
        else:
            self.atomLines = np.array(atom_lines, dtype='U81')

        self.numAtoms = len(atom_lines)
        self.atomTypes = np.empty((self.numAtoms, ), dtype='U6')
        self.atomNums = np.empty((self.numAtoms, ), dtype=int)
        self.atomNames = np.empty((self.numAtoms, ), dtype='U5')
        self.atomAltlocs = np.empty((self.numAtoms, ), dtype='U1')
        self.residNames = np.empty((self.numAtoms, ), dtype='U3')
        self.chainIds = np.empty((self.numAtoms, ), dtype='U1')
        self.residNums = np.empty((self.numAtoms, ), dtype=int)
        self.residICodes = np.empty((self.numAtoms, ), dtype='U1')
        self.xyz = np.empty((self.numAtoms, 3), dtype=float)
        self.occupancys = np.empty((self.numAtoms, ), dtype=float)
        self.tempFactors = np.empty((self.numAtoms, ), dtype=float)
        self.elementSyms = np.empty((self.numAtoms, ), dtype='U2')
        self.atomCharges = np.empty((self.numAtoms, ), dtype='U2')

        self.iAtomNewChain = []
        self.iAtomNewResid = []
        self.iSeqNewChain = []

        last_line = '#'*81
        ierror_lines = []
        for _i, _aline in enumerate(atom_lines):
            try:
                self.atomTypes[_i] = _aline[0:6]
                self.atomNums[_i] = int(_aline[6:11])
                self.atomNames[_i] = _aline[12:16]
                self.atomAltlocs[_i] = _aline[16:17]
                self.residNames[_i] = _aline[17:20]
                self.chainIds[_i] = _aline[21:22]
                self.residNums[_i] = int(_aline[22:26])
                self.residICodes[_i] = _aline[26:27]
                self.xyz[_i, :] = [float(_aline[30:38]), float(
                    _aline[38:46]), float(_aline[46:54])]
                self.occupancys[_i] = misc.str2float(_aline[56:60], 1.0)  # should be 54:60
                self.tempFactors[_i] = misc.str2float(_aline[61:66], 0.0)  # should be 60:66
                self.elementSyms[_i] = _aline[76:78]
                self.atomCharges[_i] = _aline[78:80]
            except:
                ierror_lines.append(_i)
                continue
            # new residue (Insertion code included)
            if _aline[22:27] != last_line[22:27]:
                self.iAtomNewResid.append(_i)
                # if also a new chain
                if _aline[21] != last_line[21]:
                    self.iAtomNewChain.append(_i)
                    self.iSeqNewChain.append(len(self.iAtomNewResid)-1)
            last_line = _aline

        if len(ierror_lines) != 0:
            logging.error('Some atom/hetatm lines have wrong formats: ')
            print(self.atomLines[ierror_lines])
            self.parse_atom_lines(atom_lines=np.delete(
                self.atomLines, ierror_lines, axis=0))
            return

        # collect chain and residue information
        self.numChains = len(self.iAtomNewChain)
        self.numResids = len(self.iAtomNewResid)
        # self.numResidsPerChain =

        # print('Number of residues: {}'.format(self.numResids))
        # print('Number of chains: {}'.format(self.numChains))

        self.seqChainIds = [str(self.chainIds[_i])
                            for _i in self.iAtomNewChain]

        self.seqResidNames = []
        self.seqResidNums = []
        self.seqResidICodes = []
        self.seqResidIds = []
        for _iseq, _iatm in enumerate(self.iAtomNewResid):
            self.seqResidNames.append(str(self.residNames[_iatm].strip()))
            self.seqResidNums.append(int(self.residNums[_iatm]))
            self.seqResidICodes.append(str(self.residICodes[_iatm]))
            self.seqResidIds.append(
                str(self.seqResidNums[-1])+self.seqResidICodes[-1])

    def assemble_atom_lines(self):
        self.atomLinesAssembled = np.array([
            "{:6s}{:5d} {:^4s}{:1s}{:3s} {:1s}{:4d}{:1s}   {:8.3f}{:8.3f}{:8.3f}{:6.2f}{:6.2f}          {:>2s}{:2s}\n".format(
                self.atomTypes[_i], self.atomNums[_i], self.atomNames[_i], self.atomAltlocs[_i],
                self.residNames[_i], self.chainIds[_i], self.residNums[_i], self.residICodes[_i],
                self.xyz[_i,0], self.xyz[_i, 1], self.xyz[_i, 2], self.occupancys[_i],
                self.tempFactors[_i], self.elementSyms[_i], self.atomCharges[_i])
            for _i in range(0, self.numAtoms)], dtype='U81')
        return self.atomLinesAssembled

    def locate_chain(self, chain=None):
        ichain = []
        if chain is None:  # include all chains if unspecified
            ichain = list(range(0, self.numChains))
        else:
            if isinstance(chain, str):
                chain = [chain]
            for _s in chain:
                ichain += misc.get_list_index(self.seqChainIds, _s)
        return ichain

    def locate_residue(self, chain=None, resnum=None, resicode=None, residids=None):
        """ return the indices to the residues and chain (different lengths!!!).
            residids is a (list of) string of '{resnum}{resicode}
        """
        ichain = self.locate_chain(chain=chain)

        # assemble the list of residIds from resnum and inscode
        if (residids is None) and (resnum is not None):
            if isinstance(resnum, int):
                resnum = [resnum]
            if resicode is None:
                resicode = [' ']*len(resnum)
            elif isinstance(resicode, str):
                resicode = list(resicode)

            residids = [str(_n)+resicode[_i] for _i, _n in enumerate(resnum)]
        elif isinstance(residids, str):
            residids = [residids]

        # collect iseq
        iseq = []
        for _i in ichain:
            # find the residue index range of the chain first
            ires_start = self.iSeqNewChain[_i]
            if (_i + 1) >= self.numChains:
                ires_end = self.numResids
            else:
                ires_end = self.iSeqNewChain[_i+1]

            # find the specific residue indices
            if residids is None:
                iseq += list(range(ires_start, ires_end))
            else:
                for _id in residids:
                    iseq += misc.get_list_index(
                        self.seqResidIds[ires_start:ires_end], _id, offset=ires_start)

        return iseq, ichain

    def locate_neighboring_residue(self, chain=None, resnum=None, resicode=None, delta=1):
        """ return the indices of the neighboring residues and chain """
        ichain = self.locate_chain(chain=chain)
        iseq = []
        if not resicode:
            resicode = " "

        for _i in ichain:
            # find the residue index range of the chain first
            ires_start = self.iSeqNewChain[_i]
            if (_i + 1) >= self.numChains:
                ires_end = self.numResids
            else:
                ires_end = self.iSeqNewChain[_i+1]

            # print(ires_start, ires_end, resnum, resicode, delta )
            for __i in range(0, abs(delta)):
                iseq_neighbor = []
                if resicode != " ":  # check for the same resnum with previous/next insertion code
                    resicode_neighbor = chr(
                        ord(resicode) + (1 if delta > 0 else -1))
                    iseq_neighbor = misc.get_list_index(self.seqResidIds[ires_start:ires_end],
                                                   str(resnum)+resicode_neighbor, offset=ires_start)

                # (this block is not necessary)
                if iseq_neighbor == []:  # check for the previous/next resnum without insertion code
                    resnum_neighbor = resnum + (1 if delta > 0 else -1)
                    iseq_neighbor = misc.get_list_index(self.seqResidIds[ires_start:ires_end],
                                                   '{} '.format(resnum_neighbor), offset=ires_start)

                if iseq_neighbor == []:  # check for the previous/next resum with insertion code
                    resnum_neighbor = resnum + (1 if delta > 0 else -1)
                    iseq_neighbor = misc.get_list_index(self.seqResidNums[ires_start:ires_end],
                                                   resnum_neighbor, offset=ires_start, last=True)

                if iseq_neighbor == []:
                    print('Cannot find neighboring residues')
                    break

                iseq += iseq_neighbor
                resnum = self.seqResidNums[iseq_neighbor[0]]
                resicode = self.seqResidICodes[iseq_neighbor[0]]

        return iseq, ichain

    def locate_atom(self, chain=None, resnum=None, resicode=None, residids=None, atom=None,
                    iseq=None):
        """ return ichain (to seqChainIds), iresid (to seqResidNames), iatom (to xyz) """

        if iseq is None:
            iseq, ichain = self.locate_residue(
                chain=chain, resnum=resnum, resicode=resicode, residids=residids)
        else:
            ichain = 0

        iatom = []
        for _i in iseq:
            iatom_start = self.iAtomNewResid[_i]
            if (_i+1) >= self.numResids:
                iatom_end = self.numAtoms
            else:
                iatom_end = self.iAtomNewResid[_i+1]

            if atom is None:
                iatom += range(iatom_start, iatom_end)
            else:
                if isinstance(atom, str):
                    atom = [atom]
                for _s in atom:
                    iatom += misc.get_list_index(self.atomNames[iatom_start:iatom_end].tolist(
                    ), '{:^4s}'.format(_s.strip()), offset=iatom_start)

        return iatom, iseq, ichain

    def iselect(self, ichain=None, iseq=None, iatom=None):
        """ return a new AtomsData() with the selected atomLines by index """
        assert bool(ichain) + bool(iseq) + bool(iatom) == 1, \
            "Error: one and only one of ichain, iseq, iatom needs to be passed!"

        if iatom is not None:
            atoms_selected = AtomsData()
            atoms_selected.atomLines = self.atomLines[iatom]
            atoms_selected.parse_atom_lines()
        elif iseq is not None:
            if isinstance(iseq, int):
                iseq = [iseq]
            iatom = []
            iatom_newseq = self.iAtomNewResid + [self.numResids]
            for _iseq in iseq:
                iatom += list(range(iatom_newseq[_iseq],
                                    iatom_newseq[_iseq+1]))
            atoms_selected = self.iselect(iatom=iatom)
        elif ichain is not None:
            if isinstance(ichain, int):
                ichain = [ichain]
            iseq = []
            iseq_newchain = self.iSeqNewChain + [self.numResids]
            for _ichain in ichain:
                iseq += list(range(iseq_newchain[_ichain],
                                   iseq_newchain[_ichain+1]))
            atoms_selected = self.iselect(iseq=iseq)
        else:
            atoms_selected = AtomsData()

        return atoms_selected

    def vselect(self, chain=None, resnum=None, atom=None):
        """ return a new AtomsData() with the selected atomLines by value """
        iatom, _iseq, _ichain = self.locate_atom(
            chain=chain, resnum=resnum, atom=atom)
        return self.iselect(iatom=iatom)

    def rename_chains(self, iseq_newchain=[0], chain_ids=None):
        """  """
        new_id_list = []
        for ichain, iseq_start in enumerate(iseq_newchain):
            iatom_start = self.iAtomNewResid[iseq_start]

            if (ichain + 1) == len(iseq_newchain):
                iseq_end = self.numResids - 1
                iatom_end = self.numAtoms - 1
            else:
                iseq_end = iseq_newchain[ichain + 1]
                iatom_end = self.iAtomNewResid[iseq_end] - 1

            if chain_ids is None:
                new_id_list.append(molstru_config.ChainIdList[ichain])
            else:
                new_id_list.append(chain_ids[ichain])

            self.chainIds[iatom_start:iatom_end + 1] = new_id_list[-1]
            logger.info(f'New chain id: {new_id_list[-1]} to residues {iseq_start} to {iseq_end}')

        # need to reassemble and parse (lazy way to do this)
        self.atomLines = self.assemble_atom_lines()
        self.parse_atom_lines()

        return new_id_list

    def renumber_residues(self, startnum=1, resnums=None, resicodes=None, seq=None):
        """  """
        if seq is not None:
            if isinstance(seq, list):
                seq = ''.join(seq)
            myseq = ''.join(self.seqResidNames)
            # one must be the substring of the other
            if len(seq) > len(myseq):
                istart = misc.get_list_index(seq, myseq)
                if istart != []:
                    resnums = list(
                        range(istart[0] + 1, istart[0] + 1 + self.numResids))
            else:
                istart = misc.get_list_index(myseq, seq)
                if istart != []:
                    resnums = list(
                        range(-istart[0] + 1, -istart[0] + 1 + self.numResids))

        if resnums is None:
            resnums = list(range(startnum, startnum+self.numResids))
        else:
            if len(resnums) != self.numResids:
                print('The number of residue numbers passed is not compatible!!!')
                return

        if resicodes is None:
            resicodes = [' ']*self.numResids
        elif isinstance(resicodes, str):
            resicodes = list(resicodes)*self.numResids
        else:
            if len(resicodes) != self.numResids:
                print('The number of residue insertion codes passed is not compatible!!!')
                return

        self.seqResidNums = resnums
        self.seqResidICodes = resicodes
        self.seqResidIds = ['{}{}'.format(resnums[_i], resicodes[_i])
                            for _i in range(0, self.numResids)]

        iAtomNewResid = self.iAtomNewResid + [self.numAtoms]
        for _iseq in range(0, self.numResids):
            self.residNums[iAtomNewResid[_iseq]:iAtomNewResid[_iseq+1]] = resnums[_iseq]
            self.residICodes[iAtomNewResid[_iseq]:iAtomNewResid[_iseq+1]] = resicodes[_iseq]

        self.assemble_atom_lines()
        return

    def align_myself_kabsch(self, fixed_atoms, atom=None, resnum=None, iseq=None, by_resi=True):
        """ The residues in "self" are filtered before aligning to "fixed_atom",
        first by iseq (default is all residues), then filtered by resnum (default is all resnums).

        When comparing "self" and "fixed_atoms", "by_resi" will look for common residue numbers,
        then the "atom" in each residue is used for alignment.
         """

        # convert to list if needed
        args_iseq = [iseq] if isinstance(iseq, int) else iseq
        args_resnum = [resnum] if isinstance(resnum, int) else resnum
        args_atom = [atom] if isinstance(atom, str) else atom

        # filter iseq by resnum
        if args_iseq is None:
            args_iseq = range(0, self.numResids)
        if args_resnum is not None:
            args_iseq = [
                _i for _i in args_iseq if self.seqResidNums[_i] in args_resnum]
        if len(args_iseq) == 0:
            print('check resnum and iseq arguments, no residues found!!!')
            return [], []

        # collect the indices to atoms for both moving and fixed
        iatom_moving, iatom_fixed = [], []
        for _iseq in args_iseq:
            # Find the corresponding iseq_fixed
            if (args_resnum is not None) or by_resi:  # require the same resnum and resicode
                if self.seqResidIds[_iseq] not in fixed_atoms.seqResidIds:
                    continue
                # maybe should check for more than one resnum match
                iseq_fixed = fixed_atoms.seqResidIds.index(
                    self.seqResidIds[_iseq])
            else:
                iseq_fixed = _iseq
                if iseq_fixed >= fixed_atoms.numResids:
                    break

            _iatom_start = self.iAtomNewResid[_iseq]

            # Get atom names for both moving and fixed (not an efficient way...)
            iend_moving = self.iAtomNewResid[_iseq+1] if (
                _iseq+1) < self.numResids else self.numAtoms
            all_atom_moving = self.atomNames[_iatom_start:iend_moving].tolist()

            iend_fixed = fixed_atoms.iAtomNewResid[iseq_fixed+1] \
                if (iseq_fixed + 1) < fixed_atoms.numResids else fixed_atoms.numAtoms
            all_atom_fixed = fixed_atoms.atomNames[fixed_atoms.iAtomNewResid[iseq_fixed]:iend_fixed].tolist(
            )

            # separate the step to find common atoms
            atom_align = all_atom_moving if args_atom is None else args_atom
            for _i, _atom in enumerate(atom_align):
                _atom = '{:^4s}'.format(_atom)

                if (_atom not in all_atom_fixed) or (_atom not in all_atom_moving):
                    continue
                # print('Adding {} of residue {} to alignment'.format(_atom, self.seqResidNums[_iseq]))

                iatom_moving.append(_iatom_start+all_atom_moving.index(_atom))
                iatom_fixed.append(
                    fixed_atoms.iAtomNewResid[iseq_fixed]+all_atom_fixed.index(_atom))

        # all right, now ready to align
        num_atoms = len(iatom_moving)
        if num_atoms == 0:
            print('No common residues/atoms found, returning!!!')
            return np.eye(3), np.zeros((3, 1))

        # get the xyz and get the alignment matrices
        moving_xyz = self.xyz[iatom_moving, :]
        fixed_xyz = fixed_atoms.xyz[iatom_fixed, :]
        print('Aligning a total of {} points...'.format(num_atoms))
        self.rota_mat, self.tran_mat = geom.align_kabsch_get(
            moving_xyz, fixed_xyz)
        aligned_xyz = geom.align_kabsch_apply(
            moving_xyz, rotate=self.rota_mat, translate=self.tran_mat)
        print('RMSD:', np.mean(np.square(aligned_xyz-fixed_xyz))*3)

        return self.rota_mat, self.tran_mat

    def add_missing_atoms(self):

        # new_atom_lines = []

        # iatom_newseq = self.iAtomNewResid + [self.numAtoms]

        # # probably better to hard-code the atomlines for RNA from NDB/x3DNA
        # for _iseq in range(0, self.numResids):
        #     iatom_start = iatom_newseq[_iseq]
        #     iatom_end = iatom_newseq[_iseq+1]

        pass

    def add_missing_residues(self, resinfo=None, chain=None, resname=None, resnum=None, resicode=None):
        """ resinfo is a list of [model(unused), chain, resname, rename, resicode] """

        resnums_by_chain, resnames_by_chain, resicodes_by_chain = self.resinfo_list2dict(
            resinfo=resinfo, chain=chain, resname=resname, resnum=resnum, resicode=resicode)

        # add for one chain at a time
        is_all_missing_residues_added = True
        for _chain_id in resnums_by_chain.keys():
            if _chain_id not in self.seqChainIds:
                # print("Chain: {} not found in the PDB data, skipping...".format(_chain_id))
                continue
            logger.info('Adding residues to chain: {}'.format(_chain_id))
            logger.info('    residue numbers: {}, residue names: {}, residue icodes: {}'.format(
                resnums_by_chain[_chain_id], resnames_by_chain[_chain_id], resicodes_by_chain[_chain_id]))

            # find continuous fragment in the missing residues (should be okay to ignore resicode here)
            i_newfrag = [0] + [_i+1 for _i, _resnum in enumerate(resnums_by_chain[_chain_id][1:])
                               if (_resnum - resnums_by_chain[_chain_id][_i]) > 1 or
                               abs(ord(resicodes_by_chain[_chain_id][_i+1])-ord(resicodes_by_chain[_chain_id][_i]) > 1)]

            i_newfrag.append(len(resnums_by_chain[_chain_id]))
            atoms_frag = None
            for _ifrag in range(0, len(i_newfrag)-1):
                resnums_frag = resnums_by_chain[_chain_id][i_newfrag[_ifrag]:i_newfrag[_ifrag+1]]
                resnames_frag = resnames_by_chain[_chain_id][i_newfrag[_ifrag]:i_newfrag[_ifrag+1]]
                resicodes_frag = resicodes_by_chain[_chain_id][i_newfrag[_ifrag]:i_newfrag[_ifrag+1]]

                print('>>>>>>> fragment #', _ifrag+1, ' chain id: ',
                      _chain_id, 'length: ', len(resnums_frag))
                if len(resnums_frag) > 100:
                    logger.warning('Too many residues to add, skipp!!!')
                    is_all_missing_residues_added = False
                    continue
                print('residue numbers: ', resnums_frag)
                print('residue names:', resnames_frag)
                print('residue icodes:', resicodes_frag)

                # iatom_5p, iseq_5p, ichain_5p = self.locate_atom(chain=_chain_id,
                #     resnum=list(range(resnums_frag[0]-num_overlaps, resnums_frag[0])))
                # iatom_3p, iseq_3p, ichain_3p = self.locate_atom(chain=_chain_id,
                #     resnum=list(range(resnums_frag[-1]+1, resnums_frag[-1]+1+num_overlaps)))

                # the iseq starts from the reference residue to the left (5') or right (3')
                iseq_5p, __ = self.locate_neighboring_residue(
                    chain=_chain_id, resnum=resnums_frag[0], resicode=resicodes_frag[0], delta=-4)
                iseq_3p, __ = self.locate_neighboring_residue(
                    chain=_chain_id, resnum=resnums_frag[-1], resicode=resicodes_frag[-1], delta=4)

                # get rid of non RNA residues
                iseq_5p = [__i for __i in iseq_5p if molstru_config.res2seqcode(
                    self.seqResidNames[__i])[0] != '?']
                iseq_3p = [__i for __i in iseq_3p if molstru_config.res2seqcode(
                    self.seqResidNames[__i])[0] != '?']

                # requires at least 1 residue at either end for doing alignment
                # the returned residues are counted from the reference residue
                iseq_5p = [] if len(iseq_5p) < 1 else iseq_5p if len(iseq_5p) == 1 else [iseq_5p[1], iseq_5p[0]]
                iseq_3p = [] if len(iseq_3p) < 2 else iseq_3p if len(iseq_3p) == 1 else [iseq_3p[0], iseq_3p[1]]

                if len(iseq_5p) == 0 and len(iseq_3p) == 0:
                    logger.warning("Failed to find residues upstream and downstream, cannot add")
                    is_all_missing_residues_added = False
                    continue

                resnames_5p = ''.join(molstru_config.res2seqcode(
                    [self.seqResidNames[_iseq] for _iseq in iseq_5p]))
                resnames_3p = ''.join(molstru_config.res2seqcode(
                    [self.seqResidNames[_iseq] for _iseq in iseq_3p]))

                resnums_5p = [self.seqResidNums[__i] for __i in iseq_5p]
                resnums_3p = [self.seqResidNums[__i] for __i in iseq_3p]

                resicodes_5p = [self.seqResidICodes[__i] for __i in iseq_5p]
                resicodes_3p = [self.seqResidICodes[__i] for __i in iseq_3p]

                resnums_dssr = resnums_5p + resnums_frag + resnums_3p
                resicodes_dssr = resicodes_5p + resicodes_frag + resicodes_3p

                resnames_dssr = resnames_5p + ''.join(resnames_frag) + resnames_3p

                print('Creating new sequence <{}> with dssr...'.format(resnames_dssr))
                debug_dir = 'debug'
                os.makedirs(debug_dir, exist_ok=True)
                dssr_file = os.path.join(debug_dir, '{}_dssr_frag{}.pdb'.format(self.id, _ifrag+1))
                dssr_cmd = '/home/xqiu/programs/bin/x3dna-dssr fiber --rna --seq={} -o={} >/dev/null 2>&1'.format(
                    resnames_dssr, dssr_file)
                print('Run command: {}'.format(dssr_cmd))
                if os.system(dssr_cmd):
                    logging.warning('Some error occurred with running dssr!!!')

                # read in and renumber the residues
                atoms_frag = AtomsData(dssr_file)
                atoms_frag.chainIds[:] = _chain_id
                atoms_frag.renumber_residues(
                    resnums=resnums_dssr, resicodes=resicodes_dssr)
                atoms_frag.atomLines = atoms_frag.assemble_atom_lines()
                atoms_frag.write_pdb(dssr_file)

                # align them
                # just to avoid vscode complaints
                xyz_aligned_5p, xyz_aligned_3p = atoms_frag.xyz, atoms_frag.xyz

                if len(iseq_5p) > 0:
                    rota_5p, tran_5p = atoms_frag.align_myself_kabsch(
                                        self, resnum=resnums_5p, by_resi=True)
                    print(rota_5p, tran_5p, atoms_frag.xyz.shape)
                    xyz_aligned_5p = geom.align_kabsch_apply(
                                        atoms_frag.xyz, rotate=rota_5p, translate=tran_5p)
                if len(iseq_3p) > 0:
                    rota_3p, tran_3p = atoms_frag.align_myself_kabsch(
                                        self, resnum=resnums_3p, by_resi=True)
                    xyz_aligned_3p = geom.align_kabsch_apply(
                                        atoms_frag.xyz, rotate=rota_3p, translate=tran_3p)

                # combine the aligned segements if needed
                iatom_frag_start = atoms_frag.iAtomNewResid[len(iseq_5p)]
                iatom_frag_end = atoms_frag.iAtomNewResid[len(iseq_5p)+len(resnums_frag)] \
                    if len(iseq_5p)+len(resnums_frag) < atoms_frag.numResids else atoms_frag.numAtoms
                iatom_5p_end, iatom_3p_start = -1, self.numAtoms

                if len(iseq_5p) > 0 and len(iseq_3p) > 0:
                    # just a linear mixing here, 11-1--->0-00
                    mixing_ratio = np.concatenate((
                                np.ones((len(iseq_5p),)),
                                np.array([0.5]) if len(resnums_frag) == 1 else
                                        np.linspace(1, 0, num=len(resnums_frag)),
                                np.zeros((len(iseq_3p),))), axis=0)
                    iatom_newseq_frag = atoms_frag.iAtomNewResid + \
                        [atoms_frag.numAtoms]
                    for _iseq, _iatom_start in enumerate(iatom_newseq_frag[0:-1]):
                        iatom_end = iatom_newseq_frag[_iseq+1]
                        atoms_frag.xyz[_iatom_start:iatom_end, :] = \
                            xyz_aligned_5p[_iatom_start:iatom_end, :]*mixing_ratio[_iseq] + \
                            xyz_aligned_3p[_iatom_start:iatom_end,
                                           :]*(1-mixing_ratio[_iseq])
                    iatom_5p_end = self.iAtomNewResid[iseq_5p[-1]+1]-1
                    iatom_3p_start = self.iAtomNewResid[iseq_3p[0]]

                elif len(iseq_5p) > 0 and len(iseq_3p) == 0:
                    atoms_frag.xyz = xyz_aligned_5p
                    iatom_5p_end = self.numAtoms-1 if iseq_5p[-1] + 1 == self.numResids \
                        else self.iAtomNewResid[iseq_5p[-1]+1]-1
                    iatom_3p_start = iatom_5p_end + 1  # self.numAtoms
                elif len(iseq_5p) == 0 and len(iseq_3p) > 0:
                    atoms_frag.xyz = xyz_aligned_3p
                    iatom_3p_start = self.iAtomNewResid[iseq_3p[0]]
                    iatom_5p_end = iatom_3p_start - 1
                else:
                    iatom_5p_end = self.numAtoms-1
                    iatom_3p_start = self.numAtoms
                # merge
                atoms_frag.atomLines = atoms_frag.assemble_atom_lines()
                atoms_frag.write_pdb(dssr_file.replace('.pdb', '_aligned.pdb'))
                self.atomLines = np.concatenate((
                            self.atomLines[0:iatom_5p_end+1],
                            atoms_frag.atomLines[iatom_frag_start:iatom_frag_end],
                            self.atomLines[iatom_3p_start:]),
                            axis=0)
                self.parse_atom_lines()

        return is_all_missing_residues_added


    def resinfo_list2dict(self, resinfo=None, chain=None, resnum=None, resname=None, resicode=None):
        """ convert residue info in list format to three dicts grouped by chain """

        assert (resinfo is not None) or (len(resnum) == len(resname)), \
            'Error: different lengths of chain, resnums, or resnames!!!'

        if resinfo is not None:
            chain, resnum, resname, resicode = [], [], [], []
            for _res in resinfo:
                chain.append(_res[1])
                resname.append(_res[2])
                resnum.append(_res[3])
                resicode.append(_res[4])

        # handle input arguments
        argin_resnum = [resnum] if isinstance(resnum, int) else resnum
        argin_resname = [resname] * \
            len(resnum) if isinstance(resname, str) else resname
        # replace "N" to "U"
        argin_resname = [_s if _s in ['A', 'U', 'G', 'C']
                         else 'U' for _s in argin_resname]
        argin_resname = molstru_config.res2seqcode(argin_resname, undef_code='U')

        if not resicode:
            argin_resicode = [' ']*len(resnum)
        elif isinstance(resicode, str):
            argin_resicode = [resicode]*len(resnum)
        else:
            argin_resicode = resicode

        argin_chain = []
        if chain is None:  # assume the first chain
            argin_chain = [str(self.chainIds[0])]*len(resnum)
        elif isinstance(chain, str):
            argin_chain = [chain]*len(resnum)
        elif len(chain) != len(resnum):
            logger.warning('Chain length must be the same as resnum, returning!!!')
            return
        else:
            argin_chain = chain

        # group resnums, resnames, and resicodes by chain id
        resnums_by_chain, resnames_by_chain, resicodes_by_chain = dict(), dict(), dict()
        for _i, _chain_id in enumerate(argin_chain):
            if _chain_id not in resnums_by_chain:
                resnums_by_chain[_chain_id] = []
                resnames_by_chain[_chain_id] = []
                resicodes_by_chain[_chain_id] = []
            resnums_by_chain[_chain_id].append(argin_resnum[_i])
            resnames_by_chain[_chain_id].append(argin_resname[_i])
            resicodes_by_chain[_chain_id].append(argin_resicode[_i])

        return resnums_by_chain, resnames_by_chain, resicodes_by_chain


    def mutate_residues(self, resinfo=None, chain=None, resnum=None, resname=None, resicode=None):
        """ resinfo is a list of [model(unused), chain, resname, rename, resicode]
            resinfo overwrites other arguments!!!
            checks if the new residue (given by resnum and resname) is the same as existing, do nothing if so.
        """
        resnums_by_chain, resnames_by_chain, resicodes_by_chain = self.resinfo_list2dict(
            resinfo=resinfo, chain=chain, resname=resname, resnum=resnum, resicode=resicode)

        iatom_start = 0
        new_atom_lines = [] # this stores all atoms lines for the new structure!
        for _chain_id in resnums_by_chain.keys():
            if _chain_id not in self.seqChainIds:
                logger.error("Chain: {} not found in the PDB, skipping...".format(_chain_id))
                continue

            logger.info('Mutating residues for chain: {}'.format(_chain_id))
            logger.info('    residue numbers: {}, residue names: {}, residue icodes: {}'.format(
                resnums_by_chain[_chain_id], resnames_by_chain[_chain_id], resicodes_by_chain[_chain_id]))

            for _i, _resnum in enumerate(resnums_by_chain[_chain_id]):
                ires, ichain = self.locate_residue(chain=_chain_id, resnum=_resnum, resicode=resicodes_by_chain[_chain_id][_i])

                if len(ires) == 0:
                    logger.error(f'Residue num: {_resnum} not found in PDB, skip!!!')
                    continue
                elif len(ires) > 1:
                    logger.error(f'More than one residues with resnum: {_resnum} found in PDB, skip!!!')
                    continue
                else:
                    ires = ires[0]
                    ichain = ichain[0]

                if resnames_by_chain[_chain_id][_i] == self.seqResidNames[ires]:
                    logger.info(f'Same residue names for resnum: {_resnum}, resname: {self.seqResidNames[ires]}, skip!!!')
                    continue

                # add the atomlines from iatom_start to the beginning of current base
                if self.iAtomNewResid[ires] > iatom_start:
                    new_atom_lines.append(self.atomLines[iatom_start:self.iAtomNewResid[ires]])
                iatom_start = self.iAtomNewResid[ires+1] if (ires + 1) < self.numResids else self.numAtoms

                atoms_newbase = AtomsData(molstru_config.RNA_AtomLines[resnames_by_chain[_chain_id][_i]])
                atoms_newbase.chainIds[:] = _chain_id
                atoms_newbase.renumber_residues(resnums=[_resnum], resicodes=[resicodes_by_chain[_chain_id][_i]])
                atoms_newbase.atomLines = atoms_newbase.assemble_atom_lines()
                # atoms
                rota_mat, tran_mat = atoms_newbase.align_myself_kabsch(self, resnum=_resnum, by_resi=True)
                print(rota_mat, tran_mat, atoms_newbase.xyz.shape)
                atoms_newbase.xyz = geom.align_kabsch_apply(atoms_newbase.xyz, rotate=rota_mat, translate=tran_mat)
                new_atom_lines.append(atoms_newbase.assemble_atom_lines())

        if iatom_start < self.numAtoms:
            new_atom_lines.append(self.atomLines[iatom_start:])
        self.parse_atom_lines(np.concatenate(new_atom_lines, axis=0))


    def join_atoms(self, atoms2, by_resi=True):
        """ A general routine to join any two structures """

        # chains_list = list(set(self.seqChainIds + atoms2.seqChainIds)).sort()
        # for _chain in chains_list:
        #     ichain1 = self.seqChainIds.index(_chain)
        #     ichain2 = atoms2.seqChainIds.index(_chain)
        #     iseq1 = [self.iSeqNewChain[ichain1],
        #              self.iSeqNewChain[ichain1+1] if ichain1+1 < self.numChains else self.numResids]
        #     iseq2 = [atoms2.iSeqNewChain[ichain2],
        #              atoms2.iSeqNewChain[ichain2+1] if ichain2+1 < atoms2.numChains else atoms2.numResids]

        # resnums1 = self.seqResidNums[iseq1[0]:iseq1[1]]
        # resnums2 = atoms2.seqResidNums[iseq2[0]:iseq2[1]]

        # how
        # resnums_left = self.loc
        # # get indices to the fragement only
        # iseq_start = len(iseq_5p) ; iseq_end = len(iseq_5p) + len(resnums_frag)
        # iatom_start = atoms_frag.iAtomNewResid[iseq_start]
        # iatom_end = atoms_frag.iAtomNewResid[iseq_end] if iseq_end < atoms_frag.numResids else self.numAtoms

        # iatom_newseq_frag = atoms_frag.iAtomNewResid[iseq_start:iseq_end]
        # if iatom_newseq_frag[0] != 0 :
        #     iatom_newseq_frag = [_iatom - iatom_newseq_frag[0] for _iatom in atoms_frag.iAtomNewResid]

        pass

    def calc_res2res_distance(self, atom=None, neighbor_only=True, end_padding=0):
        """  """

        # calculate the distance from its prevoius residue
        if neighbor_only == True:
            self.dist_res2res_cen2cen = np.zeros((self.numResids,), dtype=float)
            self.dist_res2res_closest = np.zeros((self.numResids,), dtype=float)
            # undefied distance for the first residue
            self.dist_res2res_cen2cen[0] = end_padding
            self.dist_res2res_closest[0] = end_padding

            iAtomNewResid = self.iAtomNewResid + [self.numAtoms]
            # compute the first residue as the previous residue
            xyz_pre = self.xyz[iAtomNewResid[0]:iAtomNewResid[1]]
            xyz_cen_pre = np.mean(xyz_pre, axis=0)
            # start from 1!!!
            for _iseq, _iatm in enumerate(self.iAtomNewResid[1:]):
                # _iatm_pre = iAtomNewResid[_iseq]    # first atom for the previous residue
                # _iatm_nex = iAtomNewResid[_iseq+2]  # first atom for the next residue
                # xyz_pre = self.xyz[_iatm_pre:_iatm] # xyzs for all atoms in the previous residue
                xyz = self.xyz[_iatm:iAtomNewResid[_iseq+2]]
                xyz_cen = np.mean(xyz, axis=0)

                self.dist_res2res_cen2cen[_iseq + 1] = np.linalg.norm(xyz_cen_pre - xyz_cen)

                atm2atm_dist = np.linalg.norm(xyz_pre[:, None, :] - xyz[None, :, :], axis=-1)
                self.dist_res2res_closest[_iseq+1] = np.min(atm2atm_dist)
                # the current becomes the previous
                xyz_pre, xyz_cen_pre = xyz, xyz_cen

        elif atom is not None:
            atom = '{:^4s}'.format(atom)
            self.dist_res2res_atom = np.zeros(
                (self.numResids, self.numResids), dtype=float)
            # get the atom indices to
            iatm_selected = -np.ones((self.numResids,), dtype=int)
            atm_selected = np.array(['']*self.numResids, dtype='U4')

            iAtomNewResid = self.iAtomNewResid + [self.numAtoms]
            for _iseq, _iatm in enumerate(self.iAtomNewResid):
                _iatm_selected = np.where(self.atomNames[_iatm:iAtomNewResid[_iseq+1]] == atom)[0]
                if len(_iatm_selected) == 0:
                    iatm_selected[_iseq] = _iatm  # just choose the first one
                else:
                    iatm_selected[_iseq] = _iatm + _iatm_selected[0]
                atm_selected[_iseq] = self.atomNames[iatm_selected[_iseq]]

            xyz = self.xyz[iatm_selected, :]
            # wasted 50% of calculation power (scipy.spatial.pdist may work better)
            self.dist_res2res_byatom = np.linalg.norm(
                        xyz[:, np.newaxis, :] - xyz[np.newaxis, :, :], axis=-1)
            self.dist_res2res_iatm = iatm_selected
            self.dist_res2res_atmname = atm_selected

    def remove_residues(self, iseq=None):
        if iseq is None: return
        logger.info(f'Removing the following residues: {iseq} ...')
        iatom, _, _ = self.locate_atom(iseq=iseq)
        self.atomLines = np.delete(self.atomLines, iatom, axis=0)
        self.parse_atom_lines()

    def remove_dangling_residues(self, max_length=2, dist_cutoff=7):
        """ Simply check the closest residue to residue distances
        max_length - the maximum length of fragments to remove
        dist_cuoff - the maximum distance between connected residues
        """
        if not hasattr(self, 'dist_res2res_closest'):
            self.calc_res2res_distance(neighbor_only=True)

        idx_breaks = np.where(self.dist_res2res_closest >= dist_cutoff)[0]
        if len(idx_breaks) == 0:
            print('No neighboring residues with distance >= {}'.format(dist_cutoff))
            return []
        else:
            print('Found {} residues with distances >={}'.format(
                len(idx_breaks), dist_cutoff))

        # Find the indices
        idx2rmv = []
        idx_breaks = np.append(idx_breaks, [self.numResids], axis=0)
        for _i, _iseq in enumerate(idx_breaks[0:-1]):
            # the length between two breaks must be smaller than max_length
            if (idx_breaks[_i+1] - _iseq) <= max_length:
                idx2rmv += list(range(_iseq, idx_breaks[_i+1]))

        # remove them
        if len(idx2rmv) == self.numResids:
            logging.warning("All residues are dangling for {}".format(self.id))
            idx2rmv = []

        if idx2rmv != []:
            print('Removing the following residues: ', idx2rmv)
            iatom, _, _ = self.locate_atom(iseq=idx2rmv)
            self.atomLines = np.delete(self.atomLines, iatom, axis=0)
            self.parse_atom_lines()

        return idx2rmv

    def write_casp_pdb(self, pdbfile, fdir='',
            pfrmat='TS',
            target='R1117',
            author='9676-0265-9751',
            method='SABARNA: Sequence Alignment Based Assembly of RNA',
            model=1,
            parent='6e1t_A',    # N/A for ab initio prediction
            score=None,
            qscore=None,
            ):
        """ incomplete as one needs to deal with TER, 
            decided to use shell script
        """
        headers = [
            f'PFRMAT    {pfrmat}',
            f'TARGET    {target}',
            f'AUTHOR    {author}',
            f'METHOD    {method}',
            f'MODEL     {model}',
            f'PARENT    {parent}',
            ]
        if score is not None:
            headers.append(f'SCORE  {score}')
        if qscore is not None:
            headers.append(f'QSCORE {qscore}')

        footers = [
            f'END',
            ]
        
        with open(os.path.join(fdir, pdbfile), 'w') as hfile:
            hfile.writelines('\n'.join(headers))
            for ichain in range(self.numChains):
                hfile.writelines(self.atomLines)
            hfile.writelines('\n'.join(headers))

    def write_pdb(self, pdbfile, fdir=''):
        with open(os.path.join(fdir, pdbfile), 'w') as hfile:
            hfile.writelines(self.atomLines)

    def write_fasta(self, fastafile, fdir=''):
        with open(os.path.join(fdir, fastafile), 'w') as hfile:
            hfile.writelines(
                ['>{}\n'.format(self.id), ''.join(self.seqResidNames)])

    def write_dist_matrix(self, fname, suffix='.dist', fdir=''):
        dist_file = os.path.join(fdir, fname + suffix)
        # sav_mat = self.dist_res2res_byatom
        # np.savetxt(dist_file, sav_mat, fmt='%8.3f')
        # with open(dist_file, 'w') as ofile :
        # ofile.write('         '.join(self.seqResidNames)+'\n')
        # sav_mat = np.array2string(self.dist_res2res_byatom, max_line_width=np.Inf,
        # formatter=dict(float_kind=lambda x: "%8.3f" % x))
        # sav_mat = sav_mat[1:-1].split('\n')
        # for _i in range(self.numResids) :
        # sav_mat[_i] = ' ' + self.seqResidNames + sav_mat[_i][2:-1]
        # sav_mat[_i][-1] = '\n'
        # sav_mat[-1][-2] = ''
        sav_mat = self.dist_res2res_byatom.astype('U8')
        col_label = np.array([''] + self.seqResidNames).reshape((self.numResids+1, 1))
        np.savetxt(dist_file, np.hstack((
                    col_label,
                    np.vstack((np.array(self.seqResidNames), sav_mat))
        )), fmt='%9s')

        # np.zeros((self.numResids[0]+1, self.numResids[0]+1), dtype=int)
        # sav_mat[1:,1:] = self.bpmat.astype()
        # sav_mat[0,:] = np.linspace(0,self.numResids[0], num=self.numResids[0]+1, dtype=int)
        # sav_mat[:,0] = np.linspace(0,self.numResids[0], num=self.numResids[0]+1, dtype=int)
        # sav_mat = sav_mat.astype(str)
        # sav_mat[0,0] = ''


def test_adding_residues():
    atoms = AtomsData('debug/2FK6_1_R_patched3.pdb')
    atoms.calc_res2res_distance()

    if 1 == 0:
        atoms = AtomsData('debug/2FK6_1_R.pdb')
        atoms.add_missing_residues(chain='R', resnum=list(range(74, 80)), resname=[
                                   'U', 'A', 'A', 'A', 'U', 'G'], resicode=[' ']*6)
        atoms.write_pdb('debug/2FK6_1_R_patched1.pdb')
        atoms.add_missing_residues(chain='R', resnum=list(
            range(-5, 1)), resname=['U', 'A', 'A', 'A', 'U', 'G'], resicode=[' ']*6)
        atoms.write_pdb('debug/2FK6_1_R_patched2.pdb')
        atoms.add_missing_residues(chain='R', resnum=list(range(26, 46)),
                                   resname=['A', 'C', 'U', 'U', 'C', 'C', 'A', 'U', 'G',
                                            'G', 'U', 'A', 'A', 'G', 'G', 'A', 'A', 'G', 'A', 'G'],
                                   resicode=[' ']*20)
        atoms.write_pdb('debug/2FK6_1_R_patched3.pdb')

# %%
if __name__ == "__main__":
    local_vars = locals()
    for name, func in local_vars.items():
        if name.startswith('DEBUG_') and callable(func): # and value.__module__ == __name__
            func()
