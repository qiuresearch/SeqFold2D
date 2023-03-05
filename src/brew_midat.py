#!/usr/bin/env python
import os
import sys
import json
import functools
import itertools
from collections import Counter, namedtuple
from pathlib import Path
from sklearn.model_selection import train_test_split
import math
import pandas as pd
import numpy as np
from tqdm import tqdm
import logging
logger = logging.getLogger(__name__)

# homebrew
import molstru
import misc
import gwio
from molstru import seq_fuzzy_match, reslet_lookup


def get_midat(data_names,
              data_dir=None,
              return_save_prefix=False,
              info=False,
              header='infer',
              **kwargs):
    """ a flexible way to read midata  """

    save_prefix = None

    if isinstance(data_names, pd.DataFrame) or isinstance(data_names, pd.Series):
        df = data_names
    elif isinstance(data_names, dict):
        df = pd.DataFrame(data_names, copy=False)
    else: # file names
        if type(data_names) not in (list, tuple):
            data_names = [data_names]

        if data_dir is None:
            data_names = [Path(_f) for _f in data_names]
        else:
            data_names = [Path(data_dir) / _f for _f in data_names]

        if data_names[0].suffix.endswith(('.fasta', '.fa')):
            df = molstru.SeqsData(data_names, fmt='fasta').to_df(
                    seq=True, ct=False, has=False, res_count=True)
        else:
            df = gwio.read_dataframes(data_names, fmt='infer', return_file=False,
                    ignore_index=True, concat=True, header=header)

        save_prefix = '-'.join([_f.stem for _f in data_names])

    logger.debug(('' if save_prefix is None else f'File: {save_prefix}, ') \
                 + "INVALID or NONEXISTING!!!" if df is None else f'midat shape: {df.shape}')
    if info:
        logger.info(df.info())

    if return_save_prefix:
        return df, save_prefix
    else:
        return df


def split_seqs(data_names,
               seq_fmt='fasta',
               num_chunks=10,
               num_seqs=None,
               **kwargs):
    """ split sequences into multiple files (much slower than fasta_split.sh) """

    if isinstance(data_names, str): data_names = [data_names]

    save_prefix = Path(data_names[0]).stem

    seqs_obj = molstru.SeqsData()
    for data_name in data_names:
        seqs_obj.parse_files(data_name, fmt=seq_fmt)

    if num_seqs is None:
        num_seqs = math.ceil(len(seqs_obj.id) / num_chunks)

    for ifile in range(0, len(seqs_obj.id), num_seqs):
        save_path = f'{save_prefix}_{ifile+1}.fasta'
        logger.info(f'Saving fasta file: {save_path}...')
        with open(save_path, 'w') as iofile:
            iofile.writelines(
                seqs_obj.get_fasta_lines(list(range(ifile, ifile + num_seqs)), line_break=True)
                )


def count_all_residues(
        data_names,
        recap=misc.Struct(),
        prefix='',
        show=True,
        tqdm_disable=False,
        **kwargs):
    """ count the residue names in ALL rows and report statistics """

    df = get_midat(data_names)

    res_count = Counter()
    for seq in df.seq:
        res_count.update(Counter(seq))

    total_count = float(sum(res_count.values()))
    for key, val in res_count.most_common():
        recap.update({f'{prefix}num_{key}': val})
        recap.update({f'{prefix}pct_{key}': val / total_count})

    recap.update({f'{prefix}total': total_count})

    if show: print(gwio.json_str(recap.__dict__))


def count_ct_bps(
        data_names,
        recap=misc.Struct(),
        prefix='',
        show=True,
        plot=False,
        tqdm_disable=False,
        **kwargs):
    """ count the paired bases in pkldata and report statistics """

    df = get_midat(data_names)

    bp_counter = Counter()
    for idx, seq_ds in df.iterrows():
        bp_counter += molstru.count_ct_bptypes(seq_ds.ct, seq_ds.seq, return_tuple=False)

    total_count = float(sum(bp_counter.values()))
    for key, val in bp_counter.most_common():
        recap.update({f'{prefix}num_{key}': val})
        recap.update({f'{prefix}pct_{key}': val / total_count})

    recap.update({f'{prefix}total': total_count})

    if show: print(gwio.json_str(recap.__dict__))
    if plot:
        pass


def count_ct_stems(data_names,
                   recap=misc.Struct(),
                   recap_prefix='',
                   show=False,
                   warn_single=False,      # whether to warn stem length of 1.0
                   tqdm_disable=False,
                   **kwargs):
    """ count the length of stems and delta_ij in ct data and report statistics """

    df = get_midat(data_names)

    stem_counter = Counter()
    deltaij_counter = Counter()

    for idx, seq_ds in df.iterrows():
        ct = seq_ds.ct
        ct_len = len(seq_ds.ct)

        if ct_len == 0:
            logger.warning(f'No ct found for {seq_ds.file}')
        elif ct_len == 1:
            logger.warning(f'Only one ct found for {seq_ds.file}')
            print(seq_ds.ct)
            stem_counter.update([1]) # length of 1
            deltaij_counter.upate(seq_ds.ct[1] - seq_ds.ct[0])
        else:
            contig_breaks = np.nonzero(np.logical_or(
                np.diff(ct.sum(axis=1), prepend=0, append=0),   # i+j not constant
                np.diff(ct[:, 0], prepend=-2, append=-2) != 1)  # i not continuous
                )[0]
            stem_lengths = contig_breaks[1:] - contig_breaks[0:-1]

            if warn_single:
                idx_singles = np.nonzero(stem_lengths == 1)[0]
                if len(idx_singles):
                    logger.warning(f'Found stem length of 1 for file: {seq_ds.file}')
                    for _i in idx_singles:
                        print(seq_ds.ct[contig_breaks[_i]-1:contig_breaks[_i]+2])

            stem_counter.update(stem_lengths)
            deltaij_counter.update(ct[:,1] - ct[:,0])

            # stem_counter.update(np.char.add(stem_lengths.astype(str), 'S'))

    total_count = float(sum(stem_counter.values()))
    for key, val in stem_counter.most_common():
        recap.update({f'{recap_prefix}num_stem_{key}': val})
        recap.update({f'{recap_prefix}pct_stem_{key}': val / total_count})

    total_count = float(sum(deltaij_counter.values()))
    for key, val in deltaij_counter.most_common():
        recap.update({f'{recap_prefix}num_deltaij_{key}': val})
        recap.update({f'{recap_prefix}pct_deltaij_{key}': val / total_count})
    recap.update({f'{recap_prefix}total': total_count})

    if show: print(gwio.json_str(recap.__dict__))


def count_pseudo_knots(
        data_names,
        recap=misc.Struct(),
        prefix='',
        show=False,
        tqdm_disable=False,
        **kwargs):
    """ count the number of pseudo knots in pkldata and report statistics """

    df = get_midat(data_names)

    pknot_counter = Counter()
    # ct_contig = np.full(df['len'].max() // 2, True)

    for idx, seq_ds in df.iterrows():
        ct_len = len(seq_ds.ct)

        if ct_len < 2:
            logger.warning(f'Only {ct_len} ct found for {seq_ds.file}')
            pknot_counter.update(['0K'])
            continue

        num_pknots = molstru.count_pseudoknot(seq_ds.ct)
        if num_pknots > 10:
            logger.info(f'{num_pknots} pseudo knots found for {seq_ds.file}')

        pknot_counter.update([f'{num_pknots}K'])

    total_count = float(sum(pknot_counter.values()))
    for key, val in pknot_counter.most_common():
        recap.update({f'{prefix}num_{key}': val})
        recap.update({f'{prefix}pct_{key}': val / total_count})

    recap.update({f'{prefix}total': total_count})

    if show: print(gwio.json_str(recap.__dict__))


def save_individual_files(
            data_names,
            rows=None,            # the list of rows to save, default: all
            save_dir=None,        # default is the stem of the data file
            save_prefix=None,     # add a prefix to each individual file name
            save_genre=['seq'],   # 'bpseq', 'ppm' 'ct', 'ctmat' 'ppm'
            named_after='file',   # which colum to use for individual file names
            tqdm_disable=False,
            **kwargs):
    """ save the sequences into a lumpsum or individual fasta files """

    df, auto_save_prefix = get_midat(data_names, return_save_prefix=True)
    if df is None or len(df) == 0:
        logger.critical(f'Nonexisting or empty files, exiting!!!')
        return

    save_dir = Path(auto_save_prefix) if save_dir is None else Path(save_dir)

    if save_dir.is_file():
        logger.error(f'{save_dir} already exists as a file, please change!')
        return
    save_dir.mkdir(parents=True, exist_ok=True)

    save_prefix = '' if save_prefix is None else save_prefix

    if rows is None:
        rows = np.arange(df.shape[0])
    elif not hasattr(rows, '__len__'):
        rows = [rows]

    if named_after not in df.columns:
        logger.critical(f'dataframe does not have named_by col: {named_after} !!!')
        named_after = 'id'

    logger.info(f'Saving individual files in: {save_dir} ...')
    logger.info(f'Number of sequences to save: {len(rows)}')

    save_dir = save_dir.as_posix()
    save_seq = 'seq' in save_genre and 'seq' in df.columns
    save_bpseq = 'bpseq' in save_genre and ('ct' in df.columns or 'ctmat' in df.columns)
    do_ct2bpseq = 'ct' in df.columns
    do_ctmat2bpseq = 'ctmat' in df.columns
    save_ct = 'ct' in save_genre and 'ct' in df.columns
    save_ctmat = 'ctmat' in save_genre and ('ctmat' in df.columns or 'ct' in df.columns)
    do_ct2ctmat = 'ctmat' not in df.columns
    save_ppmat = 'ppmat' in save_genre and 'ppmat' in df.columns

    if save_ct: logger.critical(f'Saving in ct format is not yet implemented!')

    for i in tqdm(rows, mininterval=1, desc=save_dir, disable=tqdm_disable):
        src_name = Path(str(df.iloc[i][named_after])).stem
        save_path = os.path.join(save_dir, f'{save_prefix}{src_name}')

        if save_seq:
            with open(save_path + '.fasta', 'w') as iofile:
                if df.iloc[i]["id"]:
                    iofile.writelines(f'>{df.iloc[i]["id"]}\n{df.iloc[i]["seq"]}\n')
                else:
                    iofile.writelines(f'>{df.iloc[i]["idx"]:d}\n{df.iloc[i]["seq"]}\n')

        if save_bpseq:
            if do_ctmat2bpseq:
                bpseq_lines = molstru.ctmat2bpseq_lines(
                    df.iloc[i]['ctmat'], df.iloc[i]['seq'], id=df.iloc[i]['id'], return_list=True)
            elif do_ct2bpseq:
                bpseq_lines = molstru.ct2bpseq_lines(
                    df.iloc[i]['ct'], df.iloc[i]['seq'], id=df.iloc[i]['id'], return_list=True)
            else:
                logger.critical(f'No ct and ctmat in dataframe, cannot save bpseq files!!!')

            with open(save_path + '.bpseq', 'w') as iofile:
                iofile.writelines('\n'.join(bpseq_lines))

        if save_ct: # and len(df.iloc[i]['ct']) > 1:
            if len(df.iloc[i]['seq']) < df.iloc[i]['ct'].max():
                logger.error(f'Seq len is shorter than ct mat resnum: {df.iloc[i]["id"]}, not saved!!!')
                continue
            # ct_mat = molstru.ct2ctmat(df.iloc[i]['ct'], len(df.iloc[i]['seq']))
            # # print(f'file: {seqsdata.file[i]} appears corrupted, please check!')
            # np.savetxt(save_path + '.ct', ct_mat, fmt='%1i')

        if save_ctmat:
            if do_ct2ctmat:
                ctmat = molstru.ct2ctmat(df.iloc[i]['ct'], len(df.iloc[i]['seq']))
            else:
                ctmat = df.iloc[i]['ctmat']
            np.savetxt(save_path + '.ctmat', ctmat, fmt='%1i')

        if save_ppmat:
            np.savetxt(save_path + '.ppmat', df.iloc[i]['ppmat'], fmt='%0.8f')


def save_lumpsum_files(
        data_names,
        save_dir='./',
        save_prefix=None,
        save_csv=False,
        save_pkl=False,
        save_fasta=False,
        save_unknown=False,
        save_duplicate=False,
        save_conflict=False,
        csv_exclude=['seq', 'ct', 'tangle'],
        pkl_exclude=None,
        tqdm_disable=False,
        **kwargs):
    """ save all parts of midata as processed in database_* functions  """

        # pkl_include=['idx', 'file', 'db', 'moltype',
        #     'id', 'len', 'lenCT', 'seq', 'ct',
        #     'dataset', 'numPKnots',
        #     'resNames', 'resNameCounts', 'bpTypes', 'bpTypeCounts',
        #     'bpNums', 'bpNumCounts', 'stemLens', 'stemLenCounts',
        #     'deltaijs', 'deltaijCounts',],       # NO LONGER USED!!!!!!!!!!!!!!!
    df, auto_save_prefix = get_midat(data_names, return_save_prefix=True)
    save_prefix = misc.get_1st_value([save_prefix, auto_save_prefix, 'autosave'])

    save_dir = Path.cwd() if save_dir is None else Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    # determine dataset fields if not yet set
    # if 'save2lib' in df.columns:
    #     idx_save2lib = df.index[df.save2lib]
    # else:
    #     idx_save2lib = df.index

    # if split_data:
    #     dataset_names = split_names,
    #     idx_set0, idx_set1 = train_test_split(idx_save2lib, train_size=split_size, shuffle=False,
    #                         random_state=20151029)
    #     df.loc[idx_set0, 'dataset'] = dataset_names[0]
    #     df.loc[idx_set1, 'dataset'] = dataset_names[1]
    # else:
    #     dataset_names = [save_name]
    #     df.loc[idx_save2lib, 'dataset'] = dataset_names[0]

    # save csv
    if save_csv:
        csv_columns = list(filter(lambda _s: _s not in csv_exclude, df.columns.to_list()))
        csv_kwargs = dict(index=False, float_format='%8.6f', columns=csv_columns)
        csv_file = save_dir / (save_prefix + '.csv')
        logger.info(f'Storing csv: {csv_file} of shape: {df.shape}')
        df.to_csv(csv_file, **csv_kwargs)

    # save sequences with unknown residues
    if save_unknown:
        if 'unknownSeq' not in df.columns:
            logger.critical(f'save_unknown requires column: unknownSeq, but not found!!!')
        else:
            idx_unknown_seqs = df.index[df.unknownSeq]
            num_unknown_seqs = len(idx_unknown_seqs)
            logger.info(f'Number of sequences with unknown residues: {num_unknown_seqs}')
            if num_unknown_seqs > 0:
                unk_file = save_dir / (save_prefix + '_unknown')
                df_unk = df.loc[idx_unknown_seqs]
                logger.info(f'Storing unknown sequences as: {unk_file} of shape: {df_unk.shape}')
                if save_csv: df_unk.to_csv(unk_file.with_suffix('.csv'), **csv_kwargs)
                if save_pkl: df_unk.to_pickle(unk_file.with_suffix('.pkl'))
                # if save_lib: save_lib_files(df, save_dir=dir_tmp)

    # save conflict
    if save_conflict:
        if 'conflictVal' not in df.columns and 'conflictSeq' not in df.columns:
            logger.critical(f'save_conflict requires column: conflictSeq or conflictVal, but not found!!!')
        else:
            idx_conflicts = None
            if 'conflictVal' in df.columns:
                idx_conflicts = df.conflictVal
            if 'conflictSeq' in df.columns:
                idx_conflicts = df.conflictSeq if idx_conflicts is None else (idx_conflicts | df.conflictSeq)

            idx_conflicts = df.index[idx_conflicts]
            num_conflict_seqs = len(idx_conflicts)
            logger.info(f'Number of conflict sequences: {num_conflict_seqs}')
            if num_conflict_seqs > 0:
                unk_file = save_dir / (save_prefix + '_conflict')
                df_tmp = df.loc[idx_conflicts]
                logger.info(f'Storing conflict sequences as: {unk_file} of shape: {df_tmp.shape}')
                if save_csv: df_tmp.to_csv(unk_file.with_suffix('.csv'), **csv_kwargs)
                if save_pkl: df_tmp.to_pickle(unk_file.with_suffix('.pkl'))
                # if save_lib: save_lib_files(df, save_dir=dir_tmp)

    # save duplicate
    if save_duplicate:
        if 'duplicateSeq' not in df.columns:
            logger.critical(f'save_duplicate requires column: duplicateSeq, but not found!!!')
        else:
            idx_duplicate_seqs = df.index[df.duplicateSeq]
            num_duplicate_seqs = len(idx_duplicate_seqs)
            logger.info(f'Number of duplicate sequences: {num_duplicate_seqs}')
            if num_duplicate_seqs > 0:
                unk_file = save_dir / (save_prefix + '_duplicate')
                df_tmp = df.loc[idx_duplicate_seqs]
                logger.info(f'Storing duplicate sequences as: {unk_file} of shape: {df_tmp.shape}')
                if save_csv: df_tmp.to_csv(unk_file.with_suffix('.csv'), **csv_kwargs)
                if save_pkl: df_tmp.to_pickle(unk_file.with_suffix('.pkl'))
                # if save_lib: save_lib_files(df, save_dir=dir_tmp)

    # # drop rows & columns
    # if 'save2lib' in df.columns:
    #     df.drop(index=df.index[~df.save2lib], inplace=True)

    # df.drop(columns=list(set(df.columns.to_list()) - set(pkl_include)), inplace=True)
    # recap.num_seqs_saved = df.shape[0]

    if pkl_exclude:
        pkl_columns = list(filter(lambda _s: _s not in pkl_exclude, df.columns.to_list()))

    if save_pkl:
        pkl_file = save_dir / (save_prefix + '.pkl')
        logger.info(f'Storing pickle: {pkl_file} of shape: {df.shape} ...')
        if pkl_exclude:
            pd.to_pickle(df[pkl_columns], pkl_file)
        else:
            df.to_pickle(pkl_file)
        # gwio.pickle_squeeze(df_tmp.to_dict(orient='list'), pkl_file, fmt='lzma')

    if save_fasta:
        fasta_file = save_dir / (save_prefix + '.fasta')
        logger.info(f'Storing fasta file: {fasta_file} of counts: {len(df)} ...')
        fasta_lines = misc.mpi_map(_ds2fasta_line,
            tqdm(df.iterrows(), total=len(df), desc='Get fasta lines', disable=tqdm_disable),
            quiet=True, starmap=True)

        # fasta_lines = itertools.starmap(lambda idx, ds: f'>{ds.idx}_{ds.id}\n{ds.seq}\n',
        #     tqdm(df.iterrows(), total=len(df), desc='Get fasta lines'))
        with open(fasta_file, 'w') as iofile:
            # for _i in tqdm(range(len(df)), desc='Saving Fasta'):
                # fasta_lines = f'>{df.iloc[_i]["idx"]}_{df.iloc[_i]["id"]}\n{df.iloc[_i]["seq"]}\n'
            iofile.writelines(fasta_lines)


def _ds2fasta_line(index, ds):
    """ only for the mpi_map above """
    # if 'idx' in ds:
    #     return f'>{ds.id.strip()}\n{ds.seq}\n'
    # else:
    #     return f'>{ds.id.strip()}\n{ds.seq}\n'
    return f'>{ds.id.strip()}\n{ds.seq}\n'


def save_all_files(data_names, recap=misc.Struct(),
        save_dir='./',
        save_prefix=None,            # default as data_names[0].stem
        save_lumpsum=True,           # whether to save lumpsum files
        save_pkl=True,
        save_csv=False,
        save_fasta=False,
        save_unknown=False,
        save_duplicate=False,
        save_conflict=False,
        save_lib=False,              # save entries with save2lib=True
        save_individual=False,       # whether to save individual files
        save_individual_dir=None,    # default to save_prefix
        save_genre=['seq'],          # only for save_individual
        named_after='file',
        save_json=True,
        **kwargs):
    """ a portal for save midat files in both lumpsum and individual forms """

    df, auto_save_prefix = get_midat(data_names, return_save_prefix=True)

    save_prefix = misc.get_1st_value([save_prefix, auto_save_prefix, 'autosave'])

    # saving files
    if save_lumpsum:
        save_lumpsum_files(df, recap=recap,
            save_dir=save_dir, save_prefix=save_prefix,
            save_pkl=save_pkl, save_fasta=save_fasta, save_csv=save_csv,
            save_unknown=save_unknown, save_duplicate=save_duplicate,
            save_conflict=save_conflict, **kwargs)

    if save_lib:
        if "save2lib" in df.columns:
            df_lib = df[df['save2lib'] == True]
        elif 'seq' in df.columns:
            logger.warning(f'save2lib not in df.columns, use unique sequences!')
            df_lib = df.groupby(by='seq').head(1)
        else:
            logger.warning(f'Neither save2lib or seq exists in df.columns, use all!!')
            df_lib = df

        recap.num_lib_seqs = len(df_lib)
        count_all_residues(df_lib, recap=recap, prefix='lib_', show=False)
        save_lumpsum_files(df_lib, recap=recap,
            save_dir=save_dir, save_prefix="libset" if save_prefix == 'allset' else (save_prefix + '_lib'),
            save_pkl=True, save_fasta=save_fasta, save_csv=save_csv,
            save_unknown=False, save_duplicate=False, save_conflict=False, **kwargs)

    if save_individual:
        save_individual_files(df, save_prefix=None,
            save_dir=save_prefix if save_individual_dir is None else save_individual_dir,
            save_genre=save_genre, named_after=named_after, **kwargs)

    # save json
    if save_json and len(recap):
        json_file = save_prefix + '.json'
        if save_dir is not None: json_file = Path(save_dir) / json_file
        logger.info(f'Storing json: {json_file}')
        gwio.dict2json(vars(recap), json_file)


def check_unknown_residues(
            data_names,
            recap=misc.Struct(),
            save2lib_only=True,
            **kwargs):
    """ check and tag entries for unknown residue names """
    # NOTE: all rows are kept for now. If too slow, one can drop invalid rows:
    # df_all.drop(index=df_all.index[~ df_all.save2lib], inplace=True)
    # recap_json.num_consistent_seq_ct = df_all.shape[0]
    # logger.info(f'{df_all.shape[0]} items with valid seq and ct out of ' + \
    #             f'{recap_json.num_seqs} total')

    # keys are the identities checked for duplication
    # vals are the columns whose values are further checked for value duplication

    df = get_midat(data_names)

    logger.info('Checking for unknown residue letters...')
    known_resnames = set(reslet_lookup.keys())
    recap.num_unknown_seqs = 0
    recap.known_resnames = ''.join(list(reslet_lookup.keys()))

    if 'unknownSeq' not in df.columns: df['unknownSeq'] = False
    if 'save2lib' not in df.columns: df['save2lib'] = True

    for i, seq in enumerate(df.seq):
        if len(set(seq) - known_resnames):
            recap.num_unknown_seqs += 1
            df.loc[i, 'unknownSeq'] = True
            df.loc[i, 'save2lib'] = False
    logger.info(f'Found {recap.num_unknown_seqs} sequences with unknown residues')

    return df, recap


def check_duplicate_keyvals(
            data_names,
            recap=misc.Struct(),
            save2lib_only=True,
            keys='seq', vals=None,
            **kwargs):
    """ check and tag entries for duplication, first by keys then by vals
        Note: the first entry with duplicate keys is now saved regardless
              of whether their vals are conflicting
    """
    # NOTE: all rows are kept for now. If too slow, one can drop invalid rows:
    # df_all.drop(index=df_all.index[~ df_all.save2lib], inplace=True)
    # recap_json.num_consistent_seq_ct = df_all.shape[0]
    # logger.info(f'{df_all.shape[0]} items with valid seq and ct out of ' + \
    #             f'{recap_json.num_seqs} total')

    # keys are the identities checked for duplication
    # vals are the columns whose values are further checked for value duplication

    df = get_midat(data_names)

    if vals is None:
        vals = []
    elif isinstance(vals, str):
        vals = [vals]

    logger.info('Checking for sequence/ct duplicates...')
    recap.keys = keys
    recap.vals = vals
    recap.num_conflict_grps = 0
    recap.num_conflict_vals = 0
    recap.num_duplicate_grps = 0
    recap.num_duplicate_seqs = 0
    recap.num_duplicate_seqvals = 0
    recap.num_duplicate_seqs_removed = 0

    if 'duplicateSeq' not in df: df['duplicateSeq'] = False
    if 'idxSameSeq' not in df: df['idxSameSeq'] = None
    if 'idxSameSeqVal' not in df: df['idxSameSeqVal'] = None
    if 'conflictVal' not in df : df['conflictVal'] = False
    if 'save2lib' not in df: df['save2lib'] = True

    if save2lib_only:
        df_grps = df[df.save2lib].groupby(by=keys)
    else:
        df_grps = df.groupby(by=keys)

    for seq, df_one_grp in tqdm(df_grps, desc='Checking duplicates'):
        if len(df_one_grp) == 1: continue

        num_seqs = len(df_one_grp)
        recap.num_duplicate_grps += 1
        recap.num_duplicate_seqs += num_seqs

        df.loc[df_one_grp.index, 'duplicateSeq'] = True
        df.loc[df_one_grp.index, 'idxSameSeq'] = df_one_grp.iloc[0]['idx']
        # df.loc[df_one_grp.index[1:], 'save2lib'] = False # keep 1st only

        all_same_vals = True
        for val in vals:
            val_1st = df_one_grp.iloc[0][val]
            same_grp_vals = [True] * num_seqs
            for igrp in range(1, len(df_one_grp)):
                same_grp_vals[igrp] = np.array_equal(val_1st, df_one_grp.iloc[igrp][val])

            if same_grp_vals.count(True) > 1:
                df.loc[df_one_grp.index[same_grp_vals], 'idxSameSeqVal'] = df_one_grp.iloc[0]['idx']

            if not all(same_grp_vals): # conflicting values within the same group
                all_same_vals = False
                recap.num_conflict_grps += 1
                recap.num_conflict_vals += num_seqs
                df.loc[df_one_grp.index, 'conflictVal'] = True
                # df.loc[df_one_grp.index, 'save2lib'] = False
                logger.warning(f'The same {keys} but different {vals} for the following:')
                print(seq)
                print(df_one_grp[['idx', 'file']])

        # keep the 1st value now, whether with the same or different values
        if all_same_vals:
            df.loc[df_one_grp.index[1:], 'save2lib'] = False
        else:
            df.loc[df_one_grp.index[1:], 'save2lib'] = False

    recap.num_duplicate_seqvals += (df.idxSameSeqVal > 0).to_numpy().astype(int).sum()
    recap.num_duplicate_seqs_removed = (df.duplicateSeq & (~ df.save2lib)).to_numpy().astype(int).sum()

    logger.info(f'Found {recap.num_duplicate_seqs} duplicate sequences in {recap.num_duplicate_grps} groups')
    logger.info(f'Found {recap.num_conflict_vals} conflicting values in {recap.num_conflict_grps} groups')

    return df, recap


def chew_midat(data_names,
               min_len=None,
               max_len=None,
               seq2upper=False,        # conver to upper case
               select_row=None,        # select passed rows
               get_duplicate=None,     # get duplicated rows by the passed column
               get_unique=None,        # select rows with unique values by the passed column
               include_seq=None,       # select sequences in passed fasta file or seq_str(s)
               exclude_seq=None,       # exclude sequences in the passed fasta file or seq_str(s)
               select_include=None,    # select by inlucding [column, values...]
               select_exclude=None,    # select by excluding [column, values...]
               split_groupby=None,     # split the dataframe by the groupby col
               tqdm_disable=False,     # NOTE: see save_all_files for saving args!!!
               **kwargs):
    """ comb through midata for various tasks such as selection and split """

    args = misc.Struct(locals())
    logger.debug('Arguments:')
    logger.debug(gwio.json_str(args.__dict__))

    df, auto_save_prefix = get_midat(data_names, return_save_prefix=True)
    auto_save_prefix = misc.get_1st_value([
        kwargs.get('save_prefix', None),
        auto_save_prefix,
        'refine_midat'])

    df_has_changed = False
    args.df_original_shape = df.shape

    if seq2upper:
        logger.info('Converting all sequences to upper case...')
        df['seq'] = df['seq'].str.upper()

    # min and max len
    num_rows = df.shape[0]
    if min_len is not None or max_len is not None:
        df_len_range = [df['len'].min(), df['len'].max()]
        logger.info(f'Length range: {df_len_range}')
        if min_len is None:
            min_len = df_len_range[0]
        else:
            df = df[df['len'] >= min_len]
            logger.info(f'Applied len>={min_len}, after shape: {df.shape}')
        if max_len is None:
            max_len = df_len_range[1]
        else:
            df = df[df['len'] <= max_len]
            logger.info(f'Applied len<={max_len}, after shape: {df.shape}')

        auto_save_prefix += f'_len{min_len}-{max_len}'
        if num_rows != df.shape[0]: 
            df_has_changed = True

    # unique rows
    num_rows = df.shape[0]
    if get_unique is not None:
        auto_save_prefix += '_unique'
        df = df.groupby(by=get_unique).head(1)
        if num_rows != df.shape[0]: 
            df_has_changed = True
        logger.info(f'Applied unique by [{get_unique}], shape: {df.shape}')

    # duplicate rows
    num_rows = df.shape[0]
    if get_duplicate is not None:
        df_grp_all = df.groupby(by=get_duplicate) # .filter(lambda x: len(x) > 1)
        auto_save_prefix += '_duplicate'
        for key, df_grp in df_grp_all:
            if len(df_grp) == 1: continue

            val = df_grp.iloc[0]['ct']
            for i in range(1, len(df_grp)):
                if not np.array_equal(val, df_grp.iloc[i]['ct']):
                    print(df_grp['id'])
        # df_dup = df[df.duplicated(subset=duplicate, keep=False)]
        if num_rows != df.shape[0]: df_has_changed = True

    # idx
    num_rows = df.shape[0]
    if select_row is not None:
        auto_save_prefix += '_rowselect'
        auto_save_prefix += 'notimplemented'
        if num_rows != df.shape[0]: df_has_changed = True

    # select by seq
    num_rows = df.shape[0]
    if include_seq is not None:
        auto_save_prefix += '_seqinclude'
        if isinstance(include_seq, str) and Path(include_seq).exists():
            seq_data = molstru.SeqsData(include_seq, fmt='fasta')
            include_seq = seq_data.seq
        if isinstance(include_seq, str):
            include_seq = [include_seq]

        df = df[df['seq'].isin(include_seq)]
        if num_rows != df.shape[0]: df_has_changed = True
        logger.info(f'Applied isin(sequence), after shape: {df.shape}')

    # select by seq
    num_rows = df.shape[0]
    if exclude_seq is not None:
        auto_save_prefix += '_seqexclude'
        if isinstance(exclude_seq, str) and Path(exclude_seq).exists():
            seq_data = molstru.SeqsData(exclude_seq, fmt='fasta')
            exclude_seq = seq_data.seq
        if isinstance(exclude_seq, str):
            exclude_seq = [exclude_seq]

        df = df[~df['seq'].isin(exclude_seq)]
        if num_rows != df.shape[0]: df_has_changed = True
        logger.info(f'Applied ~isin(sequence), after shape: {df.shape}')        

    # include column values
    num_rows = df.shape[0]
    if select_include is not None and len(select_include) > 1:
        auto_save_prefix += f'_{select_include[0]}-is-{"_".join(select_include[1:])}'
        logger.info(f'Applying select_include: {select_include}...')
        df = df[df[select_include[0]].isin(select_include[1:])]
        logger.info(f'{df.shape[0]} rows left after select_include')
        if num_rows != df.shape[0]: df_has_changed = True

    # exclude column values
    num_rows = df.shape[0]
    if select_exclude is not None and len(select_exclude) > 1:
        auto_save_prefix += f'_{select_exclude[0]}-not-{"_".join(select_exclude[1:])}'
        logger.info(f'Applying select_exclude: {select_exclude}...')
        df = df[~df[select_exclude[0]].isin(select_exclude[1:])]
        logger.info(f'{df.shape[0]} rows left after select_exclude')
        if num_rows != df.shape[0]: df_has_changed = True

    args.df_final_shape = df.shape
    if 'save_prefix' not in kwargs:
        kwargs['save_prefix'] = auto_save_prefix
    save_pkl = kwargs.pop('save_pkl', True)# and df_has_changed

    # save the main df
    if split_groupby is None:
        save_all_files(df, recap=args, tqdm_disable=tqdm_disable,
            save_pkl=save_pkl, **kwargs)
    else: # split_groupby and save each group
        if split_groupby not in df.columns:
            logger.critical(f'split_groupby: {split_groupby} not in df.columns!!!')
        else:
            df_grps = df.groupby(by=split_groupby)
            save_prefix = kwargs.pop('save_prefix')
            kwargs['save_json'] = False
            for key, df_one_grp in df_grps:
                save_all_files(df_one_grp, save_prefix=f'{save_prefix}_{key}',
                    save_pkl=True, tqdm_disable=tqdm_disable, **kwargs)

    return df


def split_midat_cv(data_names,
                   nfold=5,            # use split_midat(), take stratify, bucket_key, bucket_num
                   save_prefixes=None,
                   tqdm_disable=False, # NOTE: see save_all_files for saving args!!!
                   **kwargs):
    """ simply apply split_midat iteratively over train_set """

    recap = misc.Struct(locals(), PWD=Path.cwd().resolve().as_posix())
    logger.debug('Arguments:')
    logger.debug(gwio.json_str(recap.__dict__))

    train_set, auto_save_prefix = get_midat(data_names, return_save_prefix=True)
    if save_prefixes is None:
        save_prefix = misc.get_1st_value([auto_save_prefix, 'split_midat_cv'])
        save_prefixes = [f'{save_prefix}_cv{i}' for i in range(1, nfold + 1)]

    midat_list = []
    for i in range(nfold - 1):
        out_fraction = 1. / nfold / (1. - i / nfold)
        train_set, test_set = split_midat(train_set, fraction=1 - out_fraction,
            tqdm_disable=tqdm_disable, save_names=False, **kwargs)
        midat_list.append(test_set)
    midat_list.append(train_set)

    if save_prefixes is not False and len(save_prefixes) == nfold:
        for i in range(nfold):
            save_all_files(midat_list[i], save_prefix=save_prefixes[i],
                tqdm_disable=tqdm_disable, **kwargs)
    else:
        logger.info(f'No saving with save_prefixes: {save_prefixes}.')

    return midat_list


def split_midat(data_names,
                fraction=0.833,         # train fraction (default: 5/6)
                stratify=None,          # the stratify column
                bucket_key=None,        # divide the key col into buckets
                bucket_num=11,          # number of buckets
                shuffle=False,          # only used in train_test_split
                random_state=None,
                save_names=None,        # two names needed
                tqdm_disable=False,     # NOTE: see save_all_files for saving args!!!
                **kwargs):
    """ split midat based on stratify and/or bucket_key/num """

    recap = misc.Struct(locals())
    logger.debug('Arguments:')
    logger.debug(gwio.json_str(recap.__dict__))

    df, auto_save_prefix = get_midat(data_names, return_save_prefix=True)
    if save_names is None:
        save_prefix = misc.get_1st_value([auto_save_prefix, 'split_midat'])
        save_names = [f'{save_prefix}_train', f'{save_prefix}_test']

    df_list = [] # to be returned
    stratify_list = [None]  # this is passed to train_test_split as stratify=stratify_list[-1]
    bucketize = bucket_key is not None and bucket_num > 1

    if stratify:
        logger.info(f'Using column:{stratify} for stratified train_test_split...')
        shuffle = True
        stratify_list.append(df[stratify])

    if bucketize:
        shuffle=True
        df['splitBucket'] = 0
        # the index can be "corrupted" (e.g., duplicates) by dataframe concatation, etc.
        df.reset_index(inplace=True, drop=True)
        if stratify:
            df_grps = df.groupby(by=stratify)
        else:
            df_grps = [('all', df)] # mimic pd.grouby return

        for key, df_one_grp in df_grps:
            grp_size = len(df_one_grp)
            nbins = min([bucket_num, int(grp_size * min([fraction, 1 - fraction]) / 2)])
            if stratify:
                logger.info(f'Divide {bucket_key} into {nbins} buckets for {stratify}={key} with {grp_size} samples ...')
            else:
                logger.info(f'Divide {bucket_key} into {nbins} buckets for a total of {grp_size} samples ...')

            if nbins < 2:
                logger.debug(f'Only one bucket for {key} with {grp_size} samples!')
                # df.loc[df_one_grp.index, 'splitBucket'] = 0
            else:
                bucket_idx, bucket_grids = pd.qcut(df_one_grp[bucket_key], nbins, retbins=True, duplicates='drop', precision=1)
                logger.info(f'Bucket grids: {np.array2string(bucket_grids, precision=1, floatmode="fixed")}')
                df.loc[df_one_grp.index, 'splitBucket'] = bucket_idx.to_list()
                # df.loc[df_one_grp['__init_index__'].values]['splitBucket'] = bucket_idx.to_list()

        if stratify:
            new_col = f'{stratify}+splitBucket'
            df[new_col] = df[stratify].astype(str) + '+' + df['splitBucket'].astype(str)
            stratify_list.append(df[new_col])
        else:
            stratify_list.append(df['splitBucket'].astype(str))

        # df.drop(columns=['__init_index__'], inplace=True)
    while len(stratify_list):
        try:
            df_list = train_test_split(df,
                train_size=fraction,
                stratify=stratify_list.pop(-1),
                shuffle=shuffle,
                random_state=random_state)
            break
        except:
            logger.critical(f'Failed to train_test_split with last stratify, try coarser...')

    # data are now stored in df_list and save_name_list
    if save_names and len(df_list):
        for i, save_name in enumerate(save_names):
            save_all_files(df_list[i], recap=recap, save_prefix=save_name,
                tqdm_disable=tqdm_disable, **kwargs)

    return df_list


def bake_bert_vocab(
        letters='ACGT',    # letters used for making kmer
        k=3,               # k-mer
        padding=None,      # padding characeter (e.g., N)
        **kwargs):
    """ generate bert vocab file for kmer """

    vocab = ['[PAD]', '[UNK]', '[CLS]', '[SEP]', '[MASK]']

    if padding:
        if k % 2 != 1:
            logger.warning(f'k: {k} must be odd with padding!!!')
        nn = k // 2
        for num_pads in range(nn, 0, -1):
            pads = padding * num_pads
            for word in itertools.product(letters, repeat=k - num_pads):
                word = ''.join(word)
                vocab.append(pads + word)
                vocab.append(word + pads)

    for word in itertools.product(letters, repeat=k):
        vocab.append(''.join(word))

    save_path = f'bert{k}{padding.lower() if padding else ""}_vocab.txt'
    logger.info(f'Generated a total of {len(vocab)} words')
    logger.info(f'Saving file: {save_path} ...')
    with open(save_path, 'w') as iofile:
        iofile.writelines('\n'.join(vocab))


def bake_bert_chunks(data_names, save_dir='./',        # dataframe or dict (csv or pkl)
                     save_prefix=None,                 # save to {save_prefix}_chunks.fasta
                     convert2upper=True,               # conver to upper case
                     convert2dna=True,                 # convert to DNA
                     convert2cdna=False,               # convert to cDNA if id endswith('-')
                     vocab='ACGT',                     # the vocabulary set of characters
                     method='contig',                  # method for dividing chunks: contig/sample
                     coverage=2.0,                     # coverage only if method is sample
                     drop_last=False,                  # whether to drop the leftover if <min_len
                     min_len=30, max_len=510,          # chunk min and max lengths
                     num_cpus=0.3,                     # number of CPUs to use for multiprocessing
                     tqdm_disable=False,               # not yet implemented
                     **kwargs):
    """ start from a dataframe of sequences and chunkerize each seq for bert mlm"""

    save_dir = Path(save_dir)
    args = misc.Struct(locals())
    logger.info('Arguments:')
    print(gwio.json_str(args.__dict__))

    df, auto_save_prefix = get_midat(data_names, return_save_prefix=True)
    args.auto_prefix = misc.get_1st_value([save_prefix, auto_save_prefix, 'noname'])

    if convert2upper:
        logger.info('Converting all sequences to upper case...')
        df['seq'] = df['seq'].str.upper()

    if convert2dna:
        df['seq'] = misc.mpi_map(
            functools.partial(molstru.seq2DNA, vocab_set=set(vocab), verify=True),
            tqdm(df.seq, desc='Seq2DNA'),
            num_cpus=num_cpus,
            quiet=True)

    if convert2cdna:
        df['id'] = df['id'].str.strip()
        is_minus = df['id'].str.endswith('-')
        num_minus_strands = is_minus.sum()
        logger.info(f'Found {num_minus_strands} minus strand sequences')
        if num_minus_strands:
            df.loc[is_minus, 'seq'] = misc.mpi_map(
                functools.partial(molstru.seq2cDNA, reverse=True, verify=False),
                tqdm(df[is_minus].seq, desc='Seq2cDNA'),
                num_cpus=num_cpus, quiet=True)

    # chunkerize
    fn_seq2chunks = functools.partial(molstru.seq2chunks, vocab_set=set(vocab),
            method=method, min_len=min_len, max_len=max_len,
            drop_last=drop_last, coverage=coverage)

    logger.info(f'Chunkerize {len(df)} sequences...')
    chunk_seqs = misc.mpi_map(fn_seq2chunks, tqdm(df.seq, desc='Seq2Chunk'),
            num_cpus=num_cpus, quiet=True)

    assert len(chunk_seqs) == len(df), 'Sequence count changed after seq2chunks'

    # chunk_seqs = misc.unpack_list_tuple(chunk_seqs)
    # logger.info(f'Generated a total of {len(chunk_seqs)} chunks')

    # save to fasta
    args.save_path = save_dir / (args.auto_prefix + f'_chunks{min_len}-{max_len}_{method}.fasta')
    logger.info(f'Saving chunks to fasta file: {args.save_path}')
    args.num_chunks = 0
    with args.save_path.open('w') as iofile:
        # for i, id in tqdm(enumerate(df.id), desc='Save fasta'):
        for i in tqdm(range(len(df)), desc="Save fasta"):
            id = df.iloc[i].id
            args.num_chunks += len(chunk_seqs[i])
            if len(chunk_seqs[i]) == 0:
                logger.info(f'No chunks for id: {id}, len: {df.iloc[i]["len"]}, seq: {df.iloc[i].seq}')
                continue
            for i, seq in enumerate(chunk_seqs[i]):
                iofile.writelines([f'>{id}|chunk{i + 1}\n', seq, '\n'])

    logger.info(f'Saved a total of {args.num_chunks} chunks.')
    gwio.dict2json(args.__dict__, fname=args.save_path.with_suffix('.json'))


def bake_bert_encoding(data_names,
        model_path="/home/xqiu/github/DNABERT/examples/rnacentral5n/checkpoint-31000",
        k=5,
        padding='N',
        max_len=509,
        batch_size=4,
        save_name=None,
        **kwargs):
    """ add bert encoder representation and attention matrix to dataset """
    args = misc.Struct(locals(), PWD=Path.cwd().resolve().as_posix())
    logger.info('Arguments:')
    print(gwio.json_str(args.__dict__))

    # load sequence data
    df, save_prefix = get_midat(data_names, return_save_prefix=True)
    save_name = misc.get_1st_value((save_name, save_prefix, 'noname'))

    if max_len is not None:
        logger.info('Removing sequences longer than 510...')
        df['len'] = df.seq.str.len()
        df = df[df['len'] <= max_len]

    # load model
    import torch
    from transformers import BertTokenizer, BertModel, DNATokenizer

    tokenizer_name = f'dna{k}' + padding.lower() if padding is not None else ''
    model_path = args.model_path
    model = BertModel.from_pretrained(model_path,
            output_attentions=True, output_hidden_states=True)
    tokenizer = DNATokenizer.from_pretrained(tokenizer_name, do_lower_case=False)

    # calculate
    num_seqs = len(df)
    iseq = df.columns.get_loc('seq')
    bert_dataset, attn_dataset = [], []
    for ibatch, istart in tqdm(list(enumerate(range(0, num_seqs, batch_size)))):
        # use iloc here because index is no longer continuous
        batch_seqs = df.iloc[istart:istart + batch_size, iseq]
        batch_seqs = [molstru.seq2DNA(_s, verify=True) for _s in batch_seqs]
        batch_seqs = [_s.upper() for _s in batch_seqs]
        batch_lens = [len(_s) for _s in batch_seqs]

        batch_sentences = [molstru.seq2kmer(_s, k=5, padding='N') for _s in batch_seqs]
        # if len(batch_sentences) == 1:
        #     batch_sentences = batch_sentences[0]

        if isinstance(batch_sentences, str):
            # one sentence only, no attention mask needed
            inputs = tokenizer.encode_plus(batch_sentences, sentence_b=None,
                    return_tensors='pt', add_special_tokens=True,
                    max_length=None,
                    pad_to_max_length=False,
                    )
            attention_mask = None
        else:
            inputs = tokenizer.batch_encode_plus(batch_sentences, sentence_b=None,
                    return_tensors='pt', add_special_tokens=True,
                    max_length=None,
                    pad_to_max_length=True,
                    return_attention_masks=True,
                    )
            attention_mask = inputs['attention_mask']

        # inputs: {'input_ids': [batch_size, max(seqs_len) + 2],
        #          'token_type_ids': [batch_size, max(seqs_len) + 2],
        #          'attention_mask': [batch_size, max(seqs_len) + 2]}
        # CLS is added to the beginning
        # EOS is added to the end

        outputs = model(inputs['input_ids'], attention_mask=attention_mask)

        # outputs[0]: [batch_size, len(tokens)+2, 768] (the final hidden states of the bert base)
        # outputs[1]: [batch_size, 1, 768] (not sure what it is, sentence level summary?)
        # outputs[2]: hidden states
        #   [batch_size, max(seq_len) + 2, 768] * num_layers + 1
        # outputs[3]: attention matrix
        #   a tuple of length=num_layers
        #   each item has shape of [batch_size, nheads, len(tokens)+2, len(tokens)+2]

        # verify output[0] shoud be the same as the last hidden states: ouputs[0] == outputs[2][-1]
        if torch.any(outputs[0] != outputs[2][-1]):
            raise ValueError('Outputs[0] != outputs[2][-1], please check!!!')
        else:
            logger.debug(f'Outputs[0] == outputs[2][-1], good to go!')

        # save bert and attention matrix
        for i in range(len(batch_seqs)):
            bert_dataset.append(outputs[0][i, 1:batch_lens[i] + 1, :].detach().numpy().astype(np.float32))
            # the last layer only
            attn_dataset.append(outputs[-1][-1][i, :, 1:batch_lens[i] + 1, 1:batch_lens[i] + 1].detach().numpy().astype(np.float32))

    save_file = save_name + '_bert.pkl'
    logger.info(f'Saving pickle file: {save_file} ...')
    df['bert'] = bert_dataset
    df.to_pickle(save_file)

    save_file = save_name + '_attn.pkl'
    df.drop(columns='bert', inplace=True)
    logger.info(f'Saving pickle file: {save_file} ...')
    df['attn'] = attn_dataset
    df.to_pickle(save_file)


def mutate_sequences(data_names,             # dataframe in csv, pkl
                   rate=None,                # rate for all sequences (stems included!)
                   stem_rate=None,           # rate for stems
                   loop_rate=None,           # rate for loops
                   num_cpus=0.6,             # number/percentage of CPUs
                   save_dir='./',            # current dir is default
                   save_prefix=None,         # various suffixes will be added (overwritten by save_name)
                   save_name=None,           # the actual save_name
                   save_pkl=True,            # save results into pkl
                   save_fasta=True,          # save a single fasta file extracted from results
                   save_csv=False,           # save a summary csv file based on results
                   save_lib=False,           # save various columns into individual files (caution: many files)
                   named_after='file',       # how to name each individual file
                   save_genre=['seq'],       # which types of individual files to save
                   tqdm_disable=False,       # disable tqdm
                   **kwargs):
    """ convert each seq into its kmer representations """

    save_dir = Path(save_dir)
    args = misc.Struct(locals(), PWD=Path.cwd().resolve().as_posix())
    logger.debug('Arguments:')
    logger.debug(gwio.json_str(args.__dict__))

    df, auto_prefix = get_midat(data_names, return_save_prefix=True)
    args.auto_prefix = misc.get_1st_value([save_prefix, auto_prefix, 'noname'])

    if rate is not None:
        fn_mutate = functools.partial(molstru.mutate_residues, rate=rate, return_list=True)
        df['seq'] = misc.mpi_map(fn_mutate,
            df.seq if tqdm_disable else tqdm(df.seq, desc='MutateResidues'),
            num_cpus=num_cpus, quiet=True)
        args.auto_prefix += f'_all{rate:0.2f}'

    if stem_rate is not None or loop_rate is not None:
        if 'ct' not in df.columns:
            logger.critical(f'Cannot mutate stem or loop without ct in the dataframe!!!')
        fn_mutate = functools.partial(molstru.mutate_stems_loops, rate=0.0,
                stem_rate=stem_rate, loop_rate=loop_rate, return_list=True)
        df['seq'] = misc.mpi_starmap(fn_mutate,
            zip(df.seq, df.ct) if tqdm_disable else \
            tqdm(zip(df.seq, df.ct), desc='MutateStemLoops'),
            num_cpus=num_cpus, quiet=True)
        if stem_rate is not None: args.auto_prefix += f'_stem{stem_rate:0.2f}'
        if loop_rate is not None: args.auto_prefix += f'_loop{loop_rate:0.2f}'


    if save_pkl or save_fasta or save_csv:
        save_lumpsum_files(df, save_dir=save_dir,
            save_prefix=args.auto_prefix if save_name is None else save_name,
            save_pkl=save_pkl, save_fasta=save_fasta, save_csv=save_csv)

    if save_lib:
        lib_dir = save_dir / (args.auto_prefix if save_name is None else save_name)
        save_individual_files(df, save_dir=lib_dir, save_prefix=None,
            save_genre=save_genre, named_after=named_after,
            tqdm_disable=tqdm_disable)


def clone_seq2rna(data_names,           # dataframe in csv, pkl
                  tqdm_disable=False,   # NOTE: see save_all_files for saving args!!!
                  **kwargs):
    """ convert all nucleotides to AUCG """

    args = misc.Struct(locals(), PWD=Path.cwd().resolve().as_posix())
    logger.debug('Arguments:')
    logger.debug(gwio.json_str(args.__dict__))

    df, auto_save_prefix = get_midat(data_names, return_save_prefix=True)
    if auto_save_prefix is None: auto_save_prefix = 'autosave'

    if len(df) > 42:
        df['seq'] = misc.mpi_map(molstru.seq2RNA, tqdm(df['seq'], disable=tqdm_disable))
    else:
        df['seq'] = [molstru.seq2RNA(seq) for seq in df['seq']]

    save_all_files(df, tqdm_disable=tqdm_disable, **kwargs)


def clone_seq2kmer(data_names,           # dataframe in csv, pkl
                   save_dir='./',        # current dir is default
                   save_prefix=None,     # default names will be used if not set
                   k=5, padding='N',     # seq2kmer parameters
                   test_size=0.11,       # percentage of test dataset
                   num_cpus=0.6,         # number/percentage of CPUs
                   tqdm_disable=False,   # disable tqdm
                   **kwargs):
    """ convert each seq into its kmer representations """

    save_dir = Path(save_dir)
    args = misc.Struct(locals(), PWD=Path.cwd().resolve().as_posix())
    logger.debug('Arguments:')
    logger.debug(gwio.json_str(args.__dict__))

    df, auto_save_prefix = get_midat(data_names, return_save_prefix=True)
    args.auto_prefix = misc.get_1st_value([save_prefix, auto_save_prefix, 'noname'])

    fn_seq2kmer = functools.partial(molstru.seq2kmer, k=k, padding=padding)

    args.num_seqs = len(df)
    logger.info(f'Kmerize {args.num_seqs} sequences...')
    kmer_seqs = misc.mpi_map(fn_seq2kmer,
            df.seq if tqdm_disable else tqdm(df.seq, desc='Seq2Chunk'),
            num_cpus=num_cpus, quiet=True)

    assert len(kmer_seqs) == len(df), 'Length changed after seq2kmer'

    bert_label = f'_bert{k}' + padding if padding else ''
    args.save_path = save_dir / (args.auto_prefix + bert_label)
    gwio.dict2json(args.__dict__, fname=args.save_path.with_suffix('.json'))

    # split the dataset and save
    train_seqs, test_seqs = train_test_split(kmer_seqs, test_size=test_size)

    save_path = save_dir / (args.auto_prefix + bert_label + '_train.txt')
    logger.info(f'Saving BERT train sequences to txt file: {save_path}')
    with save_path.open('w') as iofile:
        for seq in train_seqs if tqdm_disable else tqdm(train_seqs, desc='Save train'):
            iofile.writelines([seq, '\n'])

    save_path = save_dir / (args.auto_prefix + bert_label + '_test.txt')
    logger.info(f'Saving BERT test sequences to txt file: {save_path}')
    with save_path.open('w') as iofile:
        for seq in tqdm(test_seqs, desc='Save test'):
            iofile.writelines([seq, '\n'])

    # save to fasta
    # save_path = save_dir / (save_prefix + bert_label + '.fasta')
    # logger.info(f'Saving BERT sequences to fasta file: {save_path}')
    # with save_path.open('w') as iofile:
    #     for i, seq in tqdm(enumerate(kmer_seqs)):
    #         iofile.writelines([f'>BERT-SEQ{i + 1}\n',
    #             molstru.kmer2seq(seq, padding=padding), '\n'])

    # save_path = save_dir / (save_prefix + bert_label + '.txt')
    # logger.info(f'Saving BERT sequences to txt file: {save_path}')
    # with save_path.open('w') as iofile:
    #     for seq in tqdm(kmer_seqs):
    #         iofile.writelines([seq, '\n'])
    return None


def clone_kmer2seq(
        data_names,                     # kmer sequence file (line by line)
        save_name=None, save_dir='./',  # default: noname
        padding=None,                   # whether to use
        tqdm_disable=False,
        **kwargs):
    """ infer seqence from its bert kmer sentence """

    data_names = misc.unpack_list_tuple(data_names)
    bert_seqs = gwio.get_file_lines(data_names, strip=True, keep_empty=False)
    save_name = misc.get_1st_value([save_name, 'noname'])
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    # show parameters?
    logger.info(f'   save_name: {save_name}')
    logger.info(f'    save_dir: {save_dir}')
    logger.info(f'     padding: {padding}')
    logger.info(f'   # of seqs: {len(bert_seqs)}')

    #
    fasta_seqs = []
    for seq in bert_seqs:
        fasta_seqs.append(molstru.kmer2seq(seq, padding=padding))

    save_path = (save_dir / save_name).with_suffix('.fasta')
    logger.info(f'Saving fasta file: {save_path}...')
    id_base = Path(data_names[0]).stem
    with save_path.open('w') as iofile:
        for i, seq in enumerate(fasta_seqs):
            iofile.writelines([f'>{id_base}_fasta{i}\n', fasta_seqs[i], '\n'])


def compare_dataset_pairs(
        data_names,
        fmt='pkl',
        key='seq',
        val='ct',
        save_dir='./',
        save_name=None,
        **kwargs):
    """ compare two datasets for intersection, union, etc. """
    if len(data_names) > 2:
        logger.warning('Only the first two files will be processed!')

    save_dir = Path(save_dir)

    # read files
    df_list, df_file = gwio.read_dataframes(data_names[0:2], fmt=fmt, concat=False, return_file=True)

    len_df = [_df.shape[0] for _df in df_list]
    df1 = df_list[0].assign(pkl_src=1, keep_me=True)
    df2 = df_list[1].assign(pkl_src=2, keep_me=True)
    df_all = pd.concat((df1, df2), axis=0, ignore_index=True, copy=False)

    # intersection (assume each df is already unique)
    df_intersect = df_all.groupby(by=key).filter(lambda grp: len(grp) > 1)

    df_grps = df_intersect.groupby(by=key)
    df_intersect_unique = df_grps.head(1)
    # check consistency in the intersection
    for seq, df_grp in df_grps:

        val0 = df_grp.iloc[0][val]
        same_grp_vals = [True]
        for igrp in range(1, len(df_grp)):
            same_grp_vals.append(np.array_equal(val0, df_grp.iloc[igrp][val]))

        if all(same_grp_vals): # consistent vals, keep the first one
            df_all.loc[df_grp.index[1:], 'keep_me'] = False
            # df_intersect.loc[df_grp.index[1:], 'saved'] = False
            logger.info(f'The same {key} and {val} for the following:')
            print(df_grp['id'])
        else: # inconsistent vals, keep no one
            df_all.loc[df_grp.index, 'keep_me'] = False
            df_intersect.loc[df_grp.index, 'keep_me'] = False
            logger.warning(f'The same {key} but different {val} for the following:')
            print(df_grp['id'])

    df_intersect_same = df_intersect[df_intersect.keep_me == True]
    df_intersect_diff = df_intersect[df_intersect.keep_me == False]

    # union
    df_union = df_all[df_all['keep_me'] == True]

    # save the unique ones for each list
    df1_unique = df1[df1[key].isin(df_intersect_unique[key]) == False]
    df2_unique = df2[df2[key].isin(df_intersect_unique[key]) == False]

    logger.info(f'DataFrame #1 shape: {df1.shape}')
    logger.info(f'DataFrame #1 shape: {df2.shape}')
    logger.info(f'Union shape: {df_union.shape}')
    logger.info(f'Intersect shape: {df_intersect.shape}')
    logger.info(f'Intersect same shape: {df_intersect_same.shape}')
    logger.info(f'Intersect diff shape: {df_intersect_diff.shape}')
    logger.info(f'Intersect unique shape: {df_intersect_unique.shape}')
    logger.info(f'DataFrame #1 Unique shape: {df1_unique.shape}')
    logger.info(f'DataFrame #2 Unique shape: {df2_unique.shape}')

    save_file = save_dir / 'intersect.pkl'
    logger.info(f'Saving pickle file: {save_file}')
    df_intersect.to_pickle(save_file, protocol=-1, compression='infer')


def gather_seq_pdb(data_names=None,
                   data_dir='./',
                   seq_suffix='.fasta',
                   ct_suffix='.ct',
                   pdb_suffix='.pdb',
                   angle_suffix='.json',
                   split_data=True, split_size=0.1,
                   min_len=1,
                   max_len=3200000000,
                   save_dir='./',
                   save_pkl=True,
                   save_lib=True,
                   tqdm_disable=False,
                   debug=False,
                   **kwargs):
    """ build dataset from seq and pdb databases """
    recap = locals()
    logger.debug('Arguments:')
    logger.debug(gwio.json_str(recap.__dict__))

    data_dir = Path(data_dir)
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    # get the sequence file
    seq_files = list(data_dir.glob('**/*' + seq_suffix))
    logger.info(f'Searched dir: {data_dir}, found {len(seq_files)} seq files')

    if debug:
        seq_files = seq_files[:77]

    # split the files
    all_files = dict()
    if split_data:
        data_groups = ['train', 'valid']
        all_files['train'], all_files['valid'] = train_test_split(seq_files, test_size=split_size,
                        shuffle=True, random_state=20130509)
    else:
        data_groups = ['train']
        all_files['train'] = seq_files

    # just do it
    for data_group in data_groups:
        logger.info(f'Processing data type: {data_group}...')

        seq_data = molstru.SeqsData(all_files[data_group], fmt=seq_suffix[1:])
        num_seqs = len(seq_data.seq)
        logger.info(f'Successfully parsed {num_seqs} seq files')

        pdb_data = misc.Struct(num=np.zeros(num_seqs, dtype=int),
                               len=np.zeros(num_seqs, dtype=int),
                               seq=[''] * num_seqs,
                               pdist=[None] * num_seqs,
                               valid=np.full(num_seqs, False) ,
                               )

        ct_data = misc.Struct(num=np.zeros(num_seqs, dtype=int),
                              len=np.zeros(num_seqs, dtype=int),
                              seq=[''] * num_seqs,
                              id=[''] * num_seqs,
                              ct=[np.empty(0, dtype=int)] * num_seqs,
                              valid=np.full(num_seqs, False) ,
                              )

        angle_data = misc.Struct(num=np.zeros(num_seqs, dtype=int),
                                len=np.zeros(num_seqs, dtype=int),
                                torsion=[None] * num_seqs,
                                valid=np.full(num_seqs, False),
                                )

        seq_data.valid = np.logical_and(seq_data.len >= min_len, seq_data.len <= max_len)

        for iseq, seq_file in enumerate(seq_data.file):

            if not seq_data.valid[iseq]:
                continue

            # pdb file
            _pdb_file = Path(seq_file).with_suffix(pdb_suffix)
            if _pdb_file.exists():
                pdb_data.num[iseq] = 1

                _pdb_data = molstru.AtomsData(_pdb_file.as_posix())
                pdb_data.len[iseq] = _pdb_data.numResids

                _pdb_seq = ''.join(_pdb_data.seqResidNames)
                if seq_data.len[iseq] == pdb_data.len[iseq]:
                    if seq_data.seq[iseq] == _pdb_seq: # a perfect match!
                        pdb_data.valid[iseq] = True
                    else:
                        logger.warning(f'[{seq_file}]: pdb has the same len but different seq!')
                else:
                    logger.warning(f'[{seq_file}]: pdb has different length and sequence!')

                if not pdb_data.valid[iseq]:
                    pdb_data.seq[iseq] = _pdb_seq
                    print(f'[{_pdb_file}]: SEQ: {seq_data.len[iseq]} and PDB: {_pdb_data.numResids}')
                    print(f'Sequence mismatch for {_pdb_file} -->')
                    print(f'SEQ: {seq_data.seq[iseq]}')
                    print(f'PDB: {pdb_data.seq[iseq]}')

                # calculate the distance matrix and save
                _pdb_data.calc_res2res_distance(atom='P', neighbor_only=False)
                pdb_data.pdist[iseq] = _pdb_data.dist_res2res_byatom

            # obtain&check ct file
            _ct_file = Path(seq_file).with_suffix(ct_suffix)
            if _ct_file.exists():
                ct_data.num[iseq] = 1
            else:
                _ct_file = list(_ct_file.parent.glob(_ct_file.stem + '_p*' + ct_suffix))
                ct_data.num[iseq] = len(_ct_file)
                if len(_ct_file):
                    logger.info(f'{len(_ct_file)} ct files found for {seq_file}')
                    print(_ct_file)
                    _ct_file = _ct_file[0]

            if ct_data.num[iseq] > 0: # process ct file
                _ct_data = molstru.parse_ct_lines(_ct_file, is_file=True)
                ct_data.id[iseq] = _ct_data['id']
                ct_data.len[iseq] = _ct_data['len']
                ct_data.ct[iseq] = _ct_data['ct']

                if seq_data.len[iseq] == ct_data.len[iseq]:
                    if seq_data.seq[iseq] == _ct_data['seq']: # a perfect match!
                        ct_data.valid[iseq] = True
                    else:
                        logger.warning(f'[{seq_file}]: ct has the same len but different seq!')

                elif seq_data.len[iseq] > _ct_data['len']:
                    # check whether a subset of seqdata.seq[iseq]
                    _seq_seq = ''.join([seq_data.seq[iseq][_i - 1] for _i in _ct_data['resnum']])
                    _ct_seq = ''.join([_ct_data['seq'][_i - 1] for _i in _ct_data['resnum']])
                    if _seq_seq == _ct_seq:
                        ct_data.valid[iseq] = True
                    else:
                        logger.warning(f'[{seq_file}]: ct has different length and sequence!')

                if not ct_data.valid[iseq]:
                    ct_data.seq[iseq] = _ct_data['seq']
                    print(f'[{_ct_file}]: SEQ: {seq_data.len[iseq]} and CT: {_ct_data["len"]}')
                    print(f'Sequence mismatch for {_ct_file} -->')
                    print(f'SEQ: {seq_data.seq[iseq]}')
                    print(f' CT: {ct_data.seq[iseq]}')

            # angle file
            _angle_file = Path(seq_file).with_suffix(angle_suffix)
            if _angle_file.exists():

                with _angle_file.open('r') as iofile:
                    try: # sometimes the json file is corrupted
                        _angle_data = json.load(iofile)
                    except:
                        _angle_data = dict()

                angle_data.num[iseq] = 1

                # get the torsion angles form json file
                if 'nts' in _angle_data and seq_data.len[iseq] == len(_angle_data['nts']):
                    angle_data.valid[iseq] = True
                    angle_data.len[iseq] = len(_angle_data['nts'])
                    angle_data.torsion[iseq] = [[_nt['alpha'], _nt['beta'], _nt['gamma'],
                                                 _nt['delta'], _nt['epsilon'], _nt['zeta']]
                                                 for _nt in _angle_data['nts']]
                    # angle_data.torsion[iseq][0][:2] = 0.0, 0.0 # alpha and beta for 1st nt
                    # angle_data.torsion[iseq][-1][-2:] = 0.0, 0.0 # epsilon and zeta for last nt
                    # there are more None values though!
                    angle_data.torsion[iseq] = np.stack(angle_data.torsion[iseq], axis=0)
                    angle_data.torsion[iseq][np.where(angle_data.torsion[iseq] == None)] = 0.0
                else:
                    logger.warning(f'[{seq_file}]: angle has different seq length!')

        # determine whether to save each seq
        seq2sav = np.logical_and(ct_data.valid, pdb_data.valid)
        seq2sav = np.logical_and(seq2sav, angle_data.valid)

        # save csv
        df = seq_data.to_df()
        df = df.assign(saved=seq2sav, validCT=ct_data.valid, numCT=ct_data.num, lenCT=ct_data.len,
                       lenDiffCT=seq_data.len - ct_data.len, seqCT = ct_data.seq,
                       validPDB=pdb_data.valid, numPDB=pdb_data.num, lenPDB=pdb_data.len,
                       lenDiffPDB=seq_data.len - pdb_data.len, seqPDB=pdb_data.seq,
                       validAngle=angle_data.valid, numAngle=angle_data.num,
                       lenAngle=angle_data.len,
                       )

        csv_file = save_dir / (data_group + '.csv')
        logger.info(f'Storing csv: {csv_file}')
        df.to_csv(csv_file, index=False, float_format='%8.6f')

        # remove invalid sequences
        logger.info(f'{seq_data.valid.astype(int).sum()} out of {len(seq_data.seq)} are within' + \
                    f' length range of [{min_len}, {max_len}]')
        idx2sav = np.where(seq2sav)[0]
        logger.info(f'{len(idx2sav)} entries have valid ct, pdb, and angle data')
        seq_data = seq_data.get_subset(idx2sav)

        if save_pkl:
            midata = dict(id=seq_data.id, len=seq_data.len, seq=seq_data.seq,
                          ct=[ct_data.ct[_i] for _i in idx2sav],
                          pdist=[pdb_data.pdist[_i] for _i in idx2sav],
                          angle=[angle_data.torsion[_i] for _i in idx2sav],
            )

            pkl_file = save_dir / (data_group + '.pkl')
            logger.info(f'Storing pickle: {pkl_file}')
            gwio.pickle_squeeze(midata, pkl_file, fmt='lzma')

        if save_lib:
            lib_dir = save_dir / data_group
            lib_dir.mkdir(exist_ok=True)
            logger.info(f'Saving lib files in: {lib_dir}')

            with tqdm(total=len(seq_data.seq), disable=tqdm_disable) as prog_bar:
                for i in range(len(seq_data.seq)):
                    sav_file = lib_dir / f'{i + 1:d}.input.fasta'
                    with sav_file.open('w') as iofile:
                        iofile.writelines('\n'.join(
                            seq_data.get_fasta_lines(i, dbn=False, upp=False)
                        ))

                    if data_group != 'predict':
                        ct_mat = molstru.ct2ctmat(ct_data.ct[idx2sav[i]], seq_data.len[i])
                        sav_file = lib_dir / f'{i + 1:d}.label.ctmat'
                        np.savetxt(sav_file, ct_mat, fmt='%1i')

                        sav_file = lib_dir / f'{i + 1:d}.label.pdist'
                        np.savetxt(sav_file, pdb_data.pdist[idx2sav[i]], fmt='%8.2f')

                        sav_file = lib_dir / f'{i + 1:d}.label.angle'
                        np.savetxt(sav_file, angle_data.torsion[idx2sav[i]], fmt='%8.2f')

                    if (i + 1) % 10 == 0:
                        prog_bar.update(10)
                prog_bar.update((i + 1) % 10)


def gather_fasta_st(**kwargs):
    """ a wrapper for gather_seq_ct """
    args = dict(seq_fmt='fasta', seq_suffix='.fasta', ct_fmt='st', ct_suffix='.st')
    args.update(kwargs)
    gather_seq_ct(**args)


def gather_seq_ct(
        file_prefixes=None, # prefixes are passed, usually just one file
        data_base=None,     # database: archiveii, stralign, bprna, etc.
        data_dir='./',
        seq_dir=None,       # default to data_dir
        seq_fmt='seq',
        seq_suffix=None,    # default to seq_fmt
        ct_dir=None,        # default to seq_dir, then data_dir
        ct_fmt='ct',
        ct_suffix=None,
        min_len=None,
        max_len=None,
        seq2upper=True,     # all resnames to upper case
        check_unknown=True,
        check_duplicate=True,
        debug=False,
        tqdm_disable=False,
        **kwargs):
    """ build dataset from seq and ct file databases with cross checking """
    recap = misc.Struct(locals())
    logger.debug('Arguments:')
    logger.debug(gwio.json_str(recap.__dict__))

    # get seq and ct files
    seq_dir = misc.get_1st_value([seq_dir, ct_dir, data_dir], default='./')
    seq_dir = Path(seq_dir)
    seq_suffix = misc.get_1st_value([seq_suffix, seq_fmt], '.seq')
    if len(seq_suffix) and not seq_suffix.startswith('.'):
        seq_suffix = '.' + seq_suffix
    logger.info(f'Searching dir: {seq_dir} for **/*{seq_suffix}...')
    seq_files = list(seq_dir.glob('**/*' + seq_suffix))
    logger.info(f'Found {len(seq_files)} {seq_fmt} files')

    ct_dir = misc.get_1st_value([ct_dir, seq_dir, data_dir], default='./')
    ct_dir = Path(ct_dir)
    ct_suffix = misc.get_1st_value([ct_suffix, ct_fmt], '.ct')
    if len(ct_suffix) and not ct_suffix.startswith('.'):
        ct_suffix = '.' + ct_suffix
    logger.info(f'Searching dir: {ct_dir} for **/*{ct_suffix}...')
    ct_files = list(ct_dir.glob('**/*' + ct_suffix))
    logger.info(f'Found {len(ct_files)} {ct_fmt} files')

    gwio.files_check_sibling(seq_files, fdir=None if seq_dir.samefile(ct_dir) else ct_dir,
                suffix=ct_suffix)
    gwio.files_check_sibling(ct_files, fdir=None if ct_dir.samefile(seq_dir) else seq_dir,
                suffix=seq_suffix)

    if debug: seq_files = seq_files[:100]

    # main loop
    logger.info(f'Reading {len(seq_files)} {seq_fmt} files...')
    seq_data = molstru.SeqsData(sorted(seq_files), fmt=seq_fmt, database=data_base)

    recap.update(seq_dir=str(seq_dir), seq_fmt=seq_fmt,
                 ct_dir=str(ct_dir), ct_fmt=ct_fmt,
                 num_seq_files=len(seq_files), num_ct_files=len(ct_files),
                 num_seqs=len(seq_data.seq))
    recap.len_minmax = [seq_data.len.min(), seq_data.len.max()]

    logger.info(f'Successfully parsed {recap.num_seqs} {seq_fmt} files, len_minmax: {recap.len_minmax}')

    seq_data.idx = np.arange(1, recap.num_seqs + 1)
    if min_len is not None and min_len > recap.seq_minmax[0]:
        seq_data.len_selected = seq_data.len > min_len
    else:
        seq_data.len_selected = None

    if max_len is not None and max_len < recap.seq_minmax[1]:
        if seq_data.len_selected is None:
            seq_data.len_selected = seq_data.len <= max_len
        else:
            seq_data.len_selected = np.logical_and(seq_data.selected, seq_data.len <= max_len)

    if seq_data.len_selected is None:
        recap.num_len_selected = recap.num_seqs
        seq_data.len_selected = np.full(recap.num_seqs, True)
    else:
        recap.num_len_selected = seq_data.len_selected.astype(int).sum()
        logger.info(f'{recap.num_len_selected} out of {recap.num_seqs} are ' + \
                f'within length range of [{min_len}, {max_len}]')

    ct_data = misc.Struct(num=np.zeros(recap.num_seqs, dtype=int),
                          len=np.zeros(recap.num_seqs, dtype=int),
                          seq=[''] * recap.num_seqs,
                          id=[''] * recap.num_seqs,
                          ct=[np.empty(0, dtype=int)] * recap.num_seqs,
                          sameSeq=np.full(recap.num_seqs, False),
                          sameSeqCase=np.full(recap.num_seqs, False),
                          )

    logger.info(f'Reading {ct_fmt} files...')
    for iseq, seq_file in enumerate(tqdm(seq_data.file)):

        if not seq_data.len_selected[iseq]:
            logger.debug(f'Skipping out-of-length-range {seq_fmt}: {seq_file}')
            continue

        # get ct file
        if seq_dir.samefile(ct_dir):
            _ct_file = Path(seq_file).with_suffix(ct_suffix)
        else:
            _ct_file = ct_dir / (Path(seq_file).stem + ct_suffix)

        if _ct_file.exists():
            ct_data.num[iseq] = 1
        else:
            _ct_file = list(_ct_file.parent.glob(_ct_file.stem + '_p*' + ct_suffix))
            ct_data.num[iseq] = len(_ct_file)
            if len(_ct_file):
                logger.info(f'{len(_ct_file)} {ct_fmt} files found for {seq_file}')
                print(_ct_file)
                _ct_file = _ct_file[0]

        if ct_data.num[iseq] <= 0:
            logger.warning(f'No {ct_suffix} file found for {seq_file} in {ct_dir}')
            continue

        # load ct and check consistency
        if ct_fmt == 'ct':
            _ct_data = molstru.parse_ct_lines(_ct_file)
        elif ct_fmt == 'bpseq':
            _ct_data = molstru.parse_bpseq_lines(_ct_file)
        elif ct_fmt in ['st', 'sta']:
            # [0] is used here because a st file can contain multiple sequences
            _ct_data = molstru.parse_st_lines(_ct_file, fmt=ct_fmt)[0]
        elif ct_fmt in ['bps']:
            _ct_data = dict(
                id=str(_ct_file),
                ct=molstru.parse_bps_lines(_ct_file),
                len=seq_data.len[iseq],
                seq=seq_data.seq[iseq],
                resnum=np.linspace(1, seq_data.len[iseq], seq_data.len[iseq], dtype=int),
                )
            # _ct_data =
        else:
            logger.critical(f'Unrecognized ct_fmt: {ct_fmt}')

        # _ct_data['seq'] is saved only when there is conflict with seq_data.seq[iseq]
        ct_data.id[iseq] = _ct_data['id']
        ct_data.len[iseq] = max([_ct_data['len'], len(_ct_data['seq'])])
        ct_data.ct[iseq] = _ct_data['ct']

        # only check residues in _ct_data['resnum']
        if _ct_data['resnum'][-1] <= seq_data.len[iseq]:
            _seq_seq = ''.join([seq_data.seq[iseq][_i - 1] for _i in _ct_data['resnum']])
        else:
            logger.warning(f'Larger resnum in {ct_fmt} than {seq_fmt}_len: {seq_file}')
            _seq_seq = ''.join([seq_data.seq[iseq][_i - 1] for _i in
                            _ct_data['resnum'][_ct_data['resnum'] <= seq_data.len[iseq]]])

        _ct_seq = ''.join([_ct_data['seq'][_i - 1] for _i in _ct_data['resnum']])

        if len(_seq_seq) != len(_ct_seq):
            logger.warning(f'Length mismatch of the sequence: {seq_file}')
        elif _seq_seq == _ct_seq:
            ct_data.sameSeq[iseq] = True
            ct_data.sameSeqCase[iseq] = True
        else:
            _seq_seq = _seq_seq.upper()
            _ct_seq = _ct_seq.upper()
            if _seq_seq == _ct_seq:
                ct_data.sameSeq[iseq] = True
                logger.info(f'Found seq case conflict: {seq_file}')
            elif seq_fuzzy_match(_ct_seq, _seq_seq):
                ct_data.sameSeq[iseq] = True
                logger.info(f'Found seq case conflict and fuzzy match: {seq_file}')
            else:
                logger.warning(f'[{seq_file}]: {seq_fmt} and {ct_fmt} have different sequence!')

        # # same seq and ct length
        # if seq_data.len[iseq] == len(_ct_data['seq']):
        #     if seq_data.seq[iseq] == _ct_data['seq']: # a perfect match!
        #         ct_data.sameSeq[iseq] = True
        #         ct_data.sameSeqCase[iseq] = True
        #     else: # check case and N
        #         _seq_seq = seq_data.seq[iseq].upper()
        #         _ct_seq = _ct_data['seq'].upper()

        #         if _seq_seq == _ct_seq:
        #             ct_data.sameSeq[iseq] = True
        #             logger.info(f'Seq case conflict for {seq_file}')
        #         elif seq_match_lookup(_ct_seq, _seq_seq):
        #             ct_data.sameSeq[iseq] = True
        #             logger.info(f'Seq lookup needed for {seq_file}')
        #         else:
        #             logger.warning(f'[{seq_file}]: ct has the same len but different seq!')

        # # different seq and ct length
        # elif seq_data.len[iseq] > _ct_data['len']:
        #     # check whether a subset of seqdata.seq[iseq]
        #     _seq_seq = ''.join([seq_data.seq[iseq][_i - 1] for _i in _ct_data['resnum']])
        #     _ct_seq = ''.join([_ct_data['seq'][_i - 1] for _i in _ct_data['resnum']])
        #     if _seq_seq == _ct_seq:
        #         ct_data.sameSeq[iseq] = True
        #         ct_data.sameSeqCase[iseq] = True
        #     else:
        #         _seq_seq = _seq_seq.upper()
        #         _ct_seq = _ct_seq.upper()
        #         if _seq_seq == _ct_seq:
        #             ct_data.sameSeq[iseq] = True
        #             logger.info(f'Seq case conflict for {seq_file}')
        #         elif seq_match_lookup(_ct_seq, _seq_seq):
        #             ct_data.sameSeq[iseq] = True
        #             logger.info(f'Seq lookup needed for {seq_file}')
        #         else:
        #             logger.warning(f'[{seq_file}]: ct has different length and sequence!')

        if not ct_data.sameSeq[iseq]:
            ct_data.seq[iseq] = _ct_data['seq']
            print(f'[{_ct_file}]: SEQ: {seq_data.len[iseq]} and CT: {_ct_data["len"]}')
            print(f'Sequence mismatch for {_ct_file} -->')
            print(f'SEQ: {seq_data.seq[iseq]}')
            print(f' CT: {ct_data.seq[iseq]}')

    logger.info(f'Finished reading {ct_data.num.sum()} {ct_fmt} files')

    # collect results and analyze

    logger.info(f'Collecting data as dataframes for post-processing...')
    seq_data.ct = ct_data.ct
    df_all = seq_data.to_df(seq=True, ct=True, has=True,
                            res_count=True, bp_count=True,
                            stem_count=True, pknot_count=True)
    df_all = df_all.assign(
            # data info
            lenSelected = seq_data.len_selected,
            numCT = ct_data.num,
            lenCT = ct_data.len,
            lenDiffCT = seq_data.len - ct_data.len,
            seqCT = ct_data.seq,
            sameSeqCT = ct_data.sameSeq,
            sameSeqCaseCT = ct_data.sameSeqCase,
            # fields for resolving duplicates
            unknownSeq = False,
            conflictSeq = np.logical_and(np.logical_not(ct_data.sameSeq), seq_data.len_selected),
            duplicateSeq = np.full(recap.num_seqs, False),
            conflictVal = False,
            idxSameSeq = [0] * recap.num_seqs,
            idxSameSeqVal = [0] * recap.num_seqs,
            # fields for saving pkl and lib
            save2lib = np.logical_and(ct_data.sameSeq, seq_data.len_selected),
            dataset = [''] * recap.num_seqs, # 'train' or 'valid'
            )

    recap.num_len_selected = df_all.lenSelected.sum()
    recap.num_len_mismatch = (df_all.lenDiffCT != 0).sum()
    recap.num_seq_matched = df_all.sameSeqCT.sum()
    recap.num_case_conflict = recap.num_seq_matched - df_all.sameSeqCaseCT.sum()

    count_all_residues(df_all, recap=recap, prefix='raw_')

    if seq2upper:
        logger.info('Converting to upper cases...')
        df_all.seq = misc.mpi_map(str.upper, tqdm(df_all.seq, desc='Convert2Upper'), quiet=True )

    if check_unknown:
        check_unknown_residues(df_all, recap=recap)

    if check_duplicate:
        check_duplicate_keyvals(df_all, recap=recap, keys='seq', vals=['ct'])

    # kwargs['save_prefix'] = kwargs.get('save_prefix', auto_save_prefix)
    save_all_files(df_all, recap=recap, tqdm_disable=tqdm_disable, **kwargs)

def gather_bpseq(bpseq_files=None, **kwargs):
    """ a wrapper for gather_ct """
    args = dict(ct_fmt='bpseq', )
    args.update(kwargs)
    gather_ct(ct_files=bpseq_files, **args)


def gather_ct(
        ct_files=None,               # can be zero, one, or multiple ct files
        data_base=None,              # database: archiveII, stralign, bpRNA,
        data_dir='./', ct_dir=None,  # same as data_dir
        data_fmt='ct', ct_fmt=None,  # default to ct_fmt
        data_size=None,
        ct_suffix=None,
        min_len=None,
        max_len=None,
        idx_prefix=None,             # whether add idx_ to file and id
        seq2upper=False,             # resnames to upper case
        check_unknown=False,
        check_duplicate=False,
        tqdm_disable=False,          # NOTE: see save_all_files for saving args!!!
        **kwargs):
    """ build dataset from ct-type files only
    Note:
        idx_ is added to each file name and id
    """
    recap = misc.Struct(locals())
    logger.debug('Arguments:')
    logger.debug(gwio.json_str(recap.__dict__))

    ct_dir = misc.get_1st_value([ct_dir, data_dir], default='./')
    ct_fmt = misc.get_1st_value([ct_fmt, data_fmt], default='ct')
    ct_suffix = misc.get_1st_value([ct_suffix, ct_fmt], default='.ct')
    if len(ct_suffix) and not ct_suffix.startswith('.'):
        ct_suffix = '.' + ct_suffix

    auto_save_prefix = Path(ct_dir).resolve().stem

    # get ct_files
    if ct_files is not None and len(ct_files):
        if isinstance(ct_files, str): ct_files = [ct_files]
        ct_files = [Path(_f) for _f in ct_files]
    else:
        # ct_files = [_f.resolve().as_posix() for _f in data_dir.glob('**/*.ct')]
        logger.info(f'Searching dir: {ct_dir} for **/*{ct_suffix}...')
        ct_files = list(Path(ct_dir).glob('**/*' + ct_suffix))
        logger.info(f'Found {len(ct_files)} {ct_fmt} files')

    if data_size is not None:
        logger.info(f'Selecting data_size: {data_size} ...')
        ct_files = ct_files[:data_size]

    # read ct files and apply length range [min_len, max_len]
    logger.info(f'Reading {len(ct_files)} {ct_fmt} files...')
    seq_data = molstru.SeqsData(sorted(ct_files), fmt=ct_fmt, database=data_base)
    recap.num_files = len(ct_files)
    recap.num_seqs = len(seq_data.seq)
    recap.len_minmax = [seq_data.len.min(), seq_data.len.max()]
    logger.info(f'Successfully parsed {recap.num_seqs} {ct_fmt} files, len_minmax: {recap.len_minmax}')

    seq_data.idx = np.arange(1, recap.num_seqs + 1)
    if min_len is not None and min_len > recap.seq_minmax[0]:
        seq_data.len_selected = seq_data.len > min_len
    else:
        seq_data.len_selected = None

    if max_len is not None and max_len < recap.seq_minmax[1]:
        if seq_data.len_selected is None:
            seq_data.len_selected = seq_data.len <= max_len
        else:
            seq_data.len_selected = np.logical_and(seq_data.selected, seq_data.len <= max_len)

    if seq_data.len_selected is None:
        recap.num_len_selected = recap.num_seqs
        seq_data.len_selected = np.full(recap.num_seqs, True)
    else:
        recap.num_len_selected = seq_data.len_selected.astype(int).sum()
        logger.info(f'{recap.num_len_selected} out of {recap.num_seqs} are ' + \
                f'within length range of [{min_len}, {max_len}]')

   # collect results and analyze
    logger.info(f'Collecting data into dataframes to scrutinize...')

    seq_len = np.array([len(_s) for _s in seq_data.seq], dtype=int)
    df_all = seq_data.to_df(seq=True, ct=True, tangle=ct_fmt == 'tangle', has=True,
                            res_count=True, bp_count=True,
                            stem_count=True, pknot_count=True)

    if idx_prefix:
        df_all['srcfile'] = df_all['file']
        df_all['srcid'] = df_all['id']
        df_all['file'] = [f'{df_all.iloc[_i]["idx"]}_{misc.str_deblank(Path(df_all.iloc[_i]["file"]).name)}' for _i in range(len(df_all))]
        df_all['id'] = [f'{df_all.iloc[_i]["idx"]}_{df_all.iloc[_i]["id"].strip()}' for _i in range(len(df_all))]
    # df_all['lenBin'] = df_all['len'] // bin_len if bin_len else 0

    df_all = df_all.assign(
            lenSelected = seq_data.len_selected,
            lenDiffSeq = seq_data.len - seq_len,
            # fields for resolving duplicates
            unknownSeq = False,
            conflictSeq = seq_data.len != seq_len,
            duplicateSeq = np.full(recap.num_seqs, False),
            idxSameSeq = [0] * recap.num_seqs,
            idxSameSeqVal = [0] * recap.num_seqs,
            conflictVal = np.full(recap.num_seqs, False),
            # fields for saving pkl and lib
            save2lib = seq_data.len_selected,
            dataset = [''] * recap.num_seqs, # 'train' or 'valid'
            )
    recap.num_len_mismatch = (df_all.lenDiffSeq != 0).sum()

    count_all_residues(df_all, recap=recap, prefix='raw_', show=False)

    if seq2upper:
        logger.info('Converting to upper cases...')
        df_all.seq = misc.mpi_map(str.upper, tqdm(df_all.seq, desc='Convert2Upper'), quiet=True)

    if check_unknown:
        check_unknown_residues(df_all, recap=recap)

    if check_duplicate:
        check_duplicate_keyvals(df_all, recap=recap, keys='seq', vals=['ct'])

    # # saving files
    # save_lumpsum_files(df_all, recap=recap,
    #         save_dir=save_dir, save_name=save_name,
    #         save_pkl=save_pkl, save_fasta=save_fasta, save_csv=save_csv,
    #         save_unknown=True, save_duplicate=True, save_conflict=True,
    #         **kwargs)

    # if save_lib and "save2lib" in df_all.columns:
    #     df_lib = df_all[df_all['save2lib'] == True]
    #     save_lumpsum_files(df_lib, recap=recap,
    #         save_dir=save_dir, save_name=save_name + '_lib',
    #         save_pkl=save_pkl, save_fasta=save_fasta, save_csv=save_csv,
    #         save_unknown=False, save_duplicate=False, save_conflict=False,
    #         **kwargs)

    kwargs['save_prefix'] = kwargs.get('save_prefix', auto_save_prefix)
    save_all_files(df_all, recap=recap, tqdm_disable=tqdm_disable, **kwargs)


def gather_fasta(fasta_files=None, **kwargs):
    """ a wrapper for gather_ct """
    args = dict(seq_fmt='fasta', )
    args.update(kwargs)
    gather_seq(seq_files=fasta_files, **args)


def gather_seq(
        seq_files=None,
        data_base=None,                 # achiveII/stralign/bprna, used to get mol_type
        data_dir='./', seq_dir=None,    # default: data_dir
        data_fmt='seq', seq_fmt=None,   # seq/fasta/json/
        seq_suffix=None,                # default: .seq_fmt
        data_size=None,                 # limit the data_size (usually for testing)
        has_upp=False,                  # whether has upp (unused)
        min_len=None,                   # remove seqs with len < min_len
        max_len=None,                   # remove seqs with len > max_max
        idx_prefix=None,                # add idx_ to file and id
        seq2upper=False,                # convert to upper case
        seq2dna=False,                  # convert to DNA alphabets (ATGC) (not yet implemented)
        check_unknown=False,            # check for unknown resnames
        check_duplicate=False,          # check for duplicated seqs
        tqdm_disable=False,             # NOTE: see save_all_files for saving args!!!
        **kwargs):
    """ build dataset from seq file database only """
    recap = misc.Struct(locals(), PWD=Path.cwd().resolve().as_posix())
    logger.debug('Arguments:')
    logger.debug(gwio.json_str(recap.__dict__))

    if seq_files is not None and len(seq_files):
        if isinstance(seq_files, str): seq_files = [seq_files]
        seq_files = [Path(_f) for _f in seq_files]
    else:
        seq_dir = misc.get_1st_value([seq_dir, data_dir], default='./')
        seq_suffix = misc.get_1st_value([seq_suffix, seq_fmt], '.fasta')
        if len(seq_suffix) and not seq_suffix.startswith('.'):
            seq_suffix = '.' + seq_suffix
        seq_dir = Path(seq_dir)

        logger.info(f'Searching dir: {seq_dir} for **/*{seq_suffix}...')
        seq_files = list(seq_dir.glob('**/*' + seq_suffix))
        logger.info(f'Found {len(seq_files)} {seq_fmt} files')

    if data_size is not None:
        logger.info(f'Selecting data_size: {data_size} ...')
        seq_files = seq_files[:data_size]

    # read files and apply length range [min_len, max_len]
    logger.info(f'Reading {len(seq_files)} {seq_fmt} files...')
    seq_data = molstru.SeqsData(sorted(seq_files), fmt=seq_fmt, database=data_base)

    recap.update(data_dir=data_dir, seq_dir=seq_dir, seq_fmt=seq_fmt)
    recap.num_files = len(seq_files)
    recap.num_seqs = len(seq_data.seq)
    recap.len_minmax = [seq_data.len.min(), seq_data.len.max()]
    logger.info(f'Successfully parsed {recap.num_seqs} {seq_fmt} files')

    seq_data.idx = np.arange(1, recap.num_seqs + 1)
    if min_len is not None and min_len > recap.seq_minmax[0]:
        seq_data.len_selected = seq_data.len > min_len
    else:
        seq_data.len_selected = None

    if max_len is not None and max_len < recap.seq_minmax[1]:
        if seq_data.len_selected is None:
            seq_data.len_selected = seq_data.len <= max_len
        else:
            seq_data.len_selected = np.logical_and(seq_data.selected, seq_data.len <= max_len)

    if seq_data.len_selected is None:
        recap.num_len_selected = recap.num_seqs
        seq_data.len_selected = np.full(recap.num_seqs, True)
    else:
        recap.num_len_selected = seq_data.len_selected.astype(int).sum()
        logger.info(f'{recap.num_len_selected} out of {recap.num_seqs} are ' + \
                f'within length range of [{min_len}, {max_len}]')

   # collect results and analyze
    logger.info(f'Collecting data into dataframes to scrutinize...')

    seq_len = np.array([len(_s) for _s in seq_data.seq], dtype=int)

    df_all = seq_data.to_df(seq=True, has=True, res_count=True)

    if idx_prefix:
        df_all['srcfile'] = df_all['file']
        df_all['srcid'] = df_all['id']
        df_all['file'] = [f'{df_all.iloc[_i]["idx"]}_{misc.str_deblank(Path(df_all.iloc[_i]["file"]).name)}' for _i in range(len(df_all))]
        df_all['id'] = [f'{df_all.iloc[_i]["idx"]}_{df_all.iloc[_i]["id"].strip()}' for _i in range(len(df_all))]

    df_all = df_all.assign(
            lenSelected = seq_data.len_selected,
            lenDiffSeq = seq_data.len - seq_len,
            # fields for resolving duplicates
            unknownSeq = False,
            conflictSeq = seq_data.len != seq_len,
            duplicateSeq = np.full(recap.num_seqs, False, dtype=bool),
            idxSameSeq = [0] * recap.num_seqs,
            idxSameSeqVal = [0] * recap.num_seqs,
            conflictVal = False,
            # fields for saving pkl and lib
            save2lib = seq_data.len_selected,
            dataset = [''] * recap.num_seqs, # 'train' or 'valid'
            )

    count_all_residues(df_all, recap=recap, prefix='raw_')

    if seq2upper:
        logger.info('Converting to upper cases...')
        df_all.seq = misc.mpi_map(str.upper, tqdm(df_all.seq, desc='Convert2Upper'), quiet=True )

    if seq2dna:
        logger.critical(f'seq2dna is not implemented yet!!!')

    if check_unknown:
        check_unknown_residues(df_all, recap=recap)

    if check_duplicate:
        check_duplicate_keyvals(df_all, recap=recap, keys='seq', vals=None)

    # kwargs.update({'save_fmts': ['seq', 'ctmat']})
    # save_all_files(df_all, recap=recap, save_lib=save_lib,
    #         save_dir=save_dir, save_name=save_name,
    #         save_pkl=save_pkl, save_fasta=save_fasta,
    #         save_csv=save_csv, save_individual=save_individual,
    #         save_genre=save_fmts, named_after=named_after,
    #         # save_unknown=True, save_duplicate=True, save_conflict=True,
    #         save_json=True, **kwargs)
    save_all_files(df_all, recap=recap, tqdm_disable=tqdm_disable, **kwargs)


def gather_upp2021(data_names=None, data_dir='./', beam_size=1000, bp_cutoff=0.0,
            save_dir=None, save_pickle=True, save_pkl=True, save_lib=True,
            tqdm_disable=False, **kwargs):
    """ build the 2021 upp data from paddle"""

    import rna_utils

    data_dir = Path(data_dir)
    save_dir = Path(save_dir) if save_dir else data_dir
    save_dir.mkdir(parents=True, exist_ok=True)

    if data_names is None:
        data_names = ['train', 'valid', 'test', 'predict']
    elif isinstance(data_names, str):
        data_names = [data_names]

    for fname in data_names:
        logger.info(f'Reading data type: {fname}...')
        seqsdata = molstru.SeqsData(data_dir / (fname + '.txt'),
                fmt='fasta', dbn=True, upp=(fname not in ['predict', 'test']))
        logger.info(f'Successfully parsed {len(seqsdata.seq)} sequences')

        # compute dbn and bpp with linear_fold and linear_partition
        partial_func = functools.partial(rna_utils.linear_fold_v, beam_size=beam_size)

        dbn_v = misc.mpi_map(partial_func, seqsdata.seq)

        # [0]: dbn, [1]: energy
        dbn_v = [_v[0] for _v in dbn_v]

        partial_func = functools.partial(rna_utils.linear_partition_c, beam_size=beam_size, bp_cutoff=bp_cutoff)
        # [0]: partition function, [1]: base pairing probabilities
        bpp_c = misc.mpi_map(partial_func, seqsdata.seq)
        upp_c = []
        for i, bpp in enumerate(bpp_c):
            if len(bpp[1]):
                upp_c.append(molstru.bpp2upp(np.stack(bpp[1], axis=0), seqsdata.len[i]))
            else:
                upp_c.append(np.zeros(seqsdata.len[i], dtype=np.float32))
                logger.info(f'No bpp_c found for seq: {seqsdata.seq[i]}')
        # upp_c = itertools.starmap(mol_stru.bpp2upp, zip(bpp_c, seqsdata.len))

        partial_func = functools.partial(rna_utils.linear_partition_v, beam_size=beam_size, bp_cutoff=bp_cutoff)
        bpp_v = misc.mpi_map(partial_func, seqsdata.seq)

        upp_v = []
        for i, bpp in enumerate(bpp_v):
            if len(bpp[1]):
                upp_v.append(molstru.bpp2upp(np.stack(bpp[1], axis=0), seqsdata.len[i]))
            else:
                upp_v.append(np.zeros(seqsdata.len[i], dtype=np.float32))
                logger.info(f'No bpp_v found for seq: {seqsdata.seq[i]}')
        # upp_c = itertools.starmap(mol_stru.bpp2upp, zip(bpp_c, seqsdata.len))

        upp_pred = [np.stack((upp_c[_i], upp_v[_i]), axis=1) \
                        for _i in range(len(upp_c))]

        df = seqsdata.to_df(seq=True, ct=True, upp=True, has=True, res_count=True)
        csv_file = save_dir / (fname + '.csv')
        logger.info(f'Storing csv: {csv_file}')
        df.to_csv(csv_file, index=False, float_format='%8.6f')

        if save_pickle and save_pkl:
            midata = dict(id=seqsdata.id, idx=seqsdata.idx, len=seqsdata.len, seq=seqsdata.seq,
                        dbn=dbn_v if len(dbn_v) else seqsdata.dbn,
                        file=seqsdata.file)

            if len(upp_pred):
                logger.info('Add predicted upp values as extra in midata')
                midata['extra'] = upp_pred

            if fname not in ['test', 'predict']:
                midata['upp'] = seqsdata.upp

            pkl_file = save_dir / (fname + '.pkl')
            logger.info(f'Storing pickle: {pkl_file}')
            gwio.pickle_squeeze(midata, pkl_file, fmt='lzma')

        if save_lib:
            lib_dir = save_dir / fname
            lib_dir.mkdir(exist_ok=True)
            logger.info(f'Saving lib files in: {lib_dir}')

            with tqdm(total=len(seqsdata.seq), disable=tqdm_disable) as prog_bar:
                for i in range(len(seqsdata.seq)):
                    lib_file = lib_dir / f'{i + 1:d}.input.fasta'
                    with lib_file.open('w') as iofile:
                        iofile.writelines('\n'.join(
                            seqsdata.get_fasta_lines(i, dbn=True, upp=False)
                        ))

                    if fname not in ['test', 'predict']:
                        lib_file = lib_dir / f'{i + 1:d}.label.upp'
                        np.savetxt(lib_file, seqsdata.upp[i], '%8.6f')
                        # with lib_file.open('w') as iofile:
                        #     iofile.writelines('\n'.join(seqsdata.upp[i].astype(str)))

                    if (i + 1) % 10 == 0:
                        prog_bar.update(10)
                prog_bar.update((i + 1) % 10)


if __name__ == '__main__':

    misc.argv_fn_caller(sys.argv[1:]) # module=sys.modules[__name__], verbose=1)
