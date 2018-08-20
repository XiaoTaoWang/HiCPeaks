# -*- coding: utf-8 -*-
"""
Created on Wed Jun 27 16:48:22 2018

@author: XiaoTao Wang
"""

import os, sys, tempfile, time, logging, h5py
import pandas as pd
from cooler.util import binnify
from cooler.io import create_from_unordered, sanitize_records
from cooler.api import Cooler
from cooler import ice
from multiprocess import Pool

log = logging.getLogger(__name__)

def fetchChromSizes(assembly, chroms):

    import subprocess
    
    chromsizes = {}
    pread = subprocess.Popen(['fetchChromSizes',assembly], stdout=subprocess.PIPE)
    inStream = pread.stdout
    for line in inStream:
        parse = line.rstrip().split()
        c, s = parse[0].lstrip('chr'), parse[1]
        check = ((not chroms) or (c.isdigit() and ('#' in chroms)) or (c in chroms))
        if check:
            chromsizes[c] = int(s)
    
    pread.communicate()
    
    return chromsizes

class Genome(object):
    """
    Load bin-level Hi-C data of TXT or NPZ format, and save it into cooler files.

    Parameters
    ----------
    datasets : dict, {resolution(int):data_path(str)}
        *resolution* should be in base-pair unit. *data_path* indicates the
        absolute data path.
        
        If your data are stored in NPZ format, *data_path* should point to the
        npz file. Or if your data are stored in TXT format, *data_path* should
        point to a folder, in which case all intra-chromosomal and inter-chromosomal
        data must be stored separately (1.txt, 2.txt, ..., 1_2.txt, 1_10.txt, ...),
        and data with the same resolution should be placed in the same folder.
        
        You can generate *NPZ* files in two ways:1.By *runHiC* pipeline. *runHiC*
        is a user-friendly command-line software developed by our lab for Hi-C
        data processing. Refer to the `link <https://github.com/XiaoTaoWang/HiC_pipeline>`_
        for more details. 2.By our hierarchical TAD detecting pipeline *hitad*. Provide
        TXT Hi-C data and run *hitad* with ``--npzpre`` specified. Refer to the
        `line <https://xiaotaowang.github.io/TADLib/hitad.html>`_ for more detals.
    
    outfil : str
        Path of the output Cooler file.
    
    assembly : str
        Genome assembly name.
    
    chroms : list
        List of chromosome labels. Only Hi-C data within the specified chromosomes
        will be included. Specially, '#' stands for chromosomes with numerical
        labels. If an empty list is provided, all chromosome data will be loaded.
        (Default: ['#', 'X'])

    """

    def __init__(self, datasets, outfil, assembly='hg38', chroms=['#','X']):

        self.outfil = os.path.abspath(os.path.expanduser(outfil))
        if os.path.exists(outfil):
            log.error('Cooler file {} already exists, exit ...')
            sys.exit(1)
        self.chroms = set(chroms)

        ## Ready for data loading
        log.info('Fetch chromosome sizes from UCSC ...')
        chromsizes = fetchChromSizes(assembly, chroms)
        chromlist = chromsizes.keys()
        # sort chromosome labels
        tmp = map(str, sorted(map(int, [i for i in chromlist if i.isdigit()])))
        nondigits = [i for i in chromlist if not i.isdigit()]
        for i in ['X','Y','M']:
            if i in nondigits:
                tmp.append(nondigits.pop(nondigits.index(i)))
        chromlist = tmp + sorted(nondigits)
        lengths = [chromsizes[i] for i in chromlist]
        self.chromsizes = pd.Series(data=lengths, index=chromlist)
        log.info('Done!')

        log.info('Generate the bin table ...')
        self.bins = binnify(chromsizes, res)
        log.info('Done!')

        ## We don't read data into memory at this point.
        



def toCooler(datasets, outfil, res, assembly, chroms=['#','X'],
             count_type=int, cache_dir=None, delete_cache=True):
    """
    Create a Cooler from TXT Hi-C data.
    
    Parameters
    ----------
    outfil : str
        Path of the output Cooler file.
        
    data_path : str
        Path of original contact matrix file (in TXT format).
        
    res : int
        Resolution / Bin size of the matrix in base pairs.
    
    assembly : str
        Genome assembly name.
    
    chroms : list
        List of chromosome labels. Only Hi-C data within the specified chromosomes
        will be included. Specially, '#' stands for chromosomes with numerical
        labels. If an empty list is provided, all chromosome data will be loaded.
        (Default: ['#', 'X'])
    
    cache_dir : str or None
        All intermediate or temporary files would be generated under this folder.
        If None, the folder returned by :py:func:`tempfile.gettempdir` will be
        used. (Default: None)
    
    delete_cache : Bool
        Whether to delete temporary files when finished. (Default: True)
    
    """
    outfil = os.path.abspath(os.path.expanduser(outfil))
    if os.path.exists(outfil):
        log.error('Cooler file {} already exists, exit ...')
        sys.exit(1)
        
    ## ready for data loading
    if cache_dir is None:
        cache_dir = tempfile.gettempdir()
    
    if not os.path.exists(cache_dir):
        os.makedirs(cache_dir)
    
    log.info('Fetch chromosome sizes from UCSC ...')
    chromsizes = fetchChromSizes(assembly, chroms)
    chromlist = chromsizes.keys()
    # sort chromosome labels
    tmp = map(str, sorted(map(int, [i for i in chromlist if i.isdigit()])))
    nondigits = [i for i in chromlist if not i.isdigit()]
    for i in ['X','Y','M']:
        if i in nondigits:
            tmp.append(nondigits.pop(nondigits.index(i)))
    chromlist = tmp + sorted(nondigits)
    lengths = [chromsizes[i] for i in chromlist]
    chromsizes = pd.Series(data=lengths, index=chromlist)
    log.info('Done!')

    log.info('Generate the bin table ...')
    bins = binnify(chromsizes, res)
    log.info('Done!')

    

    
        
    tl = time.strftime('%Y%m%d%H%M%S', time.localtime(time.time()))
    kw = {'prefix':'pixels', 'suffix':tl, 'dir':cache_dir}
    pixel_fil = tempfile.mktemp(**kw)
    
    log.info('Split raw matrix by chromosome ...')
    
    with open(data_path, 'rb') as source:
        file_pool = {}
        for line in source:
            c1, p1, c2, p2, count = line.rstrip().split()
            c1 = c1.lstrip('chr')
            c2 = c2.lstrip('chr')
            check1 = ((not chroms) or (c1.isdigit() and ('#' in chroms)) or (c1 in chroms))
            check2 = ((not chroms) or (c2.isdigit() and ('#' in chroms)) or (c2 in chroms))
            if (not check1) or (not check2):
                continue
            ip_1, ip_2 = min(int(p1)+res,chromsizes[c1]), min(int(p2)+res,chromsizes[c2])
            key = tuple(sorted([c1,c2]))
            if not key in file_pool:
                tmpfil = tempfile.mktemp(**{'prefix':'_'.join(key), 'suffix':tl,
                                            'dir':cache_dir})
                file_pool[key] = open(tmpfil, 'wb')
            if (c1,c2)==key:
                if c1==c2:
                    # make it upper triangular
                    if ip_1 > ip_2:
                        newline = [c2, p2, str(ip_2), c1, p1, str(ip_1), count]
                    else:
                        newline = [c1, p1, str(ip_1), c2, p2, str(ip_2), count]
                else:
                    newline = [c1, p1, str(ip_1), c2, p2, str(ip_2), count]
            else:
                newline = [c2, p2, str(ip_2), c1, p1, str(ip_1), count]
                
            file_pool[key].write('\t'.join(newline)+'\n')
        for key in file_pool:
            file_pool[key].flush()
            file_pool[key].close()
    
    log.info('Write the pixel table ...')
    with open(pixel_fil, 'wb') as out:
        for key in file_pool:
            records = set()
            with open(file_pool[key].name,'rb') as source:
                for line in source:
                    c1, s1, e1, c2, s2, e2, count = line.rstrip().split()
                    # remove duplicate records
                    if (s1,s2) in records:
                        continue
                    out.write(line)
                    records.add((s1,s2))

    if delete_cache:
        for key in file_pool:
            os.remove(file_pool[key].name)
    
    
    
    
    
    log.info('Create the Cooler file ...')
    # output fields
    output_field_names = ['bin1_id', 'bin2_id', 'count']
    output_field_dtypes = {'bin1_id': int, 'bin2_id': int, 'count': count_type}
    
    # input fields
    input_field_names = ['chrom1', 'start1', 'end1', 'chrom2', 'start2', 'end2', 'count']
    input_field_dtypes = {
            'chrom1': str, 'start1': int, 'end1': int,
            'chrom2': str, 'start2': int, 'end2': int,
            'count': count_type,
    }
    input_field_numbers = {
            'chrom1': 0, 'start1': 1, 'end1': 2,
            'chrom2': 3, 'start2': 4, 'end2': 5,
            'count': 6,
    }
    
    pipeline = sanitize_records(bins, schema='bg2', is_one_based=False,
                                tril_action='reflect', sort=True,
                                sided_fields=('end',))
    
    reader = pd.read_table(
                    pixel_fil,
                    usecols=[input_field_numbers[name] for name in input_field_names],
                    names=input_field_names,
                    dtype=input_field_dtypes,
                    iterator=True,
                    chunksize=int(1e7))
    
    create_from_unordered(
            outfil,
            bins,
            map(pipeline, reader),
            columns=output_field_names,
            dtypes=output_field_dtypes,
            assembly=assembly,
            mergebuf=int(1e7),
            delete_temp=True,
            ensure_sorted=False)
    
    if delete_cache:
        os.remove(pixel_fil)
    
def balance(cool_path, nproc=1, chunksize=int(1e7), mad_max=5, min_nnz=10,
            min_count=0, ignore_diags=1, tol=1e-5, max_iters=200):
    """
    Cooler contact matrix balancing.
    
    Parameters
    ----------
    cool_path : str
        Path of the cooler file.
    nproc : int
        Number of processes. (Default: 1)
        
    """
    # pre-check the weight column
    with h5py.File(cool_path, 'r') as h5:
        grp = h5['/']
        if 'weight' in grp['bins']:
            del grp['bins']['weight'] # Overwrite the weight column
    
    log.info('Balancing {0}'.format(cool_path))
    
    clr = Cooler(cool_path)
    
    try:
        if nproc > 1:
            pool = Pool(nproc)
            map_ = pool.imap_unordered
        else:
            map_ = map
        
        bias, stats = ice.iterative_correction(
                clr,
                chunksize=chunksize,
                cis_only=False,
                trans_only=False,
                tol=tol,
                min_nnz=min_nnz,
                min_count=min_count,
                blacklist=None,
                mad_max=mad_max,
                max_iters=max_iters,
                ignore_diags=ignore_diags,
                rescale_marginals=True,
                use_lock=False,
                map=map_)
    finally:
        if nproc > 1:
            pool.close()
    
    if not stats['converged']:
        log.error('Iteration limit reached without convergence')
        log.error('Storing final result. Check log to assess convergence.')
    
    with h5py.File(cool_path, 'r+') as h5:
        grp = h5['/']
        # add the bias column to the file
        h5opts = dict(compression='gzip', compression_opts=6)
        grp['bins'].create_dataset('weight', data=bias, **h5opts)
        grp['bins']['weight'].attrs.update(stats)
    
