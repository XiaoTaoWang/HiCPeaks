# -*- coding: utf-8 -*-
"""
Created on Sat May 19 16:00:16 2018

@author: XiaoTao Wang
"""

import os, sys, tempfile, time, logging
import pandas as pd
from cooler.util import binnify
from cooler.io import create_from_unordered, sanitize_records

log = logging.getLogger(__name__)
    

def toCooler(outfil, data_path, res, assembly, chroms=['#','X'], symmetric=True,
             count_type=float, cache_dir=None, delete_cache=True):
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
    tl = time.strftime('%Y%m%d%H%M%S', time.localtime(time.time()))
    kw = {'prefix':'pixels', 'suffix':tl, 'dir':cache_dir}
    pixel_fil = tempfile.mktemp(**kw)
    
    # write the pixel file
    chromsizes = {}
    with open(pixel_fil, 'wb') as out:
        with open(data_path, 'rb') as source:
            for line in source:
                c1, p1, c2, p2, count = line.rstrip().split()
                c1 = c1.lstrip('chr')
                c2 = c2.lstrip('chr')
                check1 = ((not chroms) or (c1.isdigit() and ('#' in chroms)) or (c1 in chroms))
                check2 = ((not chroms) or (c2.isdigit() and ('#' in chroms)) or (c2 in chroms))
                if (not check1) or (not check2):
                    continue
                ip_1, ip_2 = int(p1)+res, int(p2)+res
                if c1 in chromsizes:
                    if ip_1 > chromsizes[c1]:
                        chromsizes[c1] = ip_1
                else:
                    chromsizes[c1] = ip_1
                if c2 in chromsizes:
                    if ip_2 > chromsizes[c2]:
                        chromsizes[c2] = ip_2
                else:
                    chromsizes[c2] = ip_2
                newline = [c1, p1, str(ip_1), c2, p2, str(ip_2), count]
                out.write('\t'.join(newline)+'\n')
    
    # generate bin table
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
    bins = binnify(chromsizes, res)
    
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
    
    tril_action = 'drop' if symmetric else 'reflect'
    pipeline = sanitize_records(bins, schema='bg2', is_one_based=False,
                                tril_action=tril_action, sort=True)
    
    reader = pd.read_table(
                    pixel_fil,
                    usecols=[input_field_numbers[name] for name in input_field_names],
                    names=input_field_names,
                    dtype=input_field_dtypes,
                    iterator=True,
                    chunksize=int(40e6))
    
    create_from_unordered(
            outfil,
            bins,
            map(pipeline, reader),
            columns=output_field_names,
            dtypes=output_field_dtypes,
            assembly=assembly,
            mergebuf=int(40e6),
            ensure_sorted=False)   
    
    
    
    
    
        