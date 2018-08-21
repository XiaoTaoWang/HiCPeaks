# -*- coding: utf-8 -*-
"""
Created on Wed Jun 27 16:48:22 2018

@author: XiaoTao Wang
"""

import os, sys, tempfile, time, logging, h5py
import numpy as np
import pandas as pd
from scipy import sparse
from cooler.util import binnify
from cooler.io import create, parse_cooler_uri
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
        data must be stored separately (1_1.txt, 1_2.txt, ..., 2_2.txt, 2_3.txt, ...),
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
        if os.path.exists(self.outfil):
            log.error('Cooler file {} already exists, exit ...')
            sys.exit(1)
        self.chroms = set(chroms)
        data = datasets

        ## Ready for data loading
        log.info('Fetch chromosome sizes from UCSC ...')
        chromsizes = fetchChromSizes(assembly, self.chroms)
        chromlist = chromsizes.keys()
        # sort chromosome labels
        tmp = map(str, sorted(map(int, [i for i in chromlist if i.isdigit()])))
        nondigits = [i for i in chromlist if not i.isdigit()]
        for i in ['X','Y','M']:
            if i in nondigits:
                tmp.append(nondigits.pop(nondigits.index(i)))
        self.chromlist = tmp + sorted(nondigits)
        lengths = [chromsizes[i] for i in self.chromlist]
        self.chromsizes = pd.Series(data=lengths, index=self.chromlist)
        log.info('Done!')

        ## We don't read data into memory at this point.
        self.Map = {}
        for res in data:
            if data[res].endswith('.npz'):
                self.Map[res] = {}
                lib = np.load(data[res])
                for i in lib.files:
                    tmp = i.split('_')
                    if len(tmp)!=2:
                        continue
                    c1, c2 = tmp
                    check1 = ((not self.chroms) or (c1.isdigit() and '#' in self.chroms) or (c1 in self.chroms))
                    check2 = ((not self.chroms) or (c2.isdigit() and '#' in self.chroms) or (c2 in self.chroms))
                    if check1 and check2:
                        self.Map[res][(c1,c2)] = lib
            else:
                self.Map[res] = self._scanFolder(data[res])

        self._intertype = np.dtype({'names':['bin1', 'bin2', 'IF'],
                                    'formats':[np.int, np.int, np.float]})
        
        log.info('Extract and save data into cooler format for each resolution ...')
        for res in self.Map:
            log.info('Current resolution: %dbp', res)
            bin_cumnums = self.binCount(res)
            log.info('Generate bin table ...')
            bintable = binnify(self.chromsizes, res)
            byres = self.Map[res]
            pixels = self._generator(byres, bin_cumnums)
            if os.path.exists(self.outfil):
                append = True
            else:
                append = False
            cooler_uri = '{}::{}'.format(self.outfil, res)
            create(cooler_uri, bintable, pixels, assembly=assembly, append=append, boundscheck=False,
                   triucheck=False, dupcheck=False, ensure_sorted=False)
                   
        log.info('Done!')
    
    def  _generator(self, byres, bin_cumnums):

        for i in range(self.chromsizes.size):
            for j in range(i, self.chromsizes.size):
                c1, c2 = self.chromlist[i], self.chromlist[j]
                if (c1,c2) in byres:
                    ci, cj = i, j
                else:
                    if (c2,c1) in byres:
                        c1, c2 = c2, c1
                        ci, cj = j, i
                    else:
                        continue
                
                log.debug('Current chromosome pairs: {}-{}'.format(c1,c2))
                
                if type(byres[(c1,c2)])==str:
                    data = np.loadtxt(byres[(c1,c2)], dtype=self._intertype)
                else:
                    data = byres[(c1,c2)][(c1,c2)]
                x, y = data['bin1'], data['bin2']
                # Fast guarantee triu matrix
                if ci > cj:
                    x, y = y, x
                    ci, cj = cj, ci
                
                xLen = x.max() + 1
                yLen = y.max() + 1
                if ci != cj:
                    tmp = sparse.csr_matrix((data['IF'], (x,y)), shape=(xLen, yLen))
                else:
                    Len = max(xLen, yLen)
                    tmp = sparse.csr_matrix((data['IF'], (x,y)), shape=(Len, Len))
                    tmp = sparse.lil_matrix(tmp)
                    tmp[y,x] = tmp[x,y]
                    tmp = sparse.triu(tmp)
                
                x, y = tmp.nonzero()
                if ci > 0:
                    x = x + bin_cumnums[ci-1]
                if cj > 0:
                    y = y + bin_cumnums[cj-1]
                
                data = tmp.data

                current = pd.DataFrame({'bin1_id':x, 'bin2_id':y, 'count':data},
                                       columns=['bin1_id', 'bin2_id', 'count'])

                yield current

    def _scanFolder(self, folder):
        """
        Create a map from chromosome pairs to file names under the folder.
        """
        import glob

        oriFiles = glob.glob(os.path.join(folder, '*_*.txt'))
        
        pairs = []
        interFiles = []
        for i in oriFiles:
            _, interName = os.path.split(i) # Full filename including path prefix
            tmp = interName.rstrip('.txt').split('_')
            if len(tmp)!=2:
                continue
            c1, c2 = tmp
            check1 = ((not self.chroms) or (c1.isdigit() and '#' in self.chroms) or (c1 in self.chroms))
            check2 = ((not self.chroms) or (c2.isdigit() and '#' in self.chroms) or (c2 in self.chroms))
            if check1 and check2:
                pairs.append((c1,c2))
                interFiles.append(i)

        Map = dict(zip(pairs, interFiles))
        
        return Map
    
    def binCount(self, res):

        def _each(chrom):
            clen = self.chromsizes[chrom]
            n_bins = int(np.ceil(clen / res))
            return n_bins+1
        
        data = [_each(c) for c in self.chromsizes.index]
        n_bins = pd.Series(data, index=self.chromsizes.index)
        cum_n = n_bins.cumsum()

        return cum_n

    
def balance(cool_uri, nproc=1, chunksize=int(1e7), mad_max=5, min_nnz=10,
            min_count=0, ignore_diags=1, tol=1e-5, max_iters=200):
    """
    Cooler contact matrix balancing.
    
    Parameters
    ----------
    cool_uri : str
        URI of cooler group.
    nproc : int
        Number of processes. (Default: 1)
        
    """
    cool_path, group_path = parse_cooler_uri(cool_uri)
    # pre-check the weight column
    with h5py.File(cool_path, 'r') as h5:
        grp = h5[group_path]
        if 'weight' in grp['bins']:
            del grp['bins']['weight'] # Overwrite the weight column
    
    log.info('Balancing {0}'.format(cool_uri))
    
    clr = Cooler(cool_uri)
    
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
        grp = h5[group_path]
        # add the bias column to the file
        h5opts = dict(compression='gzip', compression_opts=6)
        grp['bins'].create_dataset('weight', data=bias, **h5opts)
        grp['bins']['weight'].attrs.update(stats)
    
