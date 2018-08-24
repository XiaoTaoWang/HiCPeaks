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

def readChromSizes(chromsizes_file, chroms):

    chromsizes = {}
    with open(chromsizes_file, 'rb') as source:
        for line in source:
            parse = line.rstrip().split()
            c, s = parse[0].lstrip('chr'), parse[1]
            check = ((not chroms) or (c.isdigit() and ('#' in chroms)) or (c in chroms))
            if check:
                chromsizes[c] = int(s)
    
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
        Genome assembly name. (Default: hg38)

    chromsizes_file : str
        Path to the file containing chromosome size information.
    
    chroms : list
        List of chromosome labels. Only Hi-C data within the specified chromosomes
        will be included. Specially, '#' stands for chromosomes with numerical
        labels. If an empty list is provided, all chromosome data will be loaded.
        (Default: ['#', 'X'])

    onlyIntra : bool
        If specified, only include intra-chromosomal data.

    """
    def __init__(self, datasets, outfil, assembly='hg38', chromsizes_file=None, chroms=['#','X'], onlyIntra=True):

        self.outfil = os.path.abspath(os.path.expanduser(outfil))
        if os.path.exists(self.outfil):
            log.error('Cooler file {} already exists, exit ...')
            sys.exit(1)
        self.chroms = set(chroms)
        self.onlyIntra = onlyIntra
        data = datasets

        ## Ready for data loading
        if not chromsizes_file is None:
            chromsizes_path = os.path.abspath(os.path.expanduser(chromsizes_file))
            log.info('Read chromosome sizes from {}'.format(chromsizes_path))
            chromsizes = readChromSizes(chromsizes_path, self.chroms)
        else:
            log.info('Fetch chromosome sizes from UCSC ...')
            chromsizes = fetchChromSizes(assembly, self.chroms)
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
        log.info('Done')

        ## We don't read data into memory at this point.
        ## Waiting for more robust conditions, here I assume there is no sign '_' in any chromosome labels.
        self.Map = {}
        for res in data:
            if data[res].endswith('.npz'):
                self.Map[res] = {}
                lib = np.load(data[res])
                for i in lib.files:
                    if (not '_' in i) and ((not self.chroms) or (i.isdigit() and '#' in self.chroms) or (i in self.chroms)):
                        # Compatible with TADLib and old version of runHiC
                        c1 = c2 = i
                        self.Map[res][(c1,c2)] = lib
                    else:
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
            byres = self.Map[res]
            # Extract parts of chromsizes
            subset = []
            for c1, c2 in byres:
                subset.extend([c1,c2])
            subset = set(subset)
            Bool = [(i in subset) for i in self.chromsizes.index]
            chromsizes = self.chromsizes[Bool]
            bin_cumnums = self.binCount(chromsizes, res)
            log.info('Generate bin table ...')
            bintable = binnify(chromsizes, res)
            pixels = self._generator(byres, chromsizes, bin_cumnums)
            if os.path.exists(self.outfil):
                append = True
            else:
                append = False
            cooler_uri = '{}::{}'.format(self.outfil, res)
            create(cooler_uri, bintable, pixels, assembly=assembly, append=append,
                   boundscheck=False, triucheck=False, dupcheck=False, ensure_sorted=False,
                   metadata={'onlyIntra':str(self.onlyIntra)})
            
    
    def  _generator(self, byres, chromsizes, bin_cumnums):

        for i in range(chromsizes.size):
            for j in range(i, chromsizes.size):
                c1, c2 = chromsizes.index[i], chromsizes.index[j]
                if self.onlyIntra:
                    if c1!=c2:
                        continue
                if (c1,c2) in byres:
                    ci, cj = i, j
                else:
                    if (c2,c1) in byres:
                        c1, c2 = c2, c1
                        ci, cj = j, i
                    else:
                        continue
                
                if type(byres[(c1,c2)])==str:
                    data = np.loadtxt(byres[(c1,c2)], dtype=self._intertype)
                else:
                    # Make it compatible with TADLib and old version of runHiC
                    if c1!=c2:
                        data = byres[(c1,c2)][(c1,c2)]
                    else:
                        if c1 in byres[(c1,c2)].files:
                            data = byres[(c1,c2)][c1]
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
    
    def binCount(self, chromsizes, res):

        def _each(chrom):
            clen = chromsizes[chrom]
            n_bins = int(np.ceil(clen / res))
            return n_bins+1
        
        data = [_each(c) for c in chromsizes.index]
        n_bins = pd.Series(data, index=chromsizes.index)
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
        
        if clr.info['metadata']['onlyIntra']=='True':
            onlyIntra = True
        else:
            onlyIntra = False
        
        bias, stats = ice.iterative_correction(
                clr,
                chunksize=chunksize,
                cis_only=onlyIntra,
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
    
