# -*- coding: utf-8 -*-
"""
Created on Wed Jun 27 16:48:22 2018

@author: XiaoTao Wang
"""
from __future__ import division
import os, sys, tempfile, logging, h5py
import numpy as np
import pandas as pd
from scipy import sparse
from cooler.util import binnify, parse_cooler_uri
from cooler.reduce import CoolerMerger
from cooler.api import Cooler
from cooler import ice, create_cooler
from multiprocess import Pool

log = logging.getLogger(__name__)

def fetchChromSizes(assembly, chroms):

    import subprocess
    
    chromsizes = {}
    pread = subprocess.Popen(['fetchChromSizes',assembly], stdout=subprocess.PIPE)
    inStream = pread.stdout
    for line in inStream:
        parse = line.decode().rstrip().split()
        c, s = parse[0].lstrip('chr'), parse[1]
        check = ((not chroms) or (c.isdigit() and ('#' in chroms)) or (c in chroms))
        if check:
            chromsizes[c] = int(s)
    
    pread.communicate()
    
    return chromsizes

def readChromSizes(chromsizes_file, chroms):

    chromsizes = {}
    with open(chromsizes_file, 'r') as source:
        for line in source:
            parse = line.rstrip().split()
            c, s = parse[0].lstrip('chr'), parse[1]
            check = ((not chroms) or (c.isdigit() and ('#' in chroms)) or (c in chroms))
            if check:
                chromsizes[c] = int(s)
    
    return chromsizes

def create_from_unordered(cool_uri, bins, chunks, columns=None, dtypes=None, mergebuf=int(20e6),
                         delete_temp=True, temp_dir=None, **kwargs):
    """
    Create a Cooler in two passes via an external sort mechanism. In the first 
    pass, a sequence of data chunks are processed and sorted in memory and saved
    to temporary Coolers. In the second pass, the temporary Coolers are merged 
    into the output. This way the individual chunks do not need to be provided
    in any particular order.
    
    Parameters
    ----------
    cool_uri : str
        Path to Cooler file or URI to Cooler group. If the file does not exist,
        it will be created.
    bins : DataFrame
        Segmentation of the chromosomes into genomic bins. May contain 
        additional columns.
    chunks : iterable of DataFrames
        Sequence of chunks that get processed and written to separate Coolers 
        and then subsequently merged.
    columns : sequence of str, optional
        Specify here the names of any additional value columns from the input 
        besides 'count' to store in the Cooler. The standard columns ['bin1_id', 
        'bin2_id', 'count'] can be provided, but are already assumed and don't 
        need to be given explicitly. Additional value columns provided here will 
        be stored as np.float64 unless otherwised specified using `dtype`.
    dtypes : dict, optional
        Dictionary mapping column names to dtypes. Can be used to override the
        default dtypes of ``bin1_id``, ``bin2_id`` or ``count`` or assign
        dtypes to custom value columns. Non-standard value columns given in
        ``dtypes`` must also be provided in the ``columns`` argument or they
        will be ignored.
    assembly : str, optional
        Name of genome assembly.
    mode : {'w' , 'a'}, optional [default: 'w']
        Write mode for the output file. 'a': if the output file exists, append
        the new cooler to it. 'w': if the output file exists, it will be
        truncated. Default is 'w'.
    metadata : dict, optional
        Experiment metadata to store in the file. Must be JSON compatible.
    mergebuf : int, optional
        Maximum number of records to buffer in memory at any give time during 
        the merge step.
    delete_temp : bool, optional
        Whether to delete temporary files when finished. 
        Useful for debugging. Default is False.
    temp_dir : str, optional
        Create temporary files in this directory.

    See also
    --------
    sanitize_records
    sanitize_pixels

    """
    bins = bins.copy()
    bins['chrom'] = bins['chrom'].astype(object)

    tf = tempfile.NamedTemporaryFile(
                suffix='.multi.cool', 
                delete=delete_temp,
                dir=temp_dir)
        
    uris = []
    for i, chunk in enumerate(chunks):
        uri = tf.name + '::' + str(i)
        uris.append(uri)
        log.info('Writing chunk {}: {}'.format(i, uri))
        create_cooler(uri, bins, chunk, columns=columns, mode='a', boundscheck=False,
                      triucheck=False, dupcheck=False, ensure_sorted=False, ordered=True,
                      dtypes=dtypes)
        
    chunks = CoolerMerger([Cooler(uri) for uri in uris], mergebuf)

    log.info('Merging into {}'.format(cool_uri))
    create_cooler(cool_uri, bins, chunks, columns=columns, dtypes=dtypes, ordered=True,
                  **kwargs)


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
    
    dtype : {'int', float}
        The desired data type for your contact matrices.

    """
    def __init__(self, datasets, outfil, assembly='hg38', chromsizes_file=None, chroms=['#','X'], onlyIntra=True,
        dtype='int'):

        self.outfil = os.path.abspath(os.path.expanduser(outfil))
        if os.path.exists(self.outfil):
            log.error('Cooler file {} already exists, exit ...'.format(self.outfil))
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
        tmp = list(map(str, sorted(map(int, [i for i in chromlist if i.isdigit()]))))
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
                                    'formats':[np.int32, np.int32, np.float64]})
        
        log.info('Extract and save data into cooler format for each resolution ...')
        for res in self.Map:
            log.info('Current resolution: {}bp'.format(res))
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
                mode = 'a'
            else:
                mode = 'w'
            if dtype == 'int':
                dtypes = {'count': np.int32}
            else:
                dtypes = {'count': np.float64}
            cooler_uri = '{}::{}'.format(self.outfil, res)
            if self.onlyIntra:
                create_cooler(cooler_uri, bintable, pixels, assembly=assembly, mode=mode,
                       boundscheck=False, triucheck=False, dupcheck=False, ensure_sorted=False,
                       ordered=True, metadata={'onlyIntra':str(self.onlyIntra)}, dtypes=dtypes)
            else:
                create_from_unordered(cooler_uri, bintable, pixels, assembly=assembly,
                                      mode=mode, metadata={'onlyIntra':str(self.onlyIntra)},
                                      delete_temp=True, boundscheck=False, triucheck=False,
                                      dupcheck=False, ensure_sorted=False, dtypes=dtypes)
            
    
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
            return n_bins
        
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
    with h5py.File(cool_path, 'r+') as h5:
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

def find_chrom_pre(chromlabels):

    ini = chromlabels[0]
    if ini.startswith('chr'):
        return 'chr'
    
    else:
        return ''
 
def _parse_peakfile(filpath, skip=1):
    """
    Generate a peak annotation table.
    """
    D = {}
    with open(filpath, 'r') as source:
        for i, line in enumerate(source):
            if i < skip:
                continue
            parse = line.rstrip().split()
            chrom = parse[0]
            info = (int(parse[1]), int(parse[2]), int(parse[4]), int(parse[5]))
            if chrom in D:
                D[chrom].append(info)
            else:
                D[chrom] = [info]
    
    # consistent chrom labels
    keys = list(D.keys())
    pre = find_chrom_pre(keys)
    new = {}
    for chrom in D:
        k = chrom.lstrip(pre)
        new[k] = D[chrom]

    return new

def combine_annotations(byres, good_res=10000, mindis=100000, max_res=10000):
    """
    Combine peak annotations at different resolutions.

    Parameters
    ----------
    byres : dict
        Peak annotations at different resolutions. The keys are integer resolutions in base pairs,
        and the values are also dicts with peak annotations stored by chromosomes.
    
    good_res : int
        Peaks detected at finer resolutions (less than this value) are likely to be false
        positives if there are no peak annotations at coarser resolutions in the neighborhood.
        We keep these peaks only if the two loci are <mindis apart. (Default: 10000)
    
    mindis : int
        See good_res. (Default: 100000)
    
    max_res : int
        Allowed largest resolution for output, i.e., only peaks originally at this or less than
        this resolution will be outputed. (Default: 10000)
    
    Return
    ------
    peak_list : list
        Final peak list.
    """
    from scipy.spatial import distance_matrix

    thre1 = 2 * max_res
    thre2 = 5 * max_res
    if len(byres)==1:
        peak_list = []
        for r in byres:
            for c in byres[r]:
                for p in byres[r][c]:
                    tmp = (c,) + p[:2] + (c,) + p[2:]
                    peak_list.append(tmp)
        return peak_list
    
    reslist = sorted(byres)

    peak_list = set()
    record = set()
    for i in range(len(reslist)-1):
        tmp1 = byres[reslist[i]]
        for j in range(i+1,len(reslist)):
            tmp2 = byres[reslist[j]]
            for c in tmp1:
                if c in tmp2:
                    ref = [(t[0],t[2]) for t in tmp2[c]]
                else:
                    ref = []
                for p in tmp1[c]:
                    key = (c,) + p[:2] + (c,) + p[2:]
                    if key in record:
                        continue
                    if not len(ref):
                        if (reslist[i]<=max_res) and ((reslist[i]>=good_res) or (p[2]-p[0] <= mindis)):
                            peak_list.add(key)
                        continue
                    dis = distance_matrix([(p[0],p[2])], ref).ravel()
                    if reslist[i]<thre1 and reslist[j]<thre1:
                        mask = dis <= thre1
                    else:
                        mask = dis <= thre2
                    if mask.sum() > 0:
                        peak_list.add(key)
                        for idx in np.where(mask)[0]:
                            record.add((c,)+tmp2[c][idx][:2]+(c,)+tmp2[c][idx][2:])
                    else:
                        if (reslist[i]<=max_res) and ((reslist[i]>=good_res) or (p[2]-p[0] <= mindis)):
                            peak_list.add(key)
    
    for c in byres[reslist[-1]]:
        for p in byres[reslist[-1]][c]:
            key = (c,) + p[:2] + (c,) + p[2:]
            if (not key in record):
                if (reslist[-1]<=max_res) and ((reslist[-1]>=good_res) or (p[2]-p[0] <= mindis)):
                    peak_list.add(key)
    
    peak_list = sorted(peak_list)
    
    return peak_list
