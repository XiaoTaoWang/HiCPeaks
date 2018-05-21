# -*- coding: utf-8 -*-
"""
Created on Sat May 19 16:00:16 2018

@author: XiaoTao Wang
"""

import glob, re, os, sys, cPickle, tempfile, time, zipfile, logging
import numpy as np
from cooler.io import create_from_unordered, sanitize_records, sanitize_pixels

log = logging.getLogger(__name__)

def _extractChrLabel(filename):
    
    """
    Extract chromosome label from a file name.
    """
    # Full filename including path prefix
    _, interName = os.path.split(filename)
    regexp = 'chr(.*).txt'
    search_results = re.search(regexp, filename)
    label = search_results.group(1)
    
    # Remove leading zeroes.
    if label.isdigit():
        label = str(int(label))
    
    return label

def _scanFolder(folder, chroms):
    
    """
    Create a map from chromosome labels to file names under the folder.
    """
    oriFiles = glob.glob(os.path.join(folder, 'chr*.txt'))
    
    # Read chromosome labels
    labels = []
    interFiles = [] # Depend on user's selection
    for i in oriFiles:
        label = _extractChrLabel(i)
        if ((not chroms) or (label.isdigit() and '#' in chroms) or (label in chroms)):
            labels.append(label)
            interFiles.append(i)
        
    # Map from labels to files
    Map = dict(zip(labels, interFiles))
        
    return Map
    

def toCooler(outfil, data_path, resolution, chroms=['#','X'], cache_dir=None,
            delete_cache=True):
    """
    Create a Cooler from TXT or NPZ format Hi-C data.
    
    Parameters
    ----------
    outfil : str
        Path of the output Cooler file.
        
    data_path : str
        If your Hi-C data are stored in *NPZ* format, this string should point
        to the npz file. Otherwise in the *TXT* format case, data from different
        chromosomes must be stored separately and placed within a same folder,
        and this string should point to the path of the folder.
        
    resolution : int
        Resolution / Bin size of the Hi-C data in base pairs.
    
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
        Whether to delete temporary files when finished.
    
    
    """
    outfil = os.path.abspath(os.path.expanduser(outfil))
    if os.path.exists(outfil):
        log.error('Cooler file {} already exists, exit ...')
        sys.exit(1)
        
    ## ready for data loading
    tl = time.strftime('%Y%m%d%H%M%S', time.localtime(time.time()))
    kw = {'prefix':'bins', 'suffix':tl, 'dir':cache_dir}
    bin_fil = tempfile.mktemp(**kw)
    kw = {'prefix':'pixels', 'suffix':tl, 'dir':cache_dir}
    pixel_fil = tempfile.mktemp(**kw)
    pool = {}
    chromlist = []
    if data_path.endswith('.npz'):
        lib = np.load(data_path)
        for i in lib.files:
            if ((not chroms) or (i.isdigit() and '#' in chroms) or (i in chroms)):
                 chromlist.append(i)
    else:
        Map = _scanFolder(data_path)
        for i in Map:
            chromlist.append(i)
    
    # sort chromosome labels
    tmp = map(str, sorted(map(int, [i for i in chromlist if i.isdigit()])))
    nondigits = [i for i in chromlist if not i.isdigit()]
    for i in ['X','Y','M']:
        if i in nondigits:
            tmp.append(nondigits.pop(nondigits.index(i)))
    chromlist = tmp + sorted(nondigits)
    
    # write the bin table
    
        