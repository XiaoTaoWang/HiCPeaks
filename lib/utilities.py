# -*- coding: utf-8 -*-
"""
Created on Sat May 19 16:00:16 2018

@author: XiaoTao Wang
"""

import glob, re, os, sys, cPickle, tempfile, time, zipfile, logging
import numpy as np
from cooler.io import create_from_unordered, sanitize_records, sanitize_pixels

log = logging.getLogger(__name__)

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
    