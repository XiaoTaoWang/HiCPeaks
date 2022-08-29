# -*- coding: utf-8 -*-
"""
Created on Wed Sep 26 20:06:27 2018

@author: XiaoTao Wang
"""

import numpy as np
from scipy.special import ndtr

def apa_submatrix(M, pos, w=5):
    
    Len = M.shape[0]

    apa = []
    for i, j in pos:
        if (i-w>=0) and (i+w+1<=Len) and (j-w>=0) and (j+w+1<=Len):
            tmp = M[i-w:i+w+1, j-w:j+w+1].toarray()
            mask = np.isnan(tmp)
            if mask.sum() > 0:
                continue
            if tmp.mean()==0:
                continue
            
            tmp = tmp / tmp.mean()
            apa.append(tmp)
    
    return apa

def apa_analysis(apa, w=5, cw=3):
    
    # remove outliers
    mean_arr = np.r_[[np.mean(arr) for arr in apa]]
    p99 = np.percentile(mean_arr, 99)
    p1 = np.percentile(mean_arr, 1)
    mask = (mean_arr < p99) & (mean_arr > p1)
    avg = apa[mask].mean(axis=0)
    lowerpart = avg[-cw:,:cw]
    upperpart = avg[:cw,-cw:]
    maxi = upperpart.mean() * 5
    ## APA score
    score = avg[w,w] / lowerpart.mean()
    ## z-score
    z = (avg[w,w] - lowerpart.mean()) / lowerpart.std()
    p = 1 - ndtr(z)
    
    return avg, score, z, p, maxi