# -*- coding: utf-8 -*-
"""
Created on Wed Sep 26 20:06:27 2018

@author: XiaoTao Wang
"""

import numpy as np
from scipy.special import ndtr

def apa_submatrix(M, Bool, p1L, w=5, t=15):
    
    Len = M.shape[0]
    ref = np.triu(Bool, k=t)
    part1 = np.zeros(M.shape, dtype=bool)
    part1[:p1L, :p1L] = True
    part2 = np.zeros(M.shape, dtype=bool)
    part2[p1L:, p1L:] = True
    part3 = np.zeros(M.shape, dtype=bool)
    part3[:p1L, p1L:] = True

    apa = []
    for tP in [part1, part2, part3]:
        submask = ref & tP
        x, y = np.where(submask)
        for i, j in zip(x, y):
            if (i-w>=0) and (i+w+1<=Len) and (j-w>=0) and (j+w+1<=Len):
                check = tP[i-w:i+w+1, j-w:j+w+1]
                '''
                if check.sum() != check.size:
                    continue
                '''
                notcheck = np.logical_not(check)
                tmp = M[i-w:i+w+1, j-w:j+w+1]
                tmp[notcheck] = 0
                if tmp.mean()==0:
                    continue
                tmp = tmp / tmp.mean()
                apa.append(tmp)
    
    return apa

def apa_analysis(apa, w=5, cw=3):
    
    avg = apa.mean(axis=0)
    lowerpart = avg[-cw:,:cw]
    upperpart = avg[:cw,-cw:]
    maxi = upperpart.mean() * 5
    ## APA score
    score = avg[w,w] / lowerpart.mean()
    ## z-score
    z = (avg[w,w] - lowerpart.mean()) / lowerpart.std()
    p = 1 - ndtr(z)
    
    return avg, score, z, p, maxi