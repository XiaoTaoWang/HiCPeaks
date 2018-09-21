# -*- coding: utf-8 -*-
"""
Created on Sun Sep 2 20:13:22 2018

@author: XiaoTao Wang
"""
import logging
import numpy as np
from scipy import sparse
from scipy.stats import poisson
from statsmodels.sandbox.stats.multicomp import multipletests

logger = logging.getLogger(__name__)

def pw_ww_pairs(pw, ww, maxww):

    pool = []
    for p, w in zip(pw, ww):
        for i in range(w, maxww+1):
            pool.append((i, p))
    pool = [(i[1],i[0]) for i in sorted(pool)]

    return pool

def lambdachunk(E):
    
    numbin = np.int(np.ceil(np.log(E.max()) / np.log(2) * 3 + 1))
    Pool = []
    for i in range(1, numbin + 1):
        if i == 1:
            lv = 0; rv = 1
        else:
            lv = np.power(2, ((i - 2)/3.))
            rv = np.power(2, ((i - 1)/3.))
        idx = np.where((E > lv) & (E < rv))[0]
        Pool.append((lv, rv, idx))
    
    return Pool


def hiccups(M, cM, B1, B2, IR, chromLen, Diags, cDiags, num, chrom, pw=[2], ww=[5],
            maxww=20, sig=0.1, sumq=0.01, double_fold=1.75, single_fold=2, maxapart=2000000,
            res=10000):

    # more codes for lower memory
    # use reference instead of creating new arrays
    extDiags_ref = []
    for i in range(num):
        OneDArray = Diags[i]
        extODA = np.zeros(chromLen - i + maxww*2)
        extODA[maxww:-maxww] = OneDArray
        extDiags_ref.append(extODA)
    
    extDiags = {maxww: extDiags_ref}
    for w in range(min(ww), maxww):
        temp = []
        for i in range(num):
            delta = maxww-w
            extODA = extDiags_ref[i][delta:-delta]
            temp.append(extODA)
        extDiags[w] = temp
    
    EDiags = []
    x = np.r_[sorted(IR)]
    for i in x:
        OneDArray = np.ones(chromLen - i) * IR[i]
        EDiags.append(OneDArray)
    
    EM = sparse.diags(EDiags, x, format = 'csr')

    extCDiags_ref = []
    extEDiags_ref = []
    for i in range(x.size):
        extODA_C = np.zeros(chromLen - x[i] + maxww*2)
        extODA_C[maxww:-maxww] = cDiags[i]
        extCDiags_ref.append(extODA_C)
        extODA_E = np.zeros(chromLen - x[i] + maxww*2)
        extODA_E[maxww:-maxww] = EDiags[i]
        extEDiags_ref.append(extODA_E)
    
    extCDiags = {maxww: extCDiags_ref}
    extEDiags = {maxww: extEDiags_ref}
    for w in range(min(ww), maxww):
        tempC = []
        tempE = []
        for i in range(x.size):
            delta = maxww - w
            extODA_C = extCDiags_ref[i][delta:-delta]
            tempC.append(extODA_C)
            extODA_E = extEDiags_ref[i][delta:-delta]
            tempE.append(extODA_E)
        extCDiags[w] = tempC
        extEDiags[w] = tempE
    
    p_w = pw_ww_pairs(pw, ww, maxww)
                
    ## Peak Calling ...    
    vxi, vyi = M.nonzero()
    Mask = ((vyi - vxi) >= min(ww)) & ((vyi - vxi) <= (maxapart // res))
    vxi = vxi[Mask]
    vyi = vyi[Mask]
    # Here the key indicates the color the original paper uses for corresponding backgrounds
    flocals = ['K', 'Y'] # Order is important
    bSV = {}; bEV = {}
    for pi in pw: # support multiple pw and ww values
        bSV[pi] = {}; bEV[pi] = {}
        for fl in flocals:
            bSV[pi][fl] = np.zeros(vxi.size)
            bEV[pi][fl] = np.zeros(vxi.size)
    
    logger.info('Chrom:{0}, Observed Contact Number: {1}'.format(chrom, vxi.size))

    RefIdx = {}; iniNum = {}
    for pi in pw:
        RefIdx[pi] = np.arange(vxi.size)
        iniNum[pi] = vxi.size
    
    totalNum = vxi.size
    
    logger.info('Chrom:{0}, Two local neighborhoods, two expected matrices ...'.format(chrom))
    bS = {}; bE = {}
    for fl in flocals:
        bS[fl] = sparse.csr_matrix((chromLen, chromLen))
        bE[fl] = sparse.csr_matrix((chromLen, chromLen))
    Reads = sparse.csr_matrix((chromLen, chromLen))
    limitCompute = False
    last_pi = last_wi = 0
    frozen_w = maxww
    for pi, wi in p_w:
        if wi > frozen_w:
            continue
        ps = 2 * pi + 1
        ws = 2 * wi + 1
        logger.info('Chrom:{0},    Peak width:{1}, Donut width:{2}'.format(chrom, pi, wi))
        P1 = set([(i,j) for i in range(wi-pi, ps+wi-pi) for j in range(wi-pi, ps+wi-pi)]) # Center Peak Region
        P_1 = set([(i,j) for i in range(wi+1, ws) for j in range(wi)])
        P_2 = set([(i,j) for i in range(wi+1, ps+wi-pi) for j in range(wi-pi, wi)])
        P2 = P_1 - P_2 # Lower-left Region

        ss = range(ws)
        Pool_Diags = {}
        Pool_EDiags = {}
        Pool_cDiags = {}
        for i in ss:
            for j in ss:
                bgloc = max(abs(i-wi), abs(j-wi)) # mark the radial location on background matrix
                if limitCompute:
                    if ((bgloc<=last_wi) and (bgloc>max(pi,last_pi))) or (bgloc<=min(pi,last_pi)):
                        continue
                Pool_Diags[(i,j)] = []
                Pool_EDiags[(i,j)] = []
                Pool_cDiags[(i,j)] = []

                for oi in range(num):
                    if oi + i - j >= 0:
                        starti = i
                        endi = i + chromLen - (oi + i - j)
                    else:
                        starti = i - (oi + i - j)
                        endi = starti + chromLen + (oi + i - j)
                    Pool_Diags[(i,j)].append(extDiags[wi][oi][starti:endi])
                for oi in range(x.size):
                    if x[oi] + i - j >= 0:
                        starti = i
                        endi = i + chromLen - (x[oi] + i - j)
                    else:
                        starti = i - (x[oi] + i - j)
                        endi = starti + chromLen + (x[oi] + i - j)
                    Pool_EDiags[(i,j)].append(extEDiags[wi][oi][starti:endi])
                    Pool_cDiags[(i,j)].append(extCDiags[wi][oi][starti:endi])

        for key in Pool_Diags:
            bgloc = max(abs(key[0]-wi), abs(key[1]-wi))
            cDiags_matrix = sparse.diags(Pool_cDiags[key], x + (key[0] - key[1]), format = 'csr')
            EDiags_matrix = sparse.diags(Pool_EDiags[key], x + (key[0] - key[1]), format = 'csr')
            if (key[0] != wi) and (key[1] != wi) and (key not in P1) and (key not in P2):
                if (not limitCompute) or (limitCompute and bgloc>last_wi) or (limitCompute and bgloc>pi and bgloc<=last_pi):
                    bS['K'] = bS['K'] + cDiags_matrix
                    bE['K'] = bE['K'] + EDiags_matrix
                else:
                    bS['K'] = bS['K'] - cDiags_matrix
                    bE['K'] = bE['K'] - EDiags_matrix
            if key in P2:
                if (not limitCompute) or (limitCompute and bgloc>last_wi) or (limitCompute and bgloc>pi and bgloc<=last_pi):
                    bS['K'] = bS['K'] + cDiags_matrix
                    bE['K'] = bE['K'] + EDiags_matrix
                    bS['Y'] = bS['Y'] + cDiags_matrix
                    bE['Y'] = bE['Y'] + EDiags_matrix
                else:
                    bS['K'] = bS['K'] - cDiags_matrix
                    bE['K'] = bE['K'] - EDiags_matrix
                    bS['Y'] = bS['Y'] - cDiags_matrix
                    bE['Y'] = bE['Y'] - EDiags_matrix
                if (not limitCompute) or (limitCompute and pi==min(pw) and bgloc>last_wi):
                    Reads = Reads + sparse.diags(Pool_Diags[key], np.arange(num) + (key[0] - key[1]), format = 'csr')
            
        limitCompute = True
        last_pi, last_wi = pi, wi
                
        Txi = vxi[RefIdx[pi]]
        Tyi = vyi[RefIdx[pi]]
        RNums = np.array(Reads[Txi, Tyi]).ravel()
        EIdx = RefIdx[pi][RNums >= 16]
        logger.info('Chrom:{0},    ({1},{2}) Valid Contact Number from This Loop: {3}'.format(chrom, pi, wi, EIdx.size))
        Valid_Ratio = EIdx.size/float(iniNum[pi])
        Exi = vxi[EIdx]
        Eyi = vyi[EIdx]
        for fl in flocals:
            bSV[pi][fl][EIdx] = np.array(bS[fl][Exi, Eyi]).ravel()
            bEV[pi][fl][EIdx] = np.array(bE[fl][Exi, Eyi]).ravel()
                
        RefIdx[pi] = RefIdx[pi][RNums < 16]
            
        iniNum[pi] = RefIdx[pi].size

        left_Ratio = iniNum[pi]/float(totalNum)

        logger.info('Chrom:{0},    ({1},{2}) Total Valid Ratio after This Loop: {3:.3f}'.format(chrom, pi, wi, 1-left_Ratio))
        
        if (Valid_Ratio < 0.3) and (wi >= max(ww)):
            logger.info('Chrom:{0},    ({1},{2}) Ratio of valid contact is too small, assign maximum donut width to {3} ...'.format(chrom, pi, wi, wi))
            frozen_w = wi
        
        if (left_Ratio < 0.03) and (wi >= max(ww)):
            logger.info('Chrom:{0},    ({1},{2}) Very few or no contacts are left, assign maximum donut width to {3} ...'.format(chrom, pi, wi, wi))
            frozen_w = wi
        
        if wi<frozen_w:
            logger.info('Chrom:{0},    ({1},{2}) {3} Contacts will get into next loop ...'.format(chrom, pi, wi, RefIdx[pi].size))
    
    pixel_table = {} # Store final peak list
    
    logger.info('Chrom:{0}, Poisson Models and Benjamini-Hochberg Correcting for lambda chunks ...'.format(chrom))
    Description = {'K': 'Donut backgrounds', 'Y': 'Lower-left backgrounds'}
    gaps = set(np.where(np.array(cM.sum(axis=1)).ravel() == 0)[0])
    for pi, wi in zip(pw,ww):
        xpos = {}; ypos = {}; Ovalues = {}
        Fold = {}; pvalues = {}; qvalues = {}
        for fl in flocals:
            logger.info('Chrom:{0},    Peak width:{1}, Donut width:{2}, {3} ...'.format(chrom, pi, wi, Description[fl]))
            Mask = (bEV[pi][fl] != 0) & (vyi - vxi >= wi)
            tmp = sparse.lil_matrix((chromLen, chromLen))
            tmp[vxi[Mask],vyi[Mask]] = bSV[pi][fl][Mask] / bEV[pi][fl][Mask]
            cEM = EM.multiply(tmp.tocsr())
            xi, yi = cEM.nonzero()
            Evalues = np.array(cEM[xi, yi]).ravel() * B1[xi] * B2[yi]
            Mask = Evalues > 0
            Evalues = Evalues[Mask]
            xi = xi[Mask]
            yi = yi[Mask]
            Ovalues[fl] = np.array(M[xi, yi]).ravel()
            Fold[fl] =  Ovalues[fl] / Evalues
            logger.info('Chrom:{0},    ({1},{2}), Valid contact number: {3}'.format(chrom, pi, wi, xi.size))
        
            pvalue = np.ones(xi.size)
            qvalue = np.ones(xi.size)
        
            logger.info('Chrom:{0},    ({1},{2}), Lambda chunking ...'.format(chrom, pi, wi))
            chunks = lambdachunk(Evalues)
            logger.info('Chrom:{0},    ({1},{2}), Number of chunks: {3}'.format(chrom, pi, wi, len(chunks)))
            for chunk in chunks:
                logger.debug('Chrom:{0},        ({1},{2}), lv: {3:.3g}, rv: {4:.3g}, Num: {5}'.format(chrom, pi, wi, chunk[0], chunk[1], chunk[2].size))
                if chunk[2].size > 0:
                    Poiss = poisson(chunk[1])
                    logger.debug('Chrom:{0},        ({1},{2}), Assign P values ...'.format(chrom, pi, wi))
                    chunkP = 1 - Poiss.cdf(Ovalues[fl][chunk[2]])
                    pvalue[chunk[2]] = chunkP
                    logger.debug('Chrom:{0},        ({1},{2}), Multiple testing ...'.format(chrom, pi, wi))
                    cResults = multipletests(chunkP, alpha = sig, method = 'fdr_bh')
                    cP = cResults[1] # Corrected Pvalue
                    qvalue[chunk[2]] = cP
                else:
                    logger.debug('Chrom:{0},        ({1},{2}), Skipping ...'.format(chrom, pi, wi))
        
            reject = qvalue <= sig
            qvalue = qvalue[reject]
            pvalue = pvalue[reject]
            Ovalues[fl] = Ovalues[fl][reject]
            Evalues = Evalues[reject]
            Fold[fl] = Fold[fl][reject]
            xi = xi[reject]
            yi = yi[reject]
        
            logger.info('Chrom:{0},    ({1},{2}), Remove Gap Effects ...'.format(chrom, pi, wi))
        
            if len(gaps) > 0:
                fIdx = []
                for i in np.arange(xi.size):
                    lower = (xi[i] - 5) if (xi[i] > 5) else 0
                    upper = (xi[i] + 5) if ((xi[i] + 5) < chromLen) else (chromLen - 1)
                    cregion_1 = range(lower, upper)
                    lower = (yi[i] - 5) if (yi[i] > 5) else 0
                    upper = (yi[i] + 5) if ((yi[i] + 5) < chromLen) else (chromLen - 1)
                    cregion_2 = range(lower, upper)
                    cregion = set(cregion_1) | set(cregion_2)
                    intersect = cregion & gaps
                    if len(intersect) == 0:
                        fIdx.append(i)
        
                xi = xi[fIdx]
                yi = yi[fIdx]
                Ovalues[fl] = Ovalues[fl][fIdx]
                pvalue = pvalue[fIdx]
                qvalue = qvalue[fIdx]
                Fold[fl] = Fold[fl][fIdx]
                Evalues = Evalues[fIdx]
        
            xpos[fl] = xi
            ypos[fl] = yi
            pvalues[fl] = pvalue
            qvalues[fl] = qvalue
    
        logger.info('Chrom:{0},    Peak width:{1}, Donut width:{2}, Combine two local filters ...'.format(chrom, pi, wi))
    
        preDonuts = dict(zip(zip(xpos['K'], ypos['K']), zip(Ovalues['K'], Fold['K'], pvalues['K'], qvalues['K'])))
        preLL = dict(zip(zip(xpos['Y'], ypos['Y']), zip(Ovalues['Y'], Fold['Y'], pvalues['Y'], qvalues['Y'])))
    
        commonPos = set(preDonuts.keys()) & set(preLL.keys())
        postcheck = set(preDonuts.keys()) - set(preLL.keys()) # handle special cases for new peak calling
        for ci, cj in postcheck:
            if cEM[ci,cj]==0:
                commonPos.add((ci,cj))
        
        logger.info('Chrom:{0},    Peak width:{1}, Donut width:{2}, Perform greedy clustering and additional filtering ...'.format(chrom, pi, wi))
        Donuts = {}; LL = {}
        for ci, cj in commonPos:
            Donuts[(ci,cj)] = preDonuts[(ci,cj)]
            if (ci,cj) in preLL:
                LL[(ci,cj)] = preLL[(ci,cj)]
            else:
                LL[(ci,cj)] = preDonuts[(ci,cj)]
        peak_list = local_clustering(Donuts, LL, res, r=20000, sumq=sumq) # by default, radius is set to 20Kb
        for pixel, cen, radius in peak_list:
            donut, ll = Donuts[pixel], LL[pixel]
            key = (pixel[0]*res, pixel[1]*res)
            # Additional filtering of peak pixels based on local enrichment thresholds
            if (donut[1]>double_fold) and (ll[1]>double_fold) and (donut[1]>single_fold or ll[1]>single_fold):
                if not key in pixel_table:
                    pixel_table[key] = (cen[0]*res,cen[1]*res) + (radius*res,) + donut + ll[1:]
                else:
                    if (donut[1]>pixel_table[key][4]) and (ll[1]>pixel_table[key][7]):
                        pixel_table[key] = (cen[0]*res,cen[1]*res) + (radius*res,) + donut + ll[1:]
        
    
    logger.info('Chrom:{0}, Combine peak pixels of different pw-ww pairs ...'.format(chrom))
    Donuts = {(k[0]//res,k[1]//res):pixel_table[k][3:7] for k in pixel_table}
    LL = {(k[0]//res,k[1]//res):pixel_table[k][7:] for k in pixel_table}
    peak_list = local_clustering(Donuts, LL, res, r=20000, sumq=sumq)
    final_table = {}
    for pixel, cen, radius in peak_list:
        key = (pixel[0]*res, pixel[1]*res)
        final_table[key] = pixel_table[key]

    return final_table

def bhfdr(M, cM, B1, B2, IR, chromLen, Diags, cDiags, num, chrom, pw = 2, ww = 5, sig = 0.05, maxww = 20,
          maxapart = 2000000, res = 10000):
    
    # more codes for lower memory
    # use reference instead of creating new arrays
    extDiags_ref = []
    for i in range(num):
        OneDArray = Diags[i]
        extODA = np.zeros(chromLen - i + maxww*2)
        extODA[maxww:-maxww] = OneDArray
        extDiags_ref.append(extODA)
    
    extDiags = {maxww: extDiags_ref}
    for w in range(ww, maxww):
        temp = []
        for i in range(num):
            delta = maxww-w
            extODA = extDiags_ref[i][delta:-delta]
            temp.append(extODA)
        extDiags[w] = temp
    
    EDiags = []
    x = np.r_[sorted(IR)]
    for i in x:
        OneDArray = np.ones(chromLen - i) * IR[i]
        EDiags.append(OneDArray)
    
    EM = sparse.diags(EDiags, x, format = 'csr')

    extCDiags_ref = []
    extEDiags_ref = []
    for i in range(x.size):
        extODA_C = np.zeros(chromLen - x[i] + maxww*2)
        extODA_C[maxww:-maxww] = cDiags[i]
        extCDiags_ref.append(extODA_C)
        extODA_E = np.zeros(chromLen - x[i] + maxww*2)
        extODA_E[maxww:-maxww] = EDiags[i]
        extEDiags_ref.append(extODA_E)
    
    extCDiags = {maxww: extCDiags_ref}
    extEDiags = {maxww: extEDiags_ref}
    for w in range(ww, maxww):
        tempC = []
        tempE = []
        for i in range(x.size):
            delta = maxww - w
            extODA_C = extCDiags_ref[i][delta:-delta]
            tempC.append(extODA_C)
            extODA_E = extEDiags_ref[i][delta:-delta]
            tempE.append(extODA_E)
        extCDiags[w] = tempC
        extEDiags[w] = tempE
    
    ps = 2 * pw + 1 # Peak Size

                
    ## Peak Calling ...
    xi, yi = M.nonzero()
    Mask = ((yi - xi) >= ww) & ((yi - xi) <= (maxapart // res))
    xi = xi[Mask]
    yi = yi[Mask]
    bSV = np.zeros(xi.size)
    bEV = np.zeros(xi.size)
    
    logger.info('Chrom:{0}, Observed Contact Number: {1}'.format(chrom, xi.size))
    
    RefIdx = np.arange(xi.size)
    RefMask = np.ones(xi.size, dtype = bool)
    
    iniNum = totalNum = xi.size
    
    logger.info('Chrom:{0}, Calculate the expected matrix ...'.format(chrom))
    bS = sparse.csr_matrix((chromLen, chromLen))
    bE = sparse.csr_matrix((chromLen, chromLen))
    Reads = sparse.csr_matrix((chromLen, chromLen))
    limitCompute = False
    for w in range(ww, maxww + 1):
        ws = 2 * w + 1
        logger.info('Chrom:{0},    Current window width: {1}'.format(chrom, w))
        P1 = set([(i,j) for i in range(w-pw, ps+w-pw) for j in range(w-pw, ps+w-pw)])
        P_1 = set([(i,j) for i in range(w+1, ws) for j in range(w)])
        P_2 = set([(i,j) for i in range(w+1, ps+w-pw) for j in range(w-pw, w)])
        P2 = P_1 - P_2

        ss = range(ws)
        Pool_Diags = {}
        Pool_EDiags = {}
        Pool_cDiags = {}
        for i in ss:
            for j in ss:
                bgloc = max(abs(i-w), abs(j-w)) # mark the radial location on background matrix
                if limitCompute and (bgloc<w):
                    continue
                Pool_Diags[(i,j)] = []
                Pool_EDiags[(i,j)] = []
                Pool_cDiags[(i,j)] = []
                for oi in range(num):
                    if oi + i - j >= 0:
                        starti = i
                        endi = i + chromLen - (oi + i - j)
                    else:
                        starti = i - (oi + i - j)
                        endi = starti + chromLen + (oi + i - j)
                    Pool_Diags[(i,j)].append(extDiags[w][oi][starti:endi])
                for oi in range(x.size):
                    if x[oi] + i - j >= 0:
                        starti = i
                        endi = i + chromLen - (x[oi] + i - j)
                    else:
                        starti = i - (x[oi] + i - j)
                        endi = starti + chromLen + (x[oi] + i - j)
                    Pool_EDiags[(i,j)].append(extEDiags[w][oi][starti:endi])
                    Pool_cDiags[(i,j)].append(extCDiags[w][oi][starti:endi])
        
        limitCompute = True
        
        for key in Pool_Diags:
            if (key[0] != w) and (key[1] != w) and (key not in P1):
                bS = bS + sparse.diags(Pool_cDiags[key], x + (key[0] - key[1]), format = 'csr')
                bE = bE + sparse.diags(Pool_EDiags[key], x + (key[0] - key[1]), format = 'csr')
            if key in P2:
                Reads = Reads + sparse.diags(Pool_Diags[key], np.arange(num) + (key[0] - key[1]), format = 'csr')
        
        Txi = xi[RefIdx]
        Tyi = yi[RefIdx]
        RNums = np.array(Reads[Txi, Tyi]).ravel()
        EIdx = RefIdx[RNums >= 16]
        logger.info('Chrom:{0},    Valid Contact Number from This Loop: {1}'.format(chrom, EIdx.size))
        Valid_Ratio = EIdx.size/float(iniNum)
        Exi = xi[EIdx]
        Eyi = yi[EIdx]
        bSV[EIdx] = np.array(bS[Exi, Eyi]).ravel()
        bEV[EIdx] = np.array(bE[Exi, Eyi]).ravel()
        RefIdx = RefIdx[RNums < 16]
            
        iniNum = RefIdx.size

        left_Ratio = iniNum/float(totalNum)

        logger.info('Chrom:{0},    Total Valid Ratio after This Loop: {1:.3f}'.format(chrom, 1-left_Ratio))
        
        if Valid_Ratio < 0.3:
            logger.info('Chrom:{0},    Ratio of valid contact is too small, break the loop ...'.format(chrom))
            break
        
        if left_Ratio < 0.03:
            logger.info('Chrom:{0},    Very few or no contacts are left, break the loop ...'.format(chrom))
            break
        
        logger.info('Chrom:{0},    {1} Contacts will get into next loop ...'.format(chrom, RefIdx.size))
    
    RefMask[RefIdx] = False
    
    Mask = np.logical_and((bEV != 0), RefMask)
    xi = xi[Mask]
    yi = yi[Mask]
    bRV = bSV[Mask] / bEV[Mask]
    
    bR = sparse.lil_matrix((chromLen, chromLen))
    bR[xi, yi] = bRV
    
    ## Corrected Expected Matrix
    cEM = EM.multiply(bR.tocsr())
    
    logger.info('Chrom:{0}, Construct Poisson Models ...'.format(chrom))
    ## Poisson Models
    xi, yi = cEM.nonzero()
    Evalues = np.array(cEM[xi, yi]).ravel() * B1[xi] * B2[yi]
    Mask = (Evalues > 0)
    Evalues = Evalues[Mask]
    xi = xi[Mask]
    yi = yi[Mask]
    Poisses = poisson(Evalues)
    logger.info('Chrom:{0}, Number of Poisson Models: {1}'.format(chrom, Evalues.size))
    logger.info('Chrom:{0}, Assign a p-value for each Observed Contact Frequency ...'.format(chrom))
    Ovalues = np.array(M[xi, yi]).ravel()
    pvalues = 1 - Poisses.cdf(Ovalues)
    Fold = Ovalues / Evalues
    
    # Multiple Tests
    logger.info('Chrom:{0}, Benjamini-Hochberg correcting for multiple tests ...'.format(chrom))
    cResults = multipletests(pvalues, alpha = sig, method = 'fdr_bh')
    reject = cResults[0]
    cP = cResults[1] # Corrected Pvalue
    xpos = xi[reject]
    ypos = yi[reject]
    pvalues = pvalues[reject]
    qvalues = cP[reject]
    Ovalues = Ovalues[reject]
    Fold = Fold[reject]
    
    # Remove Gap Effect
    logger.info('Chrom:{0}, Remove Gap Effects ...'.format(chrom))
    gaps = set(np.where(np.array(cM.sum(axis=1)).ravel() == 0)[0])
    if len(gaps) > 0:
        fIdx = []
        for i in np.arange(xpos.size):
            lower = (xpos[i] - 5) if (xpos[i] > 5) else 0
            upper = (xpos[i] + 5) if ((xpos[i] + 5) < chromLen) else (chromLen - 1)
            cregion_1 = range(lower, upper)
            lower = (ypos[i] - 5) if (ypos[i] > 5) else 0
            upper = (ypos[i] + 5) if ((ypos[i] + 5) < chromLen) else (chromLen - 1)
            cregion_2 = range(lower, upper)
            cregion = set(cregion_1) | set(cregion_2)
            intersect = cregion & gaps
            if len(intersect) == 0:
                fIdx.append(i)
        
        xpos = xpos[fIdx]
        ypos = ypos[fIdx]
        pvalues = pvalues[fIdx]
        qvalues = qvalues[fIdx]
        Ovalues = Ovalues[fIdx]
        Fold = Fold[fIdx]
    
    logger.info('Chrom:{0}, Perform greedy clustering and additional filtering ...'.format(chrom))
    Donuts = dict(zip(zip(xpos, ypos), zip(Ovalues, Fold, pvalues, qvalues)))
    pixel_list = local_clustering(Donuts, None, res, r=20000) # by default, radius is set to 20Kb
    pixel_table = {}
    for pixel, cen, radius in pixel_list:
        donut = Donuts[pixel]
        # Additional filtering of peak pixels based on local enrichment thresholds
        if donut[1]>2:
            pixel_table[(pixel[0]*res,pixel[1]*res)] = (cen[0]*res,cen[1]*res) + (radius*res,) + donut
    
    return pixel_table


def local_clustering(Donuts, LL, res, r=20000, sumq=0.01):

    from sklearn.cluster import dbscan
    from scipy.spatial.distance import euclidean
    
    r = max(r//res,1)
    sort_list = []
    for i, j in Donuts:
        sort_list.append((Donuts[(i,j)][0], (i,j)))
    sort_list.sort(reverse=True)
    pos = np.r_[[i[1] for i in sort_list]]
    if len(pos) >= 2:
        _, labels = dbscan(pos, eps=r, min_samples=2)
        pool = set()
        final_list = []
        for i, p in enumerate(sort_list):
            if p[1] in pool:
                continue
            c = labels[i]
            pool.add(p[1])
            if c==-1:
                if not LL is None:
                    if Donuts[p[1]][-1] + LL[p[1]][-1] <= sumq:
                        final_list.append((p[1], p[1], 0))
                else:
                    if Donuts[p[1]][-1] <= sumq/2:
                        final_list.append((p[1], p[1], 0))
            else:
                sub = pos[labels==c]
                cen = p[1]
                rad = r
                Local = [p[1]]
                ini = -1
                while len(sub):
                    out = []
                    for q in sub:
                        if tuple(q) in pool:
                            continue
                        tmp = euclidean(q, cen)
                        if tmp<=rad:
                            Local.append(tuple(q))
                        else:
                            out.append(tuple(q))
                    if len(out)==ini:
                        break
                    ini = len(out)
                    tmp = np.r_[Local]
                    cen = tuple(tmp.mean(axis=0).round().astype(int)) # assign centroid to a certain pixel
                    rad = np.int(np.round(max([euclidean(cen,q) for q in Local]))) + r
                    sub = np.r_[out]
                for q in Local:
                    pool.add(q)
                final_list.append((p[1], cen, rad))
    elif len(pos)==1:
        if not LL is None:
            if Donuts[tuple(pos[0])][-1] + LL[tuple(pos[0])][-1] <= sumq:
                final_list = [(tuple(pos[0]), tuple(pos[0]), 0)]
        else:
            if Donuts[tuple(pos[0])][-1] <= sumq/2:
                final_list = [(tuple(pos[0]), tuple(pos[0]), 0)]
    else:
        final_list = []
    
    return final_list
    