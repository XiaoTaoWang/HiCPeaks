# Created on Tue Apr 21 11:20:30 2015

# Author: XiaoTao Wang
# Organization: HuaZhong Agricultural University

from __future__ import division
import argparse, sys, logging, logging.handlers

import numpy as np
from scipy import sparse
from sklearn import isotonic

def getargs():
    ## Construct an ArgumentParser object for command-line arguments
    parser = argparse.ArgumentParser(usage = '%(prog)s <-O output> [options]',
                                     description = 'Local Peak Calling for Hi-C Data',
                                     formatter_class = argparse.ArgumentDefaultsHelpFormatter)

    # Output
    parser.add_argument('-O', '--output', help = 'Output prefix.')
    parser.add_argument('--logFile', default = 'BHFDR.log', help = 'Logging file name.')
    
    group_1 = parser.add_argument_group(title = 'Relate to Hi-C data:')
    group_1.add_argument('-p', '--path', default = '.',
                         help = 'Path to Hi-C data')
    group_1.add_argument('-R', '--resolution', default = 10000, type = int,
                         help = 'Resolution of the binned data')
    group_1.add_argument('-C', '--chroms', nargs = '*', default = ['#', 'X'],
                         help = 'Which chromosomes to read. Specially, "#" stands'
                         ' for chromosomes with numerical labels. "--chroms" with zero argument'
                         ' will generate an empty list, in which case all chromosome data will'
                         ' be loaded.')
    
    ## About the algorithm
    group_2 = parser.add_argument_group(title = 'Algorithm Parameters:')
    group_2.add_argument('--pw', type = int, default = 2, help = 'Width of the interaction '
                         'region surrounding the peak. According to experience, we set it'
                         ' to 1 at 20 kb, 2 at 10 kb, and 4 at 5 kb.')
    group_2.add_argument('--ww', type = int, default = 5, help = 'The size of the donut '
                         'sampled. Set it to 3 at 20 kb, 5 at 10 kb, and 7 at 5 kb.')
    group_2.add_argument('--maxww', type = int, default = 20, help = 'Maximum donut size.')
    group_2.add_argument('--siglevel', type = float, default = 0.05, help = 'Significant Level.')
    group_2.add_argument('--maxapart', type = int, default = 2000000, help = 'Maximum genomic'
                         ' distance between two loci.')
    
    ## Parse the command-line arguments
    commands = sys.argv[1:]
    if not commands:
        commands.append('-h')
    args = parser.parse_args(commands)
    
    return args, commands

def run():
     # Parse Arguments
    args, commands = getargs()
    # Improve the performance if you don't want to run it
    if commands[0] not in ['-h', '--help']:
        ## Root Logger Configuration
        logger = logging.getLogger()
        # Logger Level
        logger.setLevel(10)
        console = logging.StreamHandler()
        filehandler = logging.handlers.RotatingFileHandler(args.logFile,
                                                           maxBytes = 100000,
                                                           backupCount = 5)
        # Set level for Handlers
        console.setLevel('INFO')
        filehandler.setLevel('DEBUG')
        # Customizing Formatter
        formatter = logging.Formatter(fmt = '%(name)-14s %(levelname)-7s @ %(asctime)s: %(message)s',
                                      datefmt = '%m/%d/%y %H:%M:%S')
        
        console.setFormatter(formatter)
        filehandler.setFormatter(formatter)
        # Add Handlers
        logger.addHandler(console)
        logger.addHandler(filehandler)
        
        ## Logging for argument setting
        arglist = ['# ARGUMENT LIST:',
                   '# output file prefix = %s' % args.output,
                   '# HiC Data Path = %s' % args.path,
                   '# chromosomes = %s' % args.chroms,
                   '# data resolution = %s' % args.resolution,
                   '# Peak window width = %s' % args.pw,
                   '# Donut size = %s' % args.ww,
                   '# Maximum donut size = %s' % args.maxww,
                   '# Significant Level = %s' % args.siglevel,
                   '# Genomic distance range = %s' % [args.ww * args.resolution, args.maxapart]
                   ]
        
        argtxt = '\n'.join(arglist)
        logger.info('\n' + argtxt)
        
        # Package Dependencies
        from mirnylib.numutils import completeIC
        
        logger.info('Locating Hi-C data ...')
        Lib = np.load(args.path)
        
        logger.info('Calling Peaks ...')
        OF = open('.'.join([args.output, 'peaks', 'txt']), 'wb')
        head = '\t'.join(['chromLabel', 'loc_1', 'loc_2', 'IF', 'Fold-Enrichment', 'pvalue', 'qvalue']) + '\n'
        OF.write(head)
        
        for key in Lib.files:
            if ((not args.chroms) or (key.isdigit() and '#' in args.chroms) or (key in args.chroms)):
                logger.info('Chromosome %s ...', key)
                sparseH = Lib[key].reshape(1)[0]
                triuH = sparseH.toarray()
                H = triuH + triuH.T - np.diag(triuH.diagonal()) # Symmetric Matrix
                del sparseH, triuH # Release Memory
                logger.info('Perform ICE ...')
                cHeatMap, biases = completeIC(H, returnBias = True)
                logger.info('Done!')
                
                logger.info('Customize Sparse Matrix ...')
                chromLen = H.shape[0]
                num = args.maxapart // args.resolution + args.maxww + 1
                Diags = [np.diagonal(H, i) for i in np.arange(num)]
                M = sparse.diags(Diags, np.arange(num), format = 'csr')
                x = np.arange(args.ww, num)
                y = []
                cDiags = []
                for i in x:
                    diag = np.diagonal(cHeatMap, i)
                    y.append(diag.mean())
                    cDiags.append(diag)
                cM = sparse.diags(cDiags, x, format = 'csr')
                IR = isotonic.IsotonicRegression(increasing = 'auto')
                IR.fit(x, y)
                
                del H, cHeatMap
                
                xpos, ypos, Ovalues, Fold, pvalues, qvalues = pcaller(M, cM, biases, IR, chromLen, Diags, cDiags, num,
                                                                      pw = args.pw, ww = args.ww, sig = args.siglevel,
                                                                      maxww = args.maxww, maxapart = args.maxapart,
                                                                      res = args.resolution)
                                                                
                for i in xrange(xpos.size):
                    line = '%s\t%d\t%d\t%.4g\t%.4g\t%.4g\t%.4g\n' % (key, xpos[i] * args.resolution, ypos[i] * args.resolution, Ovalues[i], Fold[i], pvalues[i], qvalues[i])
                    OF.write(line)
                    
        OF.flush()
        OF.close()
                            
        logger.info('Done!')

def pcaller(M, cM, biases, IR, chromLen, Diags, cDiags, num, pw = 2, ww = 5, sig = 0.05, maxww = 20,
            maxapart = 2000000, res = 10000):
    
    # Necessary Modules
    from scipy.stats import poisson
    from statsmodels.sandbox.stats.multicomp import multipletests
    
    logger = logging.getLogger()
    
    extDiags = {}
    for w in range(ww, maxww + 1):
        temp = []
        for i in xrange(num):
            OneDArray = Diags[i]
            extODA = np.zeros(chromLen - i + w*2)
            extODA[w:-w] = OneDArray
            temp.append(extODA)
        extDiags[w] = temp
    
    x = np.arange(ww, num)
    predictE = IR.predict(x)
    predictE[predictE < 0] = 0
    EDiags = []
    for i in xrange(x.size):
        OneDArray = np.ones(chromLen - x[i]) * predictE[i]
        EDiags.append(OneDArray)
    
    EM = sparse.diags(EDiags, x, format = 'csr')
    
    extCDiags = {}
    extEDiags = {}
    for w in range(ww, maxww + 1):
        tempC = []
        tempE = []
        for i in xrange(x.size):
            extODA_E = np.zeros(chromLen - x[i] + w*2)
            extODA_E[w:-w] = EDiags[i]
            tempE.append(extODA_E)
            extODA_C = np.zeros(chromLen - x[i] + w*2)
            extODA_C[w:-w] = cDiags[i]
            tempC.append(extODA_C)
        extCDiags[w] = tempC
        extEDiags[w] = tempE
    
    ps = 2 * pw + 1 # Peak Size
    
    Pool_Diags = {}
    Pool_EDiags = {}
    Pool_cDiags = {}
    Offsets_Diags = {}
    Offsets_EDiags = {}
    
    for w in range(ww, maxww + 1):
        ws = 2 * w + 1 # Window size
        ss = range(ws)
        Pool_Diags[w] = {}
        Pool_EDiags[w] = {}
        Pool_cDiags[w] = {}
        Offsets_Diags[w] = {}
        Offsets_EDiags[w] = {}
        for i in ss:
            for j in ss:
                Pool_Diags[w][(i,j)] = []
                Pool_EDiags[w][(i,j)] = []
                Pool_cDiags[w][(i,j)] = []
                Offsets_Diags[w][(i,j)] = np.arange(num) + (i - j)
                Offsets_EDiags[w][(i,j)] = x + (i - j)
                for oi in np.arange(num):
                    if Offsets_Diags[w][(i,j)][oi] >= 0:
                        starti = i
                        endi = i + chromLen - Offsets_Diags[w][(i,j)][oi]
                    else:
                        starti = i - Offsets_Diags[w][(i,j)][oi]
                        endi = starti + chromLen + Offsets_Diags[w][(i,j)][oi]
                    Pool_Diags[w][(i,j)].append(extDiags[w][oi][starti:endi])
                for oi in xrange(x.size):
                    if Offsets_EDiags[w][(i,j)][oi] >= 0:
                        starti = i
                        endi = i + chromLen - Offsets_EDiags[w][(i,j)][oi]
                    else:
                        starti = i - Offsets_EDiags[w][(i,j)][oi]
                        endi = starti + chromLen + Offsets_EDiags[w][(i,j)][oi]
                    Pool_EDiags[w][(i,j)].append(extEDiags[w][oi][starti:endi])
                    Pool_cDiags[w][(i,j)].append(extCDiags[w][oi][starti:endi])
                
    ## Peak Calling ...
    xi, yi = M.nonzero()
    Mask = ((yi - xi) >= ww) & ((yi - xi) <= (maxapart // res))
    xi = xi[Mask]
    yi = yi[Mask]
    bSV = np.zeros(xi.size)
    bEV = np.zeros(xi.size)
    
    logger.info('Observed Contact Number: %d', xi.size)
    
    RefIdx = np.arange(xi.size)
    RefMask = np.ones_like(xi, dtype = bool)
    
    iniNum = xi.size
    
    logger.info('Calculate the expected matrix ...')
    for w in range(ww, maxww + 1):
        ws = 2 * w + 1
        bS = sparse.csr_matrix((chromLen, chromLen))
        bE = sparse.csr_matrix((chromLen, chromLen))
        Reads = sparse.csr_matrix((chromLen, chromLen))
        logger.info('    Current window width: %s' % w)
        P1 = set([(i,j) for i in range(w-pw, ps+w-pw) for j in range(w-pw, ps+w-pw)])
        P_1 = set([(i,j) for i in range(w+1, ws) for j in range(w)])
        P_2 = set([(i,j) for i in range(w+1, ps+w-pw) for j in range(w-pw, w)])
        P2 = P_1 - P_2
        for key in Pool_Diags[w]:
            if (key[0] != w) and (key[1] != w) and (key not in P1):
                bS = bS + sparse.diags(Pool_cDiags[w][key], Offsets_EDiags[w][key], format = 'csr')
                bE = bE + sparse.diags(Pool_EDiags[w][key], Offsets_EDiags[w][key], format = 'csr')
            if key in P2:
                Reads = Reads + sparse.diags(Pool_Diags[w][key], Offsets_Diags[w][key], format = 'csr')
        
        Txi = xi[RefIdx]
        Tyi = yi[RefIdx]
        RNums = np.array(Reads[Txi, Tyi]).ravel()
        EIdx = RefIdx[RNums >= 16]
        logger.info('    Valid Contact Number: %d', EIdx.size)
        Valid_Ratio = EIdx.size/float(iniNum)
        logger.info('    Valid Contact Ratio: %.3f', Valid_Ratio)
        Exi = xi[EIdx]
        Eyi = yi[EIdx]
        bSV[EIdx] = np.array(bS[Exi, Eyi]).ravel()
        bEV[EIdx] = np.array(bE[Exi, Eyi]).ravel()
        RefIdx = RefIdx[RNums < 16]
            
        iniNum = RefIdx.size
        
        if Valid_Ratio < 0.1:
            logger.info('    Ratio of valid contact is too small, break the loop ...')
            break
        
        logger.info('    Continue ...')
        logger.info('    %d Contacts will get into next loop ...', RefIdx.size)
    
    RefMask[RefIdx] = False
    
    Mask = np.logical_and((bEV != 0), RefMask)
    xi = xi[Mask]
    yi = yi[Mask]
    bRV = bSV[Mask] / bEV[Mask]
    
    bR = sparse.csr_matrix((chromLen, chromLen))
    bR[xi, yi] = bRV
    
    ## Corrected Expected Matrix
    cEM = EM.multiply(bR)
    
    logger.info('Construct Poisson Models ...')
    ## Poisson Models
    xi, yi = cEM.nonzero()
    Evalues = np.array(cEM[xi, yi]).ravel() * biases[xi] * biases[yi]
    Mask = (Evalues > 0)
    Evalues = Evalues[Mask]
    xi = xi[Mask]
    yi = yi[Mask]
    Poisses = poisson(Evalues)
    logger.info('Number of Poisson Models: %d', Evalues.size)
    logger.info('Assign a p-value for each Observed Contact Frequency ...')
    Ovalues = np.array(M[xi, yi]).ravel()
    pvalues = 1 - Poisses.cdf(Ovalues)
    Fold = Ovalues / Evalues
    
    # Multiple Tests
    logger.info('Benjamini-Hochberg correcting for multiple tests ...')
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
    logger.info('Remove Gap Effects ...')
    gaps = set(np.where(np.array(M.sum(axis=1)).ravel() == 0)[0])
    if len(gaps) > 0:
        fIdx = []
        for i in xrange(xpos.size):
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
    
    return xpos, ypos, Ovalues, Fold, pvalues, qvalues
    

if __name__ == '__main__':
    run()
