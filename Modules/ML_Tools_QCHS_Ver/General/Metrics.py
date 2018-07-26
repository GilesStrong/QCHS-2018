from __future__ import division
import pandas
import numpy as np
import math
import multiprocessing as mp

from sklearn.model_selection import StratifiedKFold

from Modules.ML_Tools_QCHS_Ver.General.Misc_Functions import uncertRound

wFactor = 250000/50000

def AMS(s, b, br=10.0):
    """ Approximate Median Significance defined as:
        AMS = sqrt(
                2 { (s + b + b_r) log[1 + (s/(b+b_r))] - s}
              )        
    where b_r = 10, b = background, s = signal, log is natural logarithm """
    
    radicand = 2 *( (s+b+br) * math.log (1.0 + s/(b+br)) -s)
    if radicand < 0:
        print('radicand is negative. Exiting')
        return -1
    else:
        return math.sqrt(radicand)

def amsScanQuick(inData, wFactor=250000./50000., br=10):
    '''Determine optimum AMS and cut,
    wFactor used rescale weights to get comparable AMSs
    sufferes from float precison'''
    amsMax = 0
    threshold = 0.0
    inData = inData.sort_values(by=['pred_class'])
    s = np.sum(inData.loc[(inData['gen_target'] == 1), 'gen_weight'])
    b = np.sum(inData.loc[(inData['gen_target'] == 0), 'gen_weight'])

    for i, cut in enumerate(inData['pred_class']):
        ams = AMS(max(0, s*wFactor), max(0, b*wFactor), br)
        
        if ams > amsMax:
            amsMax = ams
            threshold = cut
        if inData['gen_target'].values[i]:
            s -= inData['gen_weight'].values[i]
        else:
            b -= inData['gen_weight'].values[i]
            
    return amsMax, threshold

def amsScanSlow(inData, wFactor=250000./50000., br=10, start=0.9):
    '''Determine optimum AMS and cut,
    wFactor used rescale weights to get comparable AMSs
    slower than quick, but doesn suffer from float precision'''
    amsMax = 0
    threshold = 0.0
    signal = inData[inData['gen_target'] == 1]
    bkg = inData[inData['gen_target'] == 0]
    
    for i, cut in enumerate(inData.loc[inData.pred_class >= start, 'pred_class'].values):
        s = np.sum(signal.loc[(signal.pred_class >= cut), 'gen_weight'])
        b = np.sum(bkg.loc[(bkg.pred_class >= cut), 'gen_weight'])
        ams = AMS(s*wFactor, b*wFactor, br)
        
        if ams > amsMax:
            amsMax = ams
            threshold = cut
            
    return amsMax, threshold

def mpAMS(data, i, wFactor, br, out_q):
    ams, cut = amsScanQuick(data, wFactor, br)
    out_q.put({str(i) + '_ams':ams, str(i) + '_cut':cut})

def mpSKFoldAMS(data, i, size, nFolds, br, out_q):
    kf = StratifiedKFold(n_splits=nFolds, shuffle=True)
    folds = kf.split(data, data['gen_target'])
    uids = range(i*nFolds,(i+1)*nFolds)
    outdict = {}

    for j, (train, test) in enumerate(folds):
        ams, cut = amsScanQuick(data.iloc[test], size/len(test), br)
        if ams > 0:
            outdict[str(uids[j]) + '_ams'] = ams
            outdict[str(uids[j]) + '_cuts'] = cut
    out_q.put(outdict)

def bootstrapMeanAMS(data, wFactor=250000./50000., N=512, br=10):
    procs = []
    out_q = mp.Queue()
    for i in range(N):
        indeces = np.random.choice(data.index, len(data), replace=True)
        p = mp.Process(target=mpAMS, args=(data.iloc[indeces], i, wFactor, br, out_q))
        procs.append(p)
        p.start() 
    resultdict = {}
    for i in range(N):
        resultdict.update(out_q.get()) 
    for p in procs:
        p.join()  
        
    amss = np.array([resultdict[x] for x in resultdict if 'ams' in x])
    cuts = np.array([resultdict[x] for x in resultdict if 'cut' in x])

    meanAMS = uncertRound(np.mean(amss), np.std(amss))
    meanCut = uncertRound(np.mean(cuts), np.std(cuts))

    ams = AMS(wFactor*np.sum(data.loc[(data.pred_class >= np.mean(cuts)) & (data.gen_target == 1), 'gen_weight']),
              wFactor*np.sum(data.loc[(data.pred_class >= np.mean(cuts)) & (data.gen_target == 0), 'gen_weight']))
    
    print('\nMean AMS={}+-{}, at mean cut of {}+-{}'.format(meanAMS[0], meanAMS[1], meanCut[0], meanCut[1]))
    print('Exact mean cut {}, corresponds to AMS of {}'.format(np.mean(cuts), ams))
    return (meanAMS[0], meanCut[0])

def bootstrapSKFoldMeanAMS(data, size=250000., N=10, nFolds=500, br=10):
    print("Warning, this method might not be trustworthy: cut decreases with nFolds")
    procs = []
    out_q = mp.Queue()
    for i in range(N):
        indeces = np.random.choice(data.index, len(data), replace=True)
        p = mp.Process(target=mpSKFoldAMS, args=(data, i, size, nFolds, br, out_q))
        procs.append(p)
        p.start() 
    resultdict = {}
    for i in range(N):
        resultdict.update(out_q.get()) 
    for p in procs:
        p.join()  
        
    amss = np.array([resultdict[x] for x in resultdict if 'ams' in x])
    cuts = np.array([resultdict[x] for x in resultdict if 'cut' in x])

    meanAMS = uncertRound(np.mean(amss), np.std(amss)/np.sqrt(N*nFolds))
    meanCut = uncertRound(np.mean(cuts), np.std(cuts)/np.sqrt(N*nFolds))

    scale = size/len(data)
    ams = AMS(scale*np.sum(data.loc[(data.pred_class >= np.mean(cuts)) & (data.gen_target == 1), 'gen_weight']),
              scale*np.sum(data.loc[(data.pred_class >= np.mean(cuts)) & (data.gen_target == 0), 'gen_weight']))
    
    print('\nMean AMS={}+-{}, at mean cut of {}+-{}'.format(meanAMS[0], meanAMS[1], meanCut[0], meanCut[1]))
    print('Exact mean cut {}, corresponds to AMS of {}'.format(np.mean(cuts), ams))
    return (meanAMS[0], meanCut[0])
