from __future__ import division

from sklearn.metrics import roc_auc_score, roc_curve

import seaborn as sns
import matplotlib.pyplot as plt
sns.set_style("whitegrid")

import numpy as np
import pandas
import types

from Modules.ML_Tools_QCHS_Ver.General.Misc_Functions import uncertRound
from Modules.ML_Tools_QCHS_Ver.Plotting_And_Evaluation.Bootstrap import * 

def plotFeat(inData, feat, cuts=None, labels=None, plotBulk=True, weightName=None, nSamples=100000, params={}):
    loop = False
    if not isinstance(cuts, type(None)):
        if isinstance(cuts, list):
            loop = True
            if not isinstance(cuts, list):
                print ("{} plots requested, but not labels passed".format(len(cuts)))
                return -1
            elif len(cuts) != len(labels):
                print ("{} plots requested, but {} labels passed".format(len(cuts), len(labels)))
                return -1
    
    weightData = None
    
    plt.figure(figsize=(16, 8))
    if loop:
        for i in range(len(cuts)):
            if isinstance(params, list):
                tempParams = params[i]
            else:
                tempParams = params

            if plotBulk: #Ignore tails for indicative plotting
                featRange = np.percentile(inData[feat], [1,99])
                #featRange = np.percentile(inData.loc[cuts[i], feat], [1,99])
                if featRange[0] == featRange[1]:break
                
                cut = (cuts[i])
                cut = cut & (inData[cut][feat] > featRange[0]) & (inData[cut][feat] < featRange[1])
                if isinstance(weightName, type(None)):
                    plotData = inData.loc[cut, feat]
                else:
                    plotData = np.random.choice(inData.loc[cut, feat], nSamples, p=inData.loc[cut, weightName]/np.sum(inData.loc[cut, weightName]))
                    
            else:
                if isinstance(weightName, type(None)):
                    plotData = inData.loc[cuts[i], feat]
                else:
                    plotData = np.random.choice(inData.loc[cuts[i], feat], nSamples, p=inData.loc[cuts[i], weightName]/np.sum(inData.loc[cuts[i], weightName]))
                
            sns.distplot(plotData, label=labels[i], **tempParams)
    else:
        if plotBulk: #Ignore tails for indicative plotting
            featRange = np.percentile(inData[feat], [1,99])
            if featRange[0] == featRange[1]:return -1
            
            cut = (inData[feat] > featRange[0]) & (inData[feat] < featRange[1])
            
            
            if isinstance(weightName, type(None)):
                plotData = inData.loc[cut, feat]
            else:
                plotData = np.random.choice(inData.loc[cut, feat], nSamples, p=inData.loc[cut, weightName]/np.sum(inData.loc[cut, weightName]))
                
                
        else:
            if isinstance(weightName, type(None)):
                plotData = inData[feat]
            else:
                plotData = np.random.choice(inData[feat], nSamples, p=inData[weightName]/np.sum(inData[weightName]))
                                
        sns.distplot(plotData, **params)
    if loop:
        plt.legend(loc='best', fontsize=16)
    plt.xticks(fontsize=16, color='black')
    plt.yticks(fontsize=16, color='black')
    plt.ylabel("Density", fontsize=24, color='black')
    plt.xlabel(feat, fontsize=24, color='black')
    plt.show()

def rocPlot(inData=None, curves=None, predName='pred_class', targetName='gen_target', weightName=None, labels=None, aucs=None, bootstrap=False, log=False, baseline=True, params=[{}]):
    buildCurves = True
    if isinstance(inData, type(None)) == isinstance(curves, type(None)):
        print ("Must pass either targets and preds, or curves")
        return -1
    if not isinstance(curves, type(None)):
        buildCurves = False

    if buildCurves:
        curves = {}
        if bootstrap:
            aucArgs = []
            for i in range(len(inData)):
                aucArgs.append({'labels':inData[i][targetName], 'preds':inData[i][predName], 'name':labels[i], 'indeces':inData[i].index.tolist()})
                if not isinstance(weightName, type(None)):
                    aucArgs[-1]['weights'] = inData[i][weightName]
            aucs = mpRun(aucArgs, rocauc)
            meanScores = {}
            for i in labels:
                meanScores[i] = (np.mean(aucs[i]), np.std(aucs[i]))
                print (str(i)+ ' ROC AUC, Mean = {} +- {}'.format(meanScores[i][0], meanScores[i][1]))
        else:
            meanScores = {}
            for i in range(len(inData)):
                if isinstance(weightName, type(None)):
                    meanScores[labels[i]] = roc_auc_score(inData[i][targetName].values, inData[i][predName])
                else:
                    meanScores[labels[i]] = roc_auc_score(inData[i][targetName].values, inData[i][predName], sample_weight=inData[i][weightName])
                print (str(i) + ' ROC AUC: {}'.format(meanScores[labels[i]]))
        for i in range(len(inData)):
            if isinstance(weightName, type(None)):
                curves[labels[i]] = roc_curve(inData[i][targetName].values, inData[i][predName].values)[:2]
            else:
                curves[labels[i]] = roc_curve(inData[i][targetName].values, inData[i][predName].values, sample_weight=inData[i][weightName].values)[:2]

    plt.figure(figsize=[8, 8])
    for i in range(len(curves)):
        if buildCurves:
            if bootstrap:
                meanScore = uncertRound(*meanScores[labels[i]])
                plt.plot(*curves[labels[i]], label=labels[i] + r', AUC$={}\pm{}$'.format(meanScore[0], meanScore[1]), **params[i])
            else:
                plt.plot(*curves[labels[i]], label=labels[i] + r', AUC$={:.5f}$'.format(meanScores[labels[i]]), **params[i])
        else:
            plt.plot(*curves[i], label=labels[i], **params[i])
    
    if baseline:
        plt.plot([0, 1], [0, 1], 'k--', label='No discrimination')
    plt.xlabel('Background acceptance', fontsize=24, color='black')
    plt.ylabel('Signal acceptance', fontsize=24, color='black')
    if len(labels):
        plt.legend(loc='best', fontsize=16)
    if log:
        plt.xscale('log', nonposx='clip')
        plt.grid(True, which="both")
    plt.xticks(fontsize=16, color='black')
    plt.yticks(fontsize=16, color='black')
    plt.show()

def getClassPredPlot(inData, labels=['Background', 'Signal'], predName='pred_class', weightName=None,
                     lim=(0,1), logy=True, params={'hist' : True, 'kde' : False, 'norm_hist' : True}):
    plt.figure(figsize=(16, 8))
    for i in range(len(inData)):
        hist_kws = {}
        if not isinstance(weightName, type(None)):
            hist_kws['weights'] = inData[i][weightName]
        sns.distplot(inData[i][predName], label=labels[i], hist_kws=hist_kws, **params)
    plt.legend(loc='best', fontsize=16)
    plt.xlabel("Class prediction", fontsize=24, color='black')
    plt.xlim(lim)
    plt.ylabel(r"$\frac{1}{N}\ \frac{dN}{dp}$", fontsize=24, color='black')
    if logy:
        plt.yscale('log', nonposy='clip')
        plt.grid(True, which="both")
    plt.xticks(fontsize=16, color='black')
    plt.yticks(fontsize=16, color='black')
    plt.show() 

def _getSamples(inData, sampleName, weightName):
    samples = set(inData[sampleName])
    weights=[np.sum(inData[inData[sampleName] == sample][weightName]) for sample in samples]
    return [x[0] for x in np.array(sorted(zip(samples, weights), key=lambda x: x[1]))] #Todo improve sorting

def getSamplePredPlot(inData, 
                      targetName='gen_target', sampleName='gen_sample', predName='pred_class', weightName='gen_weight',
                      lim=(0,1), nBins = 35, logy=True, pallet='nipy_spectral', desat=0.8,
                      hist_params={'normed': True, 'alpha': 1, 'stacked':True, 'rwidth':1.0,}):
    
    hist_params['bins'] = nBins
    hist_params['range'] = lim
    
    plt.figure(figsize=(16, 8))
    
    sig = (inData[targetName] == 1)
    bkg = (inData[targetName] == 0)
    
    with sns.color_palette(pallet, len(set(inData[sampleName])), desat=desat):
        
        samples = _getSamples(inData[bkg], sampleName, weightName)
        plt.hist([inData[inData[sampleName] == sample][predName] for sample in samples],
                 weights=[inData[inData[sampleName] == sample][weightName] for sample in samples],
                 label=[sample.decode("utf-8") for sample in samples], **hist_params)

        samples = _getSamples(inData[sig], sampleName, weightName)
        for sample in samples:
            plt.hist(inData[inData[sampleName] == sample][predName],
                     weights=inData[inData[sampleName] == sample][weightName],
                     label='Signal ' + sample.decode("utf-8"), histtype='step', linewidth='3', **hist_params)

        plt.legend(loc='best', fontsize=16)
        plt.xlabel("Class prediction", fontsize=24, color='black')
        plt.xlim(lim)
        if hist_params['normed']:
            plt.ylabel(r"$\frac{1}{\mathcal{A}\sigma} \frac{d\left(\mathcal{A}\sigma\right)}{dp}$", fontsize=24, color='black')
        else:
            plt.ylabel(r"$\frac{d\left(\mathcal{A}\sigma\right)}{dp}\ [pb]$", fontsize=24, color='black')
        if logy:
            plt.yscale('log', nonposy='clip')
            plt.grid(True, which="both")
        plt.xticks(fontsize=16, color='black')
        plt.yticks(fontsize=16, color='black')
        plt.show()      

def plotHistory(histories):
    print ("Depreciated, move to plotTrainingHistory")
    plotTrainingHistory(histories)
    
def plotTrainingHistory(histories, save=False):
    plt.figure(figsize=(16,8))
    for i, history in enumerate(histories):
        if i == 0:
            try:
                plt.plot(history['loss'], color='g', label='Training')
            except:
                pass
            try:
                plt.plot(history['val_loss'], color='b', label='Validation')
            except:
                pass
            try:
                plt.plot(history['mon_loss'], color='r', label='Monitoring')
            except:
                pass
            try:
                plt.plot(history['swa_val_loss'], color='purple', label='SWA Validation')
            except:
                pass
        else:
            try:
                plt.plot(history['loss'], color='g')
            except:
                pass
            try:
                plt.plot(history['val_loss'], color='b')
            except:
                pass
            try:
                plt.plot(history['mon_loss'], color='r')
            except:
                pass
            try:
                plt.plot(history['swa_val_loss'], color='purple')
            except:
                pass
    plt.legend(loc='best', fontsize=16)
    plt.xticks(fontsize=16, color='black')
    plt.yticks(fontsize=16, color='black')
    plt.xlabel("Epoch", fontsize=24, color='black')
    plt.ylabel("Loss", fontsize=24, color='black')
    plt.show()

    if save:
        plt.savefig(save)

def getModelHistoryComparisonPlot(histories, names, cv=False, logY=False):
    '''Compare validation loss evolution for several models
    cv=True expects history for CV training and plots mean and 68% CI bands'''
    plt.figure(figsize=(16,8))
    
    for i, (history, name) in enumerate(zip(histories, names)):
        if cv:
            sns.tsplot([history[x]['val_loss'] for x in range(len(history))], condition=name, color=sns.color_palette()[i])
        else:
            plt.plot(history['val_loss'], label=name)

    plt.legend(loc='best', fontsize=16)
    plt.xticks(fontsize=16, color='black')
    plt.yticks(fontsize=16, color='black')
    plt.xlabel("Epoch", fontsize=24, color='black')
    plt.ylabel("Loss", fontsize=24, color='black')
    if logY:
        plt.yscale('log')
        plt.grid(True, which="both")
    plt.show()

def getLRFinderComparisonPlot(lrFinders, names, logX=True, logY=True, loss='loss', cut=-10):
    '''Compare loss evolultion against learning rate for several LRFinder callbacks'''
    plt.figure(figsize=(16,8))
    
    for lrFinder, name in zip(lrFinders, names):
        plt.plot(lrFinder.history['lr'][:cut], lrFinder.history[loss][:cut], label=name)

    plt.legend(loc='best', fontsize=16)
    if logX:
        plt.xscale('log')
        plt.grid(True, which="both")
    if logY:
        plt.yscale('log')
        plt.grid(True, which="both")
    plt.xticks(fontsize=16, color='black')
    plt.yticks(fontsize=16, color='black')
    plt.xlabel("Learning rate", fontsize=24, color='black')
    plt.ylabel("Loss", fontsize=24, color='black')
    plt.show()

def getLRFinderMeanPlot(lrFinders, loss='loss', cut=-10):
    '''Get mean loss evolultion against learning rate for several LRFinder callbacks'''
    plt.figure(figsize=(16,8))
    minLen = np.min([len(lrFinders[x].history[loss][:cut]) for x in range(len(lrFinders))])
    
    sns.tsplot([lrFinders[x].history[loss][:minLen] for x in range(len(lrFinders))],
               time=lrFinders[0].history['lr'][:minLen], ci='sd')

    plt.legend(loc='best', fontsize=16)
    plt.xscale('log')
    plt.grid(True, which="both")
    plt.xticks(fontsize=16, color='black')
    plt.yticks(fontsize=16, color='black')
    plt.xlabel("Learning rate", fontsize=24, color='black')
    plt.ylabel("Loss", fontsize=24, color='black')
    plt.show()

def getMonitorComparisonPlot(monitors, names, xAxis='iter', yAxis='Loss', lrLogX=True, logY=True):
    '''Compare validation loss.accuracy evolution for several models on a per iteration/learning-rate/momentum basis'''
    plt.figure(figsize=(16,8))
    for monitor, name in zip(monitors, names):
        if isinstance(monitor.history['val_loss'][0], list):
            if yAxis == 'Loss':
                y = np.array(monitor.history['val_loss'])[:,0]
            else:
                y = np.array(monitor.history['val_loss'])[:,1]
        else:
            y = monitor.history['val_loss']
                
        if xAxis == 'iter':
            plt.plot(range(len(monitor.history['val_loss'])), y, label=name)
        elif xAxis == 'mom':
            plt.plot(monitor.history['mom'], y, label=name)
        else:
            plt.plot(monitor.history['lr'], y, label=name)

    plt.legend(loc='best', fontsize=16)
    if lrLogX: plt.xscale('log')
    if logY: plt.yscale('log')
    plt.grid(True, which="both")
    plt.xticks(fontsize=16, color='black')
    plt.yticks(fontsize=16, color='black')
    if xAxis == 'iter':
        plt.xlabel("Iteration", fontsize=24, color='black')
    elif xAxis == 'mom':
        plt.xlabel("Momentum", fontsize=24, color='black')
    else:
        plt.xlabel("Learning rate", fontsize=24, color='black')
    plt.ylabel(yAxis, fontsize=24, color='black')
    plt.show()