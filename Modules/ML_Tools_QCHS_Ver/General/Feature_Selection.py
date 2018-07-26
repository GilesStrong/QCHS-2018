from __future__ import division

import numpy as np
import pandas
import types

from xgboost import XGBClassifier

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

import seaborn as sns
import matplotlib.pyplot as plt
sns.set_style("white")

from Modules.ML_Tools_QCHS_Ver.General.PreProc import getPreProcPipes

def rankClassifierFeatures(data, trainFeatures, nSplits=10, nJobs=4, weights=None, target='gen_target', datatype='float32'):
    inputPipe, outputPipe = getPreProcPipes(normIn=True)
    inputPipe.fit(data[trainFeatures].values.astype(datatype))
    X = inputPipe.transform(data[trainFeatures].values.astype(datatype))
    y = data[target].values.astype('int')
    
    if weights != None:
        sig = data[data[target] == 1].index
        bkg = data[data[target] == 0].index
        w = data[weights].values.astype(datatype)
        w[sig] = w[sig]/np.sum(w[sig])
        w[bkg] = w[bkg]/np.sum(w[bkg])

    importantFeatures = []
    featureImportance = {}
    for i in trainFeatures:
        featureImportance[i] = 0

    kf = StratifiedKFold(n_splits=nSplits, shuffle=True)
    folds = kf.split(X, y)
    for i, (train, test) in enumerate(folds):
        print ("Running fold", i+1, "/", nSplits)
        
        xgbClass = XGBClassifier(n_jobs=nJobs)
        if weights != None:
            xgbClass.fit(X[train], y[train], sample_weight=w[train])
            print ('ROC AUC: {:.5f}'.format(roc_auc_score(y[test], xgbClass.predict_proba(X[test])[:,1]), sample_weight=w[test]))
        else:
            xgbClass.fit(X[train], y[train])
            print ('ROC AUC: {:.5f}'.format(roc_auc_score(y[test], xgbClass.predict_proba(X[test])[:,1])))
        
        
        indices = xgbClass.feature_importances_.argsort()
        N = len(indices)
        for n in range(N):
            if xgbClass.feature_importances_[indices[N-n-1]] > 0:
                importantFeatures.append(trainFeatures[indices[N-n-1]])
            featureImportance[trainFeatures[indices[N-n-1]]] += xgbClass.feature_importances_[indices[N-n-1]]

    names = np.array(list(set(importantFeatures)))
    scores = np.array([featureImportance[i]/nSplits for i in names])
    importance = np.array(sorted(zip(names, scores), key=lambda x: x[1], reverse=True))

    print (len(list(set(importantFeatures))), "important features identified")
    print ('Feature\tImportance')
    print ('---------------------')
    for i in importance:
        print (i[0], '\t', i[1])

    return [x[0] for x in importance], [x[1] for x in importance]

def getCorrMat(data0, data1 = None):
    corr = data0.corr()
    
    if not isinstance(data1, type(None)): #Plot difference in correlations
        corr -= data1.corr()

    f, ax = plt.subplots(figsize=(11, 9))
    cmap = sns.diverging_palette(220, 10, as_cmap=True)
    sns.heatmap(corr, cmap=cmap, vmax=1, center=0,
                square=True, linewidths=.5, cbar_kws={"shrink": .5})

def xgCompare(datasets, targets):
    for i in range(len(datasets)):
        X_train, X_test, y_train, y_test = train_test_split(datasets[i], targets[i])
        
        xgb = XGBClassifier(n_jobs=4)
        xgb.fit(X_train, y_train)
        
        trainAUC = roc_auc_score(y_train, xgb.predict_proba(X_train)[:,1])
        testAUC = roc_auc_score(y_test, xgb.predict_proba(X_test)[:,1])
        
        print ("Dataset {}, train:test ROC AUC {:.5f}:{:.5f}".format(i, trainAUC, testAUC))