from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA

def getPreProcPipes(normIn=False, normOut=False, pca=False, whiten=False, normPCA=False):
    stepsIn = []
    if not normIn and not pca:
        stepsIn.append(('ident', StandardScaler(with_mean=False, with_std=False))) #For compatability
    else:
        if normIn:
            stepsIn.append(('normIn', StandardScaler()))
        if pca:
            stepsIn.append(('pca', PCA(whiten=whiten)))
            if normPCA:
                stepsIn.append(('normPCA', StandardScaler()))
    inputPipe = Pipeline(stepsIn)
    stepsOut = []
    if normOut:
        stepsOut.append(('normOut', StandardScaler()))
    else:
        stepsOut.append(('ident', StandardScaler(with_mean=False, with_std=False))) #For compatability
    outputPipe = Pipeline(stepsOut)
    return inputPipe, outputPipe