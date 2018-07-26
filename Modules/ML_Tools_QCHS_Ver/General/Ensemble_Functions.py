from __future__ import division
import numpy as np
import pandas
import math
import json
import os
import types
from six.moves import cPickle as pickle
import glob

from keras.models import Sequential,model_from_json, load_model

#from rep.estimators import XGBoostClassifier

from sklearn.metrics import roc_auc_score

from Modules.ML_Tools_QCHS_Ver.General.Activations import *

def ensemblePredict(inData, ensemble, weights, outputPipe=None, nOut=1, n=-1): #Loop though each classifier and predict data class
    pred = np.zeros((len(inData), nOut))
    if n == -1:
        n = len(ensemble)+1
    ensemble = ensemble[0:n] #Use only specified number of classifiers
    weights = weights[0:n]
    weights = weights/weights.sum() #Renormalise weights
    for i, model in enumerate(ensemble):
        if isinstance(model, Sequential):
            tempPred =  model.predict(inData, verbose=0)
        elif isinstance(model, XGBoostClassifier):
            tempPred = model.predict_proba(inData)[:,1][:, np.newaxis] #Works for one output, might need to be fixed for multiclass
        else:
            print ("MVA not currently supported")
            return None
        if not isinstance(outputPipe, type(None)):
            tempPred = outputPipe.inverse_transform(tempPred)
        pred += weights[i] * tempPred
    return pred

def loadModel(cycle, compileArgs, mva='NN', loadMode='model', location='train_weights/train_'): 
    cycle = int(cycle)
    model = None
    if mva == 'NN':
        if loadMode == 'model':
            model = load_model(location + str(cycle) + '.h5')
        elif loadMode == 'weights':
            model = model_from_json(open(location + str(cycle) + '.json').read())
            model.load_weights(location + str(cycle) + '.h5')
            model.compile(**compileArgs)
        else:
            print ("No other loading currently supported")
    else:
        with open(location + str(cycle) + '.pkl', 'r') as fin:   
            model = pickle.load(fin)
    return model

def getWeights(value, metric, weighting='reciprocal'):
    if weighting == 'reciprocal':
        return 1/value
    if weighting == 'uniform':
        return 1
    else:
        print ("No other weighting currently supported")
    return None

def assembleEnsemble(results, size, metric, compileArgs=None, weighting='reciprocal', mva='NN', loadMode='model', location='train_weights/train_'):
    ensemble = []
    weights = []
    print ("Choosing ensemble by", metric)
    dtype = [('cycle', int), ('result', float)]
    values = np.sort(np.array([(i, result[metric]) for i, result in enumerate(results)], dtype=dtype),
                     order=['result'])
    for i in range(min([size, len(results)])):
        ensemble.append(loadModel(values[i]['cycle'], compileArgs, mva, loadMode, location))
        weights.append(getWeights(values[i]['result'], metric, weighting))
        print ("Model", i, "is", values[i]['cycle'], "with", metric, "=", values[i]['result'])
    weights = np.array(weights)
    weights = weights/weights.sum() #normalise weights
    return ensemble, weights

def saveEnsemble(name, ensemble, weights, compileArgs=None, overwrite=False, inputPipe=None, outputPipe=None, saveMode='model'): #Todo add saving of input feature names
    if (len(glob.glob(name + "*.json")) or len(glob.glob(name + "*.h5")) or len(glob.glob(name + "*.pkl"))) and not overwrite:
        print ("Ensemble already exists with that name, call with overwrite=True to force save")
    else:
        os.system("rm " + name + "*.json")
        os.system("rm " + name + "*.h5")
        os.system("rm " + name + "*.pkl")
        saveCompileArgs = False
        for i, model in enumerate(ensemble):
            if isinstance(model, Sequential):
                saveCompileArgs = True
                if saveMode == 'weights':
                    json_string = model.to_json()
                    open(name + '_' + str(i) + '.json', 'w').write(json_string)
                    model.save_weights(name + '_' + str(i) + '.h5')
                elif saveMode == 'model':
                    model.save(name + '_' + str(i) + '.h5')
                else:
                    print ("No other saving currently supported")
                    return None
            elif isinstance(model, XGBoostClassifier):
                with open(name + '_' + str(i) + '.pkl', 'wb') as fout:
                    pickle.dump(model, fout)
            else:
                print ("MVA not currently supported")
                return None
        if saveCompileArgs:
            with open(name + '_compile.json', 'w') as fout:
                json.dump(compileArgs, fout)
        with open(name + '_weights.pkl', 'wb') as fout:
            pickle.dump(weights, fout)
        if inputPipe != None:
            with open(name + '_inputPipe.pkl', 'wb') as fout:
                pickle.dump(inputPipe, fout)
        if outputPipe != None:
            with open(name + '_outputPipe.pkl', 'wb') as fout:
                pickle.dump(outputPipe, fout)

def loadEnsemble(name, ensembleSize=10, inputPipeLoad=False, outputPipeLoad=False, loadMode='model'): #Todo add loading of input feature names
    ensemble = []
    weights = None
    inputPipe = None
    outputPipe = None
    compileArgs = None
    try:
        with open(name + '_compile.json', 'r') as fin:
            compileArgs = json.load(fin)
    except:
        pass
    for i in range(ensembleSize):
        if len(glob.glob(name + "_" + str(i) + '.pkl')): #BDT
            with open(name + '_' + str(i) + '.pkl', 'rb') as fin:   
                model = pickle.load(fin)    
        else: #NN
            if loadMode == 'weights':
                model = model_from_json(open(name + '_' + str(i) + '.json').read())
                model.load_weights(name + "_" + str(i) + '.h5')
            elif loadMode == 'model':
                model = load_model(name + "_" + str(i) + '.h5')
        ensemble.append(model)
    with open(name + '_weights.pkl', 'rb') as fin:
        weights = pickle.load(fin)
    if inputPipeLoad:
        with open(name + '_inputPipe.pkl', 'rb') as fin:
            inputPipe = pickle.load(fin)
    if outputPipeLoad:
        with open(name + '_outputPipe.pkl', 'rb') as fin:
            outputPipe = pickle.load(fin)
    return ensemble, weights, compileArgs, inputPipe, outputPipe

def testEnsembleAUC(X, y, ensemble, weights, size=10):
    for i in range(size):
        pred = ensemblePredict(X, ensemble, weights, n=i+1)
        auc = roc_auc_score(y, pred)
        print ('Ensemble with {} classifiers, AUC = {:2f}'.format(i+1, auc))