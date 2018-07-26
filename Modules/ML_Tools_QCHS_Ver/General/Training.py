from __future__ import division

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, roc_curve

from keras.models import Sequential, model_from_json, load_model
from keras.layers import Dense, Activation
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras import backend as K
from keras import utils

from six.moves import cPickle as pickle
import timeit
import types
import numpy as np
import os

from Modules.ML_Tools_QCHS_Ver.General.Callbacks import *
from Modules.ML_Tools_QCHS_Ver.General.Misc_Functions import uncertRound
from Modules.ML_Tools_QCHS_Ver.Plotting_And_Evaluation.Plotters import plotTrainingHistory

def trainClassifier(X, y, nSplits, modelGen, modelGenParams, trainParams,
    classWeights='auto', sampleWeights=None, saveLoc='train_weights/', patience=10):
    start = timeit.default_timer()
    results = []
    histories = []
    os.system("mkdir " + saveLoc)
    os.system("rm " + saveLoc + "*.h5")
    os.system("rm " + saveLoc + "*.json")
    os.system("rm " + saveLoc + "*.pkl")

    kf = StratifiedKFold(n_splits=nSplits, shuffle=True)
    folds = kf.split(X, y)

    binary = True
    nClasses = len(np.unique(y))
    if nClasses > 2:
        print (nClasses, "classes found, running in multiclass mode\n")
        y = utils.to_categorical(y, num_classes=nClasses)
        binary = False
        modelGenParams['nOut'] = nClasses
    else:
        print (nClasses, "classes found, running in binary mode\n")

    for i, (train, test) in enumerate(folds):
        print ("Running fold", i+1, "/", nSplits)
        os.system("rm " + saveLoc + "best.h5")
        foldStart = timeit.default_timer()

        model = None
        model = modelGen(**modelGenParams)
        model.reset_states #Just checking

        lossHistory = LossHistory((X[train], y[train]))
        earlyStop = EarlyStopping(monitor='val_loss', patience=patience, verbose=1, mode='auto')
        saveBest = ModelCheckpoint(saveLoc +  "best.h5", monitor='val_loss', verbose=0, save_best_only=True, save_weights_only=True, mode='auto', period=1)
        
        weights=None
        if not isinstance(sampleWeights, types.NoneType):
            weights = sampleWeights[train]
        
        model.fit(X[train], y[train],
                  validation_data = (X[test], y[test]),
                  callbacks = [earlyStop, saveBest, lossHistory],
                  class_weight = classWeights, sample_weight=weights,
                  **trainParams)
        histories.append(lossHistory.losses)
        model.load_weights(saveLoc +  "best.h5")

        results.append({})
        results[-1]['loss'] = model.evaluate(X[test], y[test], verbose=0)
        if binary: results[-1]['AUC'] = 1-roc_auc_score(y[test], model.predict(X[test], verbose=0), sample_weight=weights)
        print ("Score is:", results[-1])

        print("Fold took {:.3f}s\n".format(timeit.default_timer() - foldStart))

        model.save(saveLoc +  'train_' + str(i) + '.h5')
        with open(saveLoc +  'resultsFile.pkl', 'wb') as fout: #Save results
            pickle.dump(results, fout)

    print("\n______________________________________")
    print("Training finished")
    print("Cross-validation took {:.3f}s ".format(timeit.default_timer() - start))
    plotTrainingHistory(histories, saveLoc + 'history.png')

    meanLoss = uncertRound(np.mean([x['loss'] for x in results]), np.std([x['loss'] for x in results])/np.sqrt(len(results)))
    print ("Mean loss = {} +- {}".format(meanLoss[0], meanLoss[1]))
    if binary:
        meanAUC = uncertRound(np.mean([x['AUC'] for x in results]), np.std([x['AUC'] for x in results])/np.sqrt(len(results)))
        print ("Mean AUC = {} +- {}".format(meanAUC[0], meanAUC[1]))
    print("______________________________________\n")

    return results, histories