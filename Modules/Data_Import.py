import pandas
from six.moves import cPickle as pickle
import numpy as np
import optparse
import os
import h5py

from sklearn.model_selection import train_test_split, StratifiedKFold, KFold

from Modules.ML_Tools_QCHS_Ver.Transformations.HEP_Proc import *
from Modules.ML_Tools_QCHS_Ver.General.PreProc import getPreProcPipes

def importData(dirLoc = "../Data/",
               rotate=False, cartesian=True, mode='OpenData',
               valSize=0.2, seed=None):
    '''Import and split data from CSV(s)'''
    if mode == 'OpenData': #If using data from CERN Open Access
        data = pandas.read_csv(dirLoc + 'atlas-higgs-challenge-2014-v2.csv')
        data.rename(index=str, columns={"KaggleWeight": "gen_weight", 'PRI_met': 'PRI_met_pt'}, inplace=True)
        data.drop(columns=['Weight'], inplace=True)
        trainingData = pandas.DataFrame(data.loc[data.KaggleSet == 't'])
        trainingData.drop(columns=['KaggleSet'], inplace=True)
        
        test = pandas.DataFrame(data.loc[(data.KaggleSet == 'b') | (data.KaggleSet == 'v')])
        test['private'] = 0
        test.loc[(data.KaggleSet == 'v'), 'private'] = 1
        test['gen_target'] = 0
        test.loc[test.Label == 's', 'gen_target'] = 1
        test.drop(columns=['KaggleSet', 'Label'], inplace=True)

    else: # If using data from Kaggle
        trainingData = pandas.read_csv(dirLoc + 'training.csv')
        trainingData.rename(index=str, columns={"Weight": "gen_weight", 'PRI_met': 'PRI_met_pt'}, inplace=True)
        test = pandas.read_csv(dirLoc + 'test.csv')
        test.rename(index=str, columns={'PRI_met': 'PRI_met_pt'}, inplace=True)


    convertData(trainingData, rotate, cartesian)
    convertData(test, rotate, cartesian)

    trainingData['gen_target'] = 0
    trainingData.loc[trainingData.Label == 's', 'gen_target'] = 1
    trainingData.drop(columns=['Label'], inplace=True)
    trainingData['gen_weight_original'] = trainingData['gen_weight'] #gen_weight might be renormalised

    trainFeatures = [x for x in trainingData.columns if 'gen' not in x and x != 'EventId' and 'kaggle' not in x.lower()]
    trainIndeces, valIndeces = train_test_split([i for i in trainingData.index.tolist()], test_size=valSize, random_state=seed)
    train = trainingData.loc[trainIndeces]
    val = trainingData.loc[valIndeces]
    print('Training on {} datapoints and validating on {}, using {} features:\n{}'.format(len(train), len(val), len(trainFeatures), [x for x in trainFeatures]))

    return {'train':train[trainFeatures + ['gen_target', 'gen_weight', 'gen_weight_original']], 
           'val':val[trainFeatures + ['gen_target', 'gen_weight', 'gen_weight_original']],
           'test':test,
           'features':trainFeatures}

def rotateEvent(inData):
    '''Rotate event in phi such that lepton is at phi == 0'''
    inData['PRI_tau_phi'] = deltaphi(inData['PRI_lep_phi'], inData['PRI_tau_phi'])
    inData['PRI_jet_leading_phi'] = deltaphi(inData['PRI_lep_phi'], inData['PRI_jet_leading_phi'])
    inData['PRI_jet_subleading_phi'] = deltaphi(inData['PRI_lep_phi'], inData['PRI_jet_subleading_phi'])
    inData['PRI_met_phi'] = deltaphi(inData['PRI_lep_phi'], inData['PRI_met_phi'])
    inData['PRI_lep_phi'] = 0

def zFlipEvent(inData):
    '''Flip event in z-axis such that primary lepton is in positive z-direction'''
    cut = (inData.PRI_lep_eta < 0)
    
    for particle in ['PRI_lep', 'PRI_tau', 'PRI_jet_leading', 'PRI_jet_subleading']:
        inData.loc[cut, particle + '_eta'] = -inData.loc[cut, particle + '_eta'] 

def xFlipEvent(inData):
    '''Flip event in x-axis such that (subleading) (leptoninc) tau is in positive x-direction'''
    cut = (inData.PRI_tau_phi < 0)
    
    for particle in ['PRI_tau', 'PRI_jet_leading', 'PRI_jet_subleading', 'PRI_met']:
        inData.loc[cut, particle + '_phi'] = -inData.loc[cut, particle + '_phi'] 
    
def convertData(inData, rotate=False, cartesian=True):
    '''Pass data through conversions and drop uneeded columns'''
    inData.replace([np.inf, -np.inf], np.nan, inplace=True)
    inData.fillna(-999.0, inplace=True)
    inData.replace(-999.0, 0.0, inplace=True)
    
    if rotate:
        rotateEvent(inData)
        zFlipEvent(inData)
        xFlipEvent(inData)
    
    if cartesian:
        moveToCartesian(inData, 'PRI_tau', drop=True)
        moveToCartesian(inData, 'PRI_lep', drop=True)
        moveToCartesian(inData, 'PRI_jet_leading', drop=True)
        moveToCartesian(inData, 'PRI_jet_subleading', drop=True)
        moveToCartesian(inData, 'PRI_met', z=False)
        
        inData.drop(columns=["PRI_met_phi"], inplace=True)
        
    if rotate and not cartesian:
        inData.drop(columns=["PRI_lep_phi"], inplace=True)
    elif rotate and cartesian:
        inData.drop(columns=["PRI_lep_py"], inplace=True)

def saveBatch(inData, n, inputPipe, outFile, normWeights, mode, features):
    '''Save fold into hdf5 file'''
    grp = outFile.create_group('fold_' + str(n))
    
    X = inputPipe.transform(inData[features].values.astype('float32'))
    inputs = grp.create_dataset("inputs", shape=X.shape, dtype='float32')
    inputs[...] = X
    
    if mode != 'testing':
        if normWeights:
            inData.loc[inData.gen_target == 0, 'gen_weight'] = inData.loc[inData.gen_target == 0, 'gen_weight']/np.sum(inData.loc[inData.gen_target == 0, 'gen_weight'])
            inData.loc[inData.gen_target == 1, 'gen_weight'] = inData.loc[inData.gen_target == 1, 'gen_weight']/np.sum(inData.loc[inData.gen_target == 1, 'gen_weight'])

        y = inData['gen_target'].values.astype('int')
        targets = grp.create_dataset("targets", shape=y.shape, dtype='int')
        targets[...] = y

        X_weights = inData['gen_weight'].values.astype('float32')
        weights = grp.create_dataset("weights", shape=X_weights.shape, dtype='float32')
        weights[...] = X_weights

        X_orig_weights = inData['gen_weight_original'].values.astype('float32')
        orig_weights = grp.create_dataset("orig_weights", shape=X_weights.shape, dtype='float32')
        orig_weights[...] = X_orig_weights
    
    else:
        X_EventId = inData['EventId'].values.astype('int')
        EventId = grp.create_dataset("EventId", shape=X_EventId.shape, dtype='int')
        EventId[...] = X_EventId

        if 'private' in inData.columns:
            X_weights = inData['gen_weight'].values.astype('float32')
            weights = grp.create_dataset("weights", shape=X_weights.shape, dtype='float32')
            weights[...] = X_weights

            X_set = inData['private'].values.astype('int')
            KaggleSet = grp.create_dataset("private", shape=X_set.shape, dtype='int')
            KaggleSet[...] = X_set

            y = inData['gen_target'].values.astype('int')
            targets = grp.create_dataset("targets", shape=y.shape, dtype='int')
            targets[...] = y

def prepareSample(inData, mode, inputPipe, normWeights, N, features, dirLoc):
    '''Split data sample into folds and save to hdf5'''
    print ("Running", mode)
    os.system('rm ' + dirLoc + mode + '.hdf5')
    outFile = h5py.File(dirLoc + mode + '.hdf5', "w")

    if mode != 'testing':
        kf = StratifiedKFold(n_splits=N, shuffle=True)
        folds = kf.split(inData, inData['gen_target'])
    else:
        kf = KFold(n_splits=N, shuffle=True)
        folds = kf.split(inData)

    for i, (train, test) in enumerate(folds):
        print ("Saving fold:", i, "of", len(test), "events")
        saveBatch(inData.iloc[test], i, inputPipe, outFile, normWeights, mode, features)

def runDataImport(dirLoc, rotate, cartesian, mode, valSize, seed, nFolds):
    '''Run through all the stages to save the data into files for training, validation, and testing'''
    #Get Data
    data = importData(dirLoc, rotate, cartesian, mode, valSize, seed)

    #Standardise and normalise
    inputPipe, _ = getPreProcPipes(normIn=True)
    inputPipe.fit(data['train'][data['features']].values.astype('float32'))
    with open(dirLoc + 'inputPipe.pkl', 'wb') as fout:
        pickle.dump(inputPipe, fout)

    prepareSample(data['train'], 'train', inputPipe, True, nFolds, data['features'], dirLoc)
    prepareSample(data['val'], 'val', inputPipe, False, nFolds, data['features'], dirLoc)
    prepareSample(data['test'], 'testing', inputPipe, False, nFolds, data['features'], dirLoc)

    with open(dirLoc + 'features.pkl', 'wb') as fout:
        pickle.dump(data['features'], fout)

if __name__ == '__main__':
    parser = optparse.OptionParser(usage = __doc__)
    parser.add_option("-d", "--dirLoc", dest = "dirLoc", action = "store", default = "./Data/", help = "Data folder location")
    parser.add_option("-r", "--rotate", dest = "rotate", action = "store", default = False, help = "Rotate events to have common alignment")
    parser.add_option("-c", "--cartesian", dest = "cartesian", action = "store", default = True, help = "Convert to Cartesian system")
    parser.add_option("-m", "--mode", dest = "mode", action = "store", default = "OpenData", help = "Using open data or Kaggle data")
    parser.add_option("-v", "--valSize", dest = "valSize", action = "store", default = 0.2, help = "Fraction of data to use for validation")
    parser.add_option("-s", "--seed", dest = "seed", action = "store", default = 1337, help = "Seed for train/val split")
    parser.add_option("-n", "--nFolds", dest = "nFolds", action = "store", default = 10, help = "Nmber of folds to split data")
    opts, args = parser.parse_args()

    runDataImport(opts.dirLoc, opts.rotate, opts.cartesian, opts.mode, opts.valSize, opts.seed, opts.nFolds)