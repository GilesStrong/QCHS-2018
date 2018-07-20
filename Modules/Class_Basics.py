# from sklearn.model_selection import StratifiedKFold
# from sklearn.preprocessing import StandardScaler
# from sklearn.pipeline import Pipeline
# from sklearn.decomposition import PCA
# from sklearn.metrics import roc_auc_score, roc_curve

# from keras.models import Sequential, model_from_json, load_model
# from keras.layers import Dense, Activation, AlphaDropout, Dropout, BatchNormalization
# from keras.optimizers import Adam
# from keras.models import Sequential

# from six.moves import cPickle as pickle
# import timeit
# import types
# import numpy as np
# import pandas

# #from rep.estimators import XGBoostClassifier
# from xgboost import XGBClassifier

# from ML_Tools.General.PreProc import *
# from ML_Tools.General.Ensemble_Functions import *
# from ML_Tools.Plotting_And_Evaluation.Bootstrap import mpRun
# from ML_Tools.General.Training import *
from ML_Tools.General.Batch_Train import *
from ML_Tools.General.Models import getModel
from ML_Tools.General.BatchYielder import *

class RotationReflectionBatch(BatchYielder):
    def __init__(self, header, datafile=None, inputPipe=None,
                 rotate=True, reflect=True, augRotMult=4,
                 trainTimeAug=True, testTimeAug=True):
        self.header = header
        self.rotateAug = rotate
        self.reflectAug = reflect
        self.augmented = True
        self.augRotMult = augRotMult
        
        if self.rotateAug and not self.reflectAug:
            self.augMult = self.augRotMult
            
        elif not self.rotateAug and self.reflectAug:
            self.reflectAxes = ['_px', '_py', '_pz']
            self.augMult = 8
            
        elif not self.rotateAug and not self.reflectAug:
            self.augmented = False
            trainTimeAug = False
            testTimeAug = False
            self.augMult = 0
            print ('No augmentation specified!')
            inputPipe = None
            self.getTestBatch = self.getBatch
            
        else: #reflect and rotate
            self.reflectAxes = ['_px', '_pz']
            self.augMult = self.augRotMult*4
            
        self.trainTimeAug = trainTimeAug
        self.testTimeAug = testTimeAug
        self.inputPipe = inputPipe
        
        if not isinstance(datafile, type(None)):
            self.addSource(datafile)
    
    def rotate(self, inData, vectors):
        for vector in vectors:
            if 'jet_leading' in vector:
                cut = inData.PRI_jet_num >= 0.9
                inData.loc[cut, vector + '_pxtmp'] = inData.loc[cut, vector + '_px']*np.cos(inData.loc[cut, 'aug_angle'])-inData.loc[:, vector + '_py']*np.sin(inData.loc[cut, 'aug_angle'])
                inData.loc[cut, vector + '_py'] = inData.loc[cut, vector + '_py']*np.cos(inData.loc[cut, 'aug_angle'])+inData.loc[:, vector + '_px']*np.sin(inData.loc[cut, 'aug_angle'])
                inData.loc[cut, vector + '_px'] = inData.loc[cut, vector + '_pxtmp']

            elif 'jet_subleading' in vector:
                cut = inData.PRI_jet_num >= 1.9
                inData.loc[cut, vector + '_pxtmp'] = inData.loc[cut, vector + '_px']*np.cos(inData.loc[cut, 'aug_angle'])-inData.loc[:, vector + '_py']*np.sin(inData.loc[cut, 'aug_angle'])
                inData.loc[cut, vector + '_py'] = inData.loc[cut, vector + '_py']*np.cos(inData.loc[cut, 'aug_angle'])+inData.loc[:, vector + '_px']*np.sin(inData.loc[cut, 'aug_angle'])
                inData.loc[cut, vector + '_px'] = inData.loc[cut, vector + '_pxtmp']
            
            else:
                inData.loc[:, vector + '_pxtmp'] = inData.loc[:, vector + '_px']*np.cos(inData.loc[:, 'aug_angle'])-inData.loc[:, vector + '_py']*np.sin(inData.loc[:, 'aug_angle'])
                inData.loc[:, vector + '_py'] = inData.loc[:, vector + '_py']*np.cos(inData.loc[:, 'aug_angle'])+inData.loc[:, vector + '_px']*np.sin(inData.loc[:, 'aug_angle'])
                inData.loc[:, vector + '_px'] = inData.loc[:, vector + '_pxtmp']
    
    def reflect(self, inData, vectors):
        for vector in vectors:
            for coord in self.reflectAxes:
                try:
                    cut = (inData['aug' + coord] == 1)
                    if 'jet_leading' in vector:
                        cut = cut & (inData.PRI_jet_num >= 0.9)
                    elif 'jet_subleading' in vector:
                        cut = cut & (inData.PRI_jet_num >= 1.9)
                    inData.loc[cut, vector + coord] = -inData.loc[cut, vector + coord]
                except KeyError:
                    pass
            
    def getBatch(self, index, datafile=None):
        if isinstance(datafile, type(None)):
            datafile = self.source
            
        index = str(index)
        weights = None
        targets = None
        if 'fold_' + index + '/weights' in datafile:
            weights = np.array(datafile['fold_' + index + '/weights'])
        if 'fold_' + index + '/targets' in datafile:
            targets = np.array(datafile['fold_' + index + '/targets'])
            
        if not self.augmented:
            return {'inputs':np.array(datafile['fold_' + index + '/inputs']),
                    'targets':targets,
                    'weights':weights}

        if isinstance(self.inputPipe, type(None)):
            inputs = pandas.DataFrame(np.array(datafile['fold_' + index + '/inputs']), columns=self.header)
        else:
            inputs = pandas.DataFrame(self.inputPipe.inverse_transform(np.array(datafile['fold_' + index + '/inputs'])), columns=self.header)            
        
        vectors = [x[:-3] for x in inputs.columns if '_px' in x]
        if self.rotateAug:
            inputs['aug_angle'] = 2*np.pi*np.random.random(size=len(inputs))
            self.rotate(inputs, vectors)
            
        if self.reflectAug:
            for coord in self.reflectAxes:
                inputs['aug' + coord] = np.random.randint(0, 2, size=len(inputs))
            self.reflect(inputs, vectors)
            
        if isinstance(self.inputPipe, type(None)):
            inputs = inputs[self.header].values
        else:
            inputs = self.inputPipe.transform(inputs[self.header].values)
        
        return {'inputs':inputs,
                'targets':targets,
                'weights':weights}
    
    def getTestBatch(self, index, augIndex, datafile=None):
        if augIndex >= self.augMult:
            print ("Invalid augmentation index passed", augIndex)
            return -1
        
        if isinstance(datafile, type(None)):
            datafile = self.source
            
        index = str(index)
        weights = None
        targets = None
        if 'fold_' + index + '/weights' in datafile:
            weights = np.array(datafile['fold_' + index + '/weights'])
        if 'fold_' + index + '/targets' in datafile:
            targets = np.array(datafile['fold_' + index + '/targets'])
            
        if isinstance(self.inputPipe, type(None)):
            inputs = pandas.DataFrame(np.array(datafile['fold_' + index + '/inputs']), columns=self.header)
        else:
            inputs = pandas.DataFrame(self.inputPipe.inverse_transform(np.array(datafile['fold_' + index + '/inputs'])), columns=self.header)            
            
        if self.reflectAug and self.rotateAug:
            rotIndex = augIndex%self.augRotMult
            refIndex = '{0:02b}'.format(int(augIndex/4))
            vectors = [x[:-3] for x in inputs.columns if '_px' in x]
            inputs['aug_angle'] = np.linspace(0, 2*np.pi, (self.augRotMult)+1)[rotIndex]
            for i, coord in enumerate(self.reflectAxes):
                inputs['aug' + coord] = int(refIndex[i])
            self.rotate(inputs, vectors)
            self.reflect(inputs, vectors)
            
        elif self.reflectAug:
            refIndex = '{0:03b}'.format(int(augIndex))
            vectors = [x[:-3] for x in inputs.columns if '_px' in x]
            for i, coord in enumerate(self.reflectAxes):
                inputs['aug' + coord] = int(refIndex[i])
            self.reflect(inputs, vectors)
            
        else:
            vectors = [x[:-3] for x in inputs.columns if '_px' in x]
            inputs['aug_angle'] = np.linspace(0, 2*np.pi, (self.augRotMult)+1)[augIndex]
            self.rotate(inputs, vectors)
            
        if isinstance(self.inputPipe, type(None)):
            inputs = inputs[self.header].values
        else:
            inputs = self.inputPipe.transform(inputs[self.header].values)

        return {'inputs':inputs,
                'targets':targets,
                'weights':weights}