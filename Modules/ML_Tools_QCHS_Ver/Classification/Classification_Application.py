from __future__ import division
import numpy as np
import pandas
import math
import json
import glob
from six.moves import cPickle as pickle

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

import theano
from keras.models import Sequential

from ML_Tools.General.Ensemble_Functions import *

class Classifier():
    def __init__(self, name, features, inputPipeLoad=True):
        self.ensemble = []
        self.weights = None
        self.inputPipe = None
        self.compileArgs = None
        self.inputFeatures = features
        self.ensemble, self.weights, self.compileArgs, self.inputPipe, outputPipe = loadEnsemble(name, inputPipeLoad=inputPipeLoad)
        
    def predict(self, inData):
        return ensemblePredict(self.inputPipe.transform(inData[self.inputFeatures].values.astype('float64')),
            self.ensemble, self.weights, n=len(self.ensemble))[:,0]