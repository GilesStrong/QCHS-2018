from __future__ import division
import numpy as np
import pandas
import math
import os
import types
import h5py
from six.moves import cPickle as pickle

import seaborn as sns
import matplotlib.pyplot as plt
sns.set_style("whitegrid")

from Modules.ML_Tools_QCHS_Ver.Plotting_And_Evaluation.Plotters import *
from Modules.ML_Tools_QCHS_Ver.General.Misc_Functions import *
from Modules.ML_Tools_QCHS_Ver.General.Metrics import *
from Modules.ML_Tools_QCHS_Ver.General.Batch_Train import getFeature

dirLoc = "../Data/"

def scoreTestOD(testData, cut):
    data = pandas.DataFrame()
    data['pred_class'] = getFeature('pred', testData)
    data['gen_weight'] = getFeature('weights', testData)
    data['gen_target'] = getFeature('targets', testData)
    data['private'] = getFeature('private', testData)
    
    accept = (data.pred_class >= cut)
    signal = (data.gen_target == 1)
    bkg = (data.gen_target == 0)
    public = (data.private == 0)
    private = (data.private == 1)
    
    publicAMS = AMS(np.sum(data.loc[accept & public & signal, 'gen_weight']),
                    np.sum(data.loc[accept & public & bkg, 'gen_weight']))
    
    privateAMS = AMS(np.sum(data.loc[accept & private & signal, 'gen_weight']),
                    np.sum(data.loc[accept & private & bkg, 'gen_weight']))
    
    print("Public:Private AMS: {} : {}".format(publicAMS, privateAMS))    
    return publicAMS, privateAMS

def saveTest(cut, name, dirLoc=dirLoc):
    testData = h5py.File(dirLoc + 'testing.hdf5', "r+")
    
    data = pandas.DataFrame()
    data['EventId'] = getFeature('EventId', testData)
    data['pred_class'] = getFeature('pred', testData)
    
    data['Class'] = 'b'
    data.loc[data.pred_class >= cut, 'Class'] = 's'

    data.sort_values(by=['pred_class'], inplace=True)
    data['RankOrder']=range(1, len(data)+1)
    data.sort_values(by=['EventId'], inplace=True)

    print (dirLoc + name + '_test.csv')
    data.to_csv(dirLoc + name + '_test.csv', columns=['EventId', 'RankOrder', 'Class'], index=False)

def convertToDF(datafile, nLoad=-1, setFold=-1):
    data = pandas.DataFrame()
    data['gen_target'] = getFeature('targets', datafile, nLoad, setFold=setFold)
    data['gen_weight'] = getFeature('weights', datafile, nLoad, setFold=setFold)
    data['pred_class'] = getFeature('pred', datafile, nLoad, setFold=setFold)
    print (len(data), "candidates loaded")
    return data

