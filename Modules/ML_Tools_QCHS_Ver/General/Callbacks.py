from __future__ import division

from keras.callbacks import Callback
from keras import backend as K

import math
import numpy as np
import types

import seaborn as sns
import matplotlib.pyplot as plt
sns.set_style("whitegrid")

class ValidationMonitor(Callback):
    '''Callback to monitor validation performance and optimiser settings after every minibatch
    For short training diagnosis; slow to run, do not use for full training'''
    def __init__(self, valData=None, valBatchSize=None, mode='sgd'):
        super(ValidationMonitor, self).__init__()
        self.valData = valData
        self.valBatchSize = valBatchSize
        self.mode = mode

    def on_train_begin(self, logs={}):
        self.history = {}
        self.history['loss'] = []
        self.history['val_loss'] = []
        self.history['lr'] = []
        self.history['mom'] = []
        self.history['acc'] = []

    def on_batch_end(self, batch, logs={}):
        self.history['loss'].append(logs.get('loss'))
        self.history['acc'].append(logs.get('acc'))
        
        if not isinstance(self.valData, type(None)):
            mbMask = np.zeros(len(self.valData['y']), dtype=int)
            mbMask[:self.valBatchSize] = 1
            np.random.shuffle(mbMask)
            mbMask = mbMask.astype(bool)
            self.history['val_loss'].append(self.model.evaluate(x=self.valData['x'][mbMask],
                                                                y=self.valData['y'][mbMask],
                                                                sample_weight=self.valData['sample_weight'][mbMask], 
                                                                verbose=0))
                
        self.history['lr'].append(float(K.get_value(self.model.optimizer.lr)))

        if self.mode == 'sgd':
            self.history['mom'].append(float(K.get_value(self.model.optimizer.momentum)))
        elif self.mode == 'adam':
            self.history['mom'].append(float(K.get_value(self.model.optimizer.beta_1)))

class LossHistory(Callback):
    def __init__(self, trData):
        self.trainingData = trData

    def on_train_begin(self, logs={}):
        self.losses = {}
        self.losses['loss'] = []
        self.losses['val_loss'] = []

    def on_epoch_end(self, epoch, logs={}):
        self.losses['loss'].append(self.model.evaluate(self.trainingData[0], self.trainingData[1], verbose=0))
        self.losses['val_loss'].append(logs.get('val_loss'))

class LRFinder(Callback):
    '''Learning rate finder callback 
    - adapted from fastai version to work in Keras and to optionally run over validation data'''

    def __init__(self, nSteps, valData=None, valBatchSize=None, lrBounds=[1e-7, 10], verbose=0):
        super(LRFinder, self).__init__()
        self.verbose = verbose
        self.lrBounds=lrBounds
        ratio = self.lrBounds[1]/self.lrBounds[0]
        self.lr_mult = ratio**(1/nSteps)
        self.valData = valData
        self.valBatchSize = valBatchSize
        
    def on_train_begin(self, logs={}):
        self.best=1e9
        self.iter = 0
        K.set_value(self.model.optimizer.lr, self.lrBounds[0])
        self.history = {}
        self.history['loss'] = []
        self.history['val_loss'] = []
        self.history['lr'] = []
        
    def calc_lr(self, lr, batch):
        return self.lrBounds[0]*(self.lr_mult**batch)
    
    def plot(self, n_skip=0, n_max=-1, yLim=None):
        plt.figure(figsize=(16,8))
        plt.plot(self.history['lr'][n_skip:n_max], self.history['loss'][n_skip:n_max], label='Training loss', color='g')
        if not isinstance(self.valData, type(None)):
            plt.plot(self.history['lr'][n_skip:n_max], self.history['val_loss'][n_skip:n_max], label='Validation loss', color='b')
        
        if np.log10(self.lrBounds[1])-np.log10(self.lrBounds[0]) >= 3:
            plt.xscale('log')
        plt.ylim(yLim)
        plt.grid(True, which="both")
        plt.legend(loc='best', fontsize=16)
        plt.xticks(fontsize=16, color='black')
        plt.yticks(fontsize=16, color='black')
        plt.ylabel("Loss", fontsize=24, color='black')
        plt.xlabel("Learning rate", fontsize=24, color='black')
        plt.show()
        
    def plot_lr(self):
        plt.figure(figsize=(4,4))
        plt.xlabel("Iterations")
        plt.ylabel("Learning rate")
        plt.plot(range(len(self.history['lr'])), self.history['lr'])
        plt.show()
    
    def plot_genError(self):
        plt.figure(figsize=(16,8))
        plt.xlabel("Iterations")
        plt.ylabel("Generalisation Error")
        plt.plot(range(len(self.history['lr'])), np.array(self.history['val_loss'])-np.array(self.history['loss']))
        plt.show()

    def on_batch_end(self, batch, logs={}):
        loss = logs.get('loss')
        self.history['loss'].append(logs.get('loss'))
        
        if not isinstance(self.valData, type(None)):
            mbMask = np.zeros(len(self.valData['y']), dtype=int)
            mbMask[:self.valBatchSize] = 1
            np.random.shuffle(mbMask)
            mbMask = mbMask.astype(bool)
            self.history['val_loss'].append(self.model.evaluate(x=self.valData['x'][mbMask],
                                                                y=self.valData['y'][mbMask],
                                                                sample_weight=self.valData['sample_weight'][mbMask], 
                                                                verbose=0))
                
        self.history['lr'].append(float(K.get_value(self.model.optimizer.lr)))
        
        self.iter += 1
        lr = self.calc_lr(float(K.get_value(self.model.optimizer.lr)), self.iter)
        K.set_value(self.model.optimizer.lr, lr)
        if self.verbose > 0:
            print('Batch %05d: LearningRateFinder increasing learning '
                  'rate to %s.' % (self.iter, lr))
        
        if math.isnan(loss) or loss>self.best*10:
            if self.verbose > 0:
                print('Ending training early due to loss increase')
            self.model.stop_training = True
        if (loss<self.best and self.iter>10): self.best=loss

class LinearCLR(Callback):
    '''Cyclical learning rate callback with linear interpolation'''
    def __init__(self, nb, maxLR, minLR, scale=2, reverse=False, plotLR=False):
        super(LinearCLR, self).__init__()
        self.nb = nb*scale
        self.cycle_iter = 0
        self.cycle_count = 0
        self.lrs = []
        self.maxLR = maxLR
        self.minLR = minLR
        self.reverse = reverse
        self.cycle_end = False
        self.plotLR = plotLR

    def on_train_begin(self, logs={}):
        self.cycle_end = False

    def on_train_end(self, logs={}):
        if self.plotLR:
            self.plot_lr()
        
    def plot_lr(self):
        plt.figure(figsize=(16,8))
        plt.xlabel("iterations", fontsize=24, color='black')
        plt.ylabel("learning rate", fontsize=24, color='black')
        plt.xticks(fontsize=16, color='black')
        plt.yticks(fontsize=16, color='black')
        plt.plot(range(len(self.lrs)), self.lrs)
        plt.show()
        
    def calc_lr(self, batch):
        cycle = math.floor(1+(self.cycle_iter/(2*self.nb)))
        x = np.abs((self.cycle_iter/self.nb)-(2*cycle)+1)
        lr = self.minLR+((self.maxLR-self.minLR)*np.max([0, 1-x]))

        self.cycle_iter += 1
        if self.reverse:
            return self.maxLR-(lr-self.minLR)
        else:
            return lr

    def on_batch_end(self, batch, logs={}):
        lr = self.calc_lr(batch)
        self.lrs.append(lr)
        K.set_value(self.model.optimizer.lr, lr)

class CosAnneal(Callback):
    '''Adapted from fastai version'''
    def __init__(self, nb, cycle_mult=1, reverse=False):
        super(CosAnneal, self).__init__()
        self.nb = nb
        self.cycle_mult = cycle_mult
        self.cycle_iter = 0
        self.cycle_count = 0
        self.lrs = []
        self.lr = -1
        self.reverse = reverse
        self.cycle_end = False

    def on_train_begin(self, logs={}):
        if self.lr == -1:
            self.lr = float(K.get_value(self.model.optimizer.lr))
        self.cycle_end = False
        
    def plot_lr(self):
        plt.figure(figsize=(16,8))
        plt.xlabel("iterations")
        plt.ylabel("learning rate")
        plt.plot(range(len(self.lrs)), self.lrs)
        plt.show()
        
    def calc_lr(self, batch):
        '''if batch < self.nb/20:
            self.cycle_iter += 1
            return self.lr/100.'''

        cos_out = np.cos(np.pi*(self.cycle_iter)/self.nb) + 1
        self.cycle_iter += 1
        if self.cycle_iter==self.nb:
            self.cycle_iter = 0
            self.nb *= self.cycle_mult
            self.cycle_count += 1
            self.cycle_end = True
        if self.reverse:
            return self.lr-(self.lr / 2 * cos_out)
        else:
            return self.lr / 2 * cos_out

    def on_batch_end(self, batch, logs={}):
        lr = self.calc_lr(batch)
        self.lrs.append(lr)
        K.set_value(self.model.optimizer.lr, lr)

class LinearCMom(Callback):
    '''Cyclical momentum callback with linear interpolation'''
    def __init__(self, nb, maxMom, minMom, scale=2, reverse=False, plotMom=False, mode='sgd'):
        super(LinearCMom, self).__init__()
        self.nb = nb*scale
        self.cycle_iter = 0
        self.cycle_count = 0
        self.moms = []
        self.maxMom = maxMom
        self.minMom = minMom
        self.reverse = reverse
        self.cycle_end = False
        self.plotMom = plotMom
        self.mode = mode

    def on_train_begin(self, logs={}):
        self.cycle_end = False

    def on_train_end(self, logs={}):
        if self.plotMom:
            self.plot_mom()
        
    def plot_mom(self):
        plt.figure(figsize=(16,8))
        plt.xlabel("iterations", fontsize=24, color='black')
        plt.ylabel("momentum", fontsize=24, color='black')
        plt.xticks(fontsize=16, color='black')
        plt.yticks(fontsize=16, color='black')
        plt.plot(range(len(self.moms)), self.moms)
        plt.show()
        
    def calc_mom(self, batch):
        cycle = math.floor(1+(self.cycle_iter/(2*self.nb)))
        x = np.abs((self.cycle_iter/self.nb)-(2*cycle)+1)
        mom = self.minMom+((self.maxMom-self.minMom)*np.max([0, 1-x]))

        self.cycle_iter += 1
        if not self.reverse:
            return self.maxMom-(mom-self.minMom)
        else:
            return mom

    def on_batch_end(self, batch, logs={}):
        mom = self.calc_mom(batch)
        self.moms.append(mom)
        if self.mode == 'sgd':
            K.set_value(self.model.optimizer.momentum, mom)
        elif self.mode == 'adam':
            K.set_value(self.model.optimizer.beta_1, mom)

class CosAnnealMomentum(Callback):
    def __init__(self, nb, cycle_mult=1, reverse=False):
        super(CosAnnealMomentum, self).__init__()
        self.nb = nb
        self.cycle_mult = cycle_mult
        self.cycle_iter = 0
        self.cycle_count = 0
        self.moms = []
        self.momentum = -1
        self.reverse = reverse
        self.cycle_end = False

    def on_train_begin(self, logs={}):
        if self.momentum == -1:
            self.momentum = float(K.get_value(self.model.optimizer.momentum))
        self.cycle_end = False
        
    def plot_momentum(self):
        plt.figure(figsize=(16,8))
        plt.xlabel("iterations")
        plt.ylabel("momentum")
        plt.plot(range(len(self.moms)), self.moms)
        plt.show()
        
    def calc_momentum(self, batch):
        cos_out = np.cos(np.pi*(self.cycle_iter)/self.nb) + 1
        self.cycle_iter += 1
        if self.cycle_iter==self.nb:
            self.cycle_iter = 0
            self.nb *= self.cycle_mult
            self.cycle_count += 1
            self.cycle_end = True
        if self.reverse:
            return self.momentum-(self.momentum / 10 * cos_out)
        else:
            return (self.momentum-(self.momentum / 5))+self.momentum / 10 * cos_out

    def on_batch_end(self, batch, logs={}):
        momentum = self.calc_momentum(batch)
        self.moms.append(momentum)
        K.set_value(self.model.optimizer.momentum, momentum)

class OneCycle(Callback):
    def __init__(self, nb, scale=30, ratio=0.5, reverse=False, lrScale=10, momScale=0.1, mode='sgd'):
        '''nb=number of minibatches per epoch, ratio=fraction of epoch spent in first stage,
           lrScale=number used to divide initial LR to get minimum LR,
           momScale=number to subtract from initial momentum to get minimum momentum'''
        super(OneCycle, self).__init__()
        self.nb = nb*scale
        self.ratio = ratio
        self.nSteps = (math.ceil(self.nb*self.ratio), math.floor((1-self.ratio)*self.nb))
        self.cycle_iter = 0
        self.cycle_count = 0
        self.lrs = []
        self.moms = []
        self.momentum = -1
        self.lr = -1
        self.reverse = reverse
        self.cycle_end = False
        self.lrScale = lrScale
        self.momScale = momScale
        self.momStep1 = -self.momScale/float(self.nSteps[0])
        self.momStep2 = self.momScale/float(self.nSteps[1])
        self.mode = mode.lower()

    def on_train_begin(self, logs={}):
        if self.momentum == -1:
            if self.mode == 'sgd':
                self.momMax = float(K.get_value(self.model.optimizer.momentum))
            elif self.mode == 'adam':
                self.momMax = float(K.get_value(self.model.optimizer.beta_1))
            self.momMin = self.momMax-self.momScale
            if self.reverse: 
                self.momentum = self.momMin
                self.momStep1 *= -1
                self.momStep2 *= -1
                if self.mode == 'sgd':
                    K.set_value(self.model.optimizer.momentum, self.momentum)
                elif self.mode == 'adam':
                    K.set_value(self.model.optimizer.beta_1, self.momentum)
            else:
                self.momentum = self.momMax

            self.momStep = self.momStep1

        if self.lr == -1:
            self.lrMax = float(K.get_value(self.model.optimizer.lr))
            self.lrMin = self.lrMax/self.lrScale
            self.lrStep1 = (self.lrMax-self.lrMin)/self.nSteps[0]
            self.lrStep2 = -(self.lrMax-self.lrMin)/self.nSteps[1]
            if self.reverse:
                self.lrStep1 *= -1
                self.lrStep2 *= -1
                self.lr = self.lrMax
            else:
                self.lr = self.lrMin
                K.set_value(self.model.optimizer.lr, self.lr)

            self.lrStep = self.lrStep1

        self.moms.append(self.momentum)
        self.lrs.append(self.lr)
        self.cycle_end = False
        
    def plot(self):
        fig, axs = plt.subplots(2,1,figsize=(16,4))
        for ax in axs:
            ax.set_xlabel("Iterations")
        axs[0].set_ylabel("Learning Rate")
        axs[1].set_ylabel("Momentum")
        axs[0].plot(range(len(self.lrs)), self.lrs)
        axs[1].plot(range(len(self.moms)), self.moms)
        plt.show()
        
    def calc(self, batch):
        if self.cycle_iter == self.nSteps[0]+1:
            self.lrStep = self.lrStep2
            self.momStep = self.momStep2

        if self.cycle_iter>=self.nb:
            self.lr/2

        else:
            self.momentum += self.momStep
            self.lr += self.lrStep

        self.moms.append(self.momentum)
        self.lrs.append(self.lr)

    def on_batch_end(self, batch, logs={}):
        self.cycle_iter += 1
        self.calc(batch)
        if self.mode == 'sgd':
            K.set_value(self.model.optimizer.momentum, self.momentum)
        elif self.mode == 'adam':
            K.set_value(self.model.optimizer.beta_1, self.momentum)
        K.set_value(self.model.optimizer.lr, self.lr)

class SWA(Callback):
    '''Modified from fastai version'''
    def __init__(self, swa_start, testBatch, testModel, verbose=False, swaRenewal=-1,
                 clrCallback=None, trainOnWeights=False, sgdReplacement=False):
        super(SWA, self).__init__()
        self.swa_model = None
        self.swa_model_new = None
        self.swa_start = swa_start
        self.epoch = -1
        self.swa_n = -1
        self.swaRenewal = swaRenewal
        self.n_since_renewal = -1
        self.losses = {'swa':None, 'base':None}
        self.active = False
        self.testBatch = testBatch
        self.weighted = trainOnWeights
        self.clrCallback = clrCallback
        self.test_model = testModel
        self.verbose = verbose
        self.sgdReplacement = sgdReplacement
        
    def on_train_begin(self, logs={}):
        if isinstance(self.swa_model, type(None)):
            self.swa_model = self.model.get_weights()
            self.swa_model_new = self.model.get_weights()
            self.epoch = 0
            self.swa_n = 0
            self.n_since_renewal = 0
            self.first_completed= False
            self.cylcle_since_replacement = 1
            
    def on_epoch_begin(self, metrics, logs={}):
        self.losses = {'swa':None, 'base':None}

    def on_epoch_end(self, metrics, logs={}):
        if (self.epoch + 1) >= self.swa_start and (isinstance(self.clrCallback, type(None)) or self.clrCallback.cycle_end):
            if self.swa_n == 0 and not self.active:
                print ("SWA beginning")
                self.active = True
            elif not isinstance(self.clrCallback, type(None)) and self.clrCallback.cycle_mult > 1:
                print ("Updating average")
                self.active = True
            self.update_average_model()
            self.swa_n += 1
            
            if self.swa_n > self.swaRenewal:
                self.first_completed = True
                self.n_since_renewal += 1
                if self.n_since_renewal > self.cylcle_since_replacement*self.swaRenewal and self.swaRenewal > 0:
                    self.compareAverages()
            
        if isinstance(self.clrCallback, type(None)) or self.clrCallback.cycle_end:
            self.epoch += 1

        if self.active and not (isinstance(self.clrCallback, type(None)) or self.clrCallback.cycle_end or self.clrCallback.cycle_mult == 1):
            self.active = False
            
    def update_average_model(self):
        # update running average of parameters
        print("model is {} epochs old".format(self.swa_n))
        for model_param, swa_param in zip(self.model.get_weights(), self.swa_model):
            swa_param *= self.swa_n
            swa_param += model_param
            swa_param /= (self.swa_n + 1)
        
        if self.swa_n > self.swaRenewal and self.first_completed:
            print("new model is {} epochs old".format(self.n_since_renewal))
            for model_param, swa_param in zip(self.model.get_weights(), self.swa_model_new):
                swa_param *= self.n_since_renewal
                swa_param += model_param
                swa_param /= (self.n_since_renewal + 1)
            
    def compareAverages(self):
        if isinstance(self.losses['swa'], type(None)):
            self.test_model.set_weights(self.swa_model)
            if self.weighted:
                self.losses['swa'] = self.test_model.evaluate(self.testBatch['inputs'], self.testBatch['targets'], sample_weight=self.testBatch['weights'], verbose=0)
            else:
                self.losses['swa'] = self.test_model.evaluate(self.testBatch['inputs'], self.testBatch['targets'], verbose=0)
        
        self.test_model.set_weights(self.swa_model_new)
        if self.weighted:
            new_loss = self.test_model.evaluate(self.testBatch['inputs'], self.testBatch['targets'], sample_weight=self.testBatch['weights'], verbose=0)
        else:
            new_loss = self.test_model.evaluate(self.testBatch['inputs'], self.testBatch['targets'], verbose=0)
        
        print("Checking renewal swa model, current model: {}, new model: {}".format(self.losses['swa'], new_loss))
        if new_loss < self.losses['swa']:
            print("New model better, replacing\n____________________\n\n")
            self.losses['swa'] = new_loss
            self.swa_n = self.n_since_renewal
            if self.sgdReplacement:
                if isinstance(self.losses['base'], type(None)):
                    if self.weighted:
                        self.losses['base'] = self.model.evaluate(self.testBatch['inputs'], self.testBatch['targets'], sample_weight=self.testBatch['weights'], verbose=0)
                    else:
                        self.losses['base'] = self.model.evaluate(self.testBatch['inputs'], self.testBatch['targets'], verbose=0)
                if self.losses['base'] > new_loss:
                    print("Old average better than current point, starting SGD from old average")
                    self.model.set_weights(self.swa_model)
                    self.n_since_renewal = 0
                else:
                    print("Old average worse than current point, resuming SGD from current point")
                    self.n_since_renewal = 1
            else:
                self.n_since_renewal = 1
            self.swa_model[:] = self.swa_model_new
            self.swa_model_new = self.model.get_weights()
            self.cylcle_since_replacement = 1

        else:
            print("Current model better, renewing\n____________________\n\n")
            self.swa_model_new = self.model.get_weights()
            self.n_since_renewal = 1
            self.test_model.set_weights(self.swa_model)
            self.cylcle_since_replacement += 1
                
    
    def get_losses(self):
        if isinstance(self.losses['swa'], type(None)):
            self.test_model.set_weights(self.swa_model)
            if self.weighted:
                self.losses['swa'] = self.test_model.evaluate(self.testBatch['inputs'], self.testBatch['targets'], sample_weight=self.testBatch['weights'], verbose=0)
            else:
                self.losses['swa'] = self.test_model.evaluate(self.testBatch['inputs'], self.testBatch['targets'], verbose=0)
        
        if isinstance(self.losses['base'], type(None)):
            if self.weighted:
                self.losses['base'] = self.model.evaluate(self.testBatch['inputs'], self.testBatch['targets'], sample_weight=self.testBatch['weights'], verbose=0)
            else:
                self.losses['base'] = self.model.evaluate(self.testBatch['inputs'], self.testBatch['targets'], verbose=0)
        
        return self.losses