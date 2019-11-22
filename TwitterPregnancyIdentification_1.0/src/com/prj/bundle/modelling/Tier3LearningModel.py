'''
Created on Jun 23, 2018

@author: neha
'''

import numpy as np
import sys
from keras.models import Sequential, Model
from keras.layers import Dense, Input
from tensorflow import set_random_seed
from numpy.random import seed

class Tier3LearningModel:

    def __init__(self):
        '''  Hybrid Dense model'''
        self.transientCNN_Tensor = np.array([])
        self.transientRNN_Tensor = np.array([]) 
        self.x_Train = np.array([])
        
    def populateArray(self, currArray,appendArray):
    
        if currArray.shape[0] == 0:
            currArray = appendArray
        else:
            currArray = np.insert(currArray, currArray.shape[0], appendArray, 0)
        return(currArray)
    
    def denseModelConfiguration(self):
        
        transientHybrid_Tensor = np.array([])
        if(self.transientCNN_Tensor.shape[0] == self.transientRNN_Tensor.shape[0]):
            print("tensor>>",self.transientCNN_Tensor.shape)
            transientHybrid_Tensor = np.concatenate((self.transientRNN_Tensor,self.transientCNN_Tensor), axis=2)
            
            '''
            print(" transient tensor>>",transientHybrid_Tensor.shape)
            modelInput = Input(shape = (transientHybrid_Tensor.shape[1],transientHybrid_Tensor.shape[2]))
            hiddenDenseLayer = Dense(1, activation='sigmoid')(modelInput)
            model = Model(inputs = modelInput, outputs = hiddenDenseLayer, name = 'denseLayer')
            
            transientFeedScores = model.predict(transientHybrid_Tensor)
            print("final Dense shape",transientFeedScores.shape)
            transientHybrid_Tensor = np.array([])
            transientHybrid_Tensor = self.populateArray(self, transientHybrid_Tensor, transientFeedScores)
            '''
            
        else:
            print("Tensor Size Mismatch Error")
            sys.exit()      
              
        print(" Hybrid Tensor Size :::",transientHybrid_Tensor.shape)   
                   
        return(transientHybrid_Tensor)
    
    
seed(1)
set_random_seed(2)    