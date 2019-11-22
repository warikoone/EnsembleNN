'''
Created on Jun 21, 2018

@author: neha
'''

import numpy as np
from keras.models import Model, Sequential
from keras.layers import Dense, LSTM, Input
from keras.optimizers import Adam
from keras.layers.embeddings import Embedding
from tensorflow import set_random_seed
from numpy.random import seed

class Tier1LearningModel:
    
    def __init__(self):
        '''  RNN model'''
        self.EMDEDDING_DIM = 0
        self.TWEET_SIZE = 0
        self.x_Train = np.array([])
    
    def populateArray(self, currArray,appendArray):
    
        if currArray.shape[0] == 0:
            currArray = appendArray
        else:
            currArray = np.insert(currArray, currArray.shape[0], appendArray, 0)
        return(currArray) 
        
    def rnnModelCofiguration(self):
        
        print("start model")
        modelInput = Input(shape = (self.TWEET_SIZE,self.EMDEDDING_DIM))
        
        '''
        hybridFeedDimension = int(np.rint(self.TWEET_SIZE/2))
        hiddenLayerLSTM, hiddenLayerLSTMState, hiddenLayerLSTMCell  = LSTM(hybridFeedDimension, return_sequences=True, return_state=True)(modelInput)
        hiddenLayerLSTM = Dense(1, activation='sigmoid')(hiddenLayerLSTM)
        model = Model(inputs = modelInput, outputs = [hiddenLayerLSTM, hiddenLayerLSTMState,  hiddenLayerLSTMCell], name = 'rnnLayer')
        '''
        
        
        sequentialLayeredLSTM = Sequential()
        hybridFeedDimension = self.EMDEDDING_DIM
        hiddenLayerLSTM = LSTM(hybridFeedDimension, return_sequences=True)(modelInput)
        model = Model(inputs = modelInput, outputs = hiddenLayerLSTM)
        sequentialLayeredLSTM.add(model)
        layers = 0
        while(hybridFeedDimension > 2):
            print(sequentialLayeredLSTM.predict(self.x_Train).shape)
            reduce = int(np.ceil(0.25*hybridFeedDimension))
            hybridFeedDimension = (hybridFeedDimension - reduce)
            layers += 1
            sequentialLayeredLSTM.add(LSTM(hybridFeedDimension,return_sequences= True))
            
        print("\n\t total number of layers>>",layers)
        sequentialLayeredLSTM.add(Dense(1, activation='sigmoid'))

        
        print("total RNN shape>>>",self.x_Train.shape)
        transientRNNTraining = np.array([])
        transientFeedScores = sequentialLayeredLSTM.predict(self.x_Train)
        print("final RNN shape",transientFeedScores.shape)
        transientRNNTraining = self.populateArray(self, transientRNNTraining, transientFeedScores.transpose(0,2,1))
        
        '''
        print("total Dense shape>>>",transientRNNTraining.shape)        
        modelInput = Input(shape = (transientRNNTraining.shape[1],transientRNNTraining.shape[2]))
        hiddenDenseLayer = Dense(1, activation='sigmoid')(modelInput)
        model = Model(inputs = modelInput, outputs = hiddenDenseLayer, name = 'denseLayer')
        
        transientFeedScores = model.predict(transientRNNTraining)
        print("final Dense shape",transientFeedScores.shape)
        transientRNNTraining = np.array([])
        transientRNNTraining = self.populateArray(self, transientRNNTraining, transientFeedScores)
        '''
        
        '''
        for tier1Index in range(4):
            #print(self.x_Train[tier1Index].shape)
            subTrainArray = np.array(self.x_Train[tier1Index])
            subTrainArray = subTrainArray.reshape(1, subTrainArray.shape[0], subTrainArray.shape[1])
            transientFeedScores = model.predict(subTrainArray)
            transientRNNTraining = self.populateArray(self, transientRNNTraining, transientFeedScores[0])
        '''
        
        print("final RNN shape",transientRNNTraining.shape)
                       
        '''
        lrAdam = Adam(lr=0.01,decay=0.001)
        model.compile(loss='binary_crossentropy', optimizer=lrAdam, metrics=['accuracy'])
        print(model.summary())
        
        model.fit(self.x_Train, self.y_Train, validation_data=(self.x_Train, self.y_Train), epochs=7, batch_size=int(self.x_Train.shape[0]/2))
        scores = model.evaluate(self.x_Train, self.y_Train, verbose=0)
        print("Accuracy: %.2f%%" % (scores[1]))
        
        for tier1Itr in range(self.x_Validation.shape[0]):
            testExample = self.x_Validation[tier1Itr].reshape(1,self.TWEET_SIZE,self.EMDEDDING_DIM)
            probability = model.predict(testExample, verbose=2)
            print(">>>",probability,"\t actual>>>",self.y_Validation[tier1Itr])
        
        '''
            
        return(transientRNNTraining)
    
seed(1)
set_random_seed(2)
