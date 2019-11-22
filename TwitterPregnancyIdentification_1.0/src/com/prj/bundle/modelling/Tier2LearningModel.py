'''
Created on Jun 22, 2018

@author: neha
'''

import numpy as np
from keras.models import Sequential, Model
from keras.layers import Dense, Input, Conv1D, MaxPool1D, Flatten, Concatenate, Add
from keras.optimizers import Adam
from keras.layers.embeddings import Embedding
from keras.activations import softmax
from tensorflow import set_random_seed
from numpy.random import seed

class Tier2LearningModel:
    
    def __init__(self):
        '''  CNN model'''
        self.EMDEDDING_DIM = 0
        self.TWEET_SIZE = 0
        self.x_Train = np.array([])
        
    def populateArray(self, currArray,appendArray):
    
        if currArray.shape[0] == 0:
            currArray = appendArray
        else:
            currArray = np.insert(currArray, currArray.shape[0], appendArray, 0)
        return(currArray) 
        
    def cnnModelCofiguration(self):
        
        print("start model")
        filterSize = [3,4,5]
        modelInput = Input(shape=(self.TWEET_SIZE, self.EMDEDDING_DIM))
        transientCNNTraining = np.array([])
        print("input CNN shape>>",self.x_Train.shape)
        
        '''
        convBlocks = []
        for eachFilter in filterSize:
            #hybridFeedDimension = int(np.rint(self.TWEET_SIZE/2))
            hybridFeedDimension = 256
            singleConv = Conv1D(filters=hybridFeedDimension, kernel_size=eachFilter, padding='valid',activation='relu',strides=2)(modelInput)
            singleConv = MaxPool1D(pool_size = 1)(singleConv)
            convBlocks.append(singleConv)
        
        
        concatenatedCNNLayer = Concatenate(axis=1)(convBlocks) if len(convBlocks) > 1 else convBlocks[0]
        concatenatedCNNLayer = Dense(1, activation='sigmoid')(concatenatedCNNLayer)
        model = Model(inputs = modelInput, outputs=concatenatedCNNLayer)
        transientFeedScores = model.predict(self.x_Train)
        print("CNN output shape>>>>", transientFeedScores.shape)
        transientCNNTraining = self.populateArray(self, transientCNNTraining, transientFeedScores.transpose(0,2,1))
        '''
        cnn_Threshold = int(self.EMDEDDING_DIM*0.50)
        convBlocks = []
        for eachFilter in filterSize:
            print("\n******************filter>>",eachFilter)
            tweetSpan = self.TWEET_SIZE
            modelInput = Input(shape=(self.TWEET_SIZE, self.EMDEDDING_DIM))
            hybridFeedDimension = self.EMDEDDING_DIM
            layers = 0
            filterSequential = Sequential()
            while((hybridFeedDimension > cnn_Threshold) and (eachFilter < tweetSpan)):
                #print("1>>>",hybridFeedDimension,"\t2>>>",self.EMDEDDING_DIM,"\t3>>>",tweetSpan)
                if hybridFeedDimension == self.EMDEDDING_DIM:
                    singleConv = Conv1D(filters=hybridFeedDimension, kernel_size=eachFilter, padding='valid',activation='relu',strides=1)(modelInput)
                    singleConv = MaxPool1D(pool_size = 1)(singleConv)
                    model = Model(inputs = modelInput, outputs = singleConv)
                    filterSequential.add(model)
                else:
                    filterSequential.add(Conv1D(filters=hybridFeedDimension, kernel_size=eachFilter, padding='valid',activation='relu',strides=1))
                    filterSequential.add(MaxPool1D(pool_size = 1))
                    
                print("\t>>",filterSequential.predict(self.x_Train).shape)
                tweetSpan = filterSequential.predict(self.x_Train).shape[1]
                stepReduce = int(np.ceil(0.25*hybridFeedDimension))
                hybridFeedDimension = (hybridFeedDimension - stepReduce)
                print(">>>",stepReduce,"CUrr>>",hybridFeedDimension)
                layers +=1
            
            print("layers>>>",layers)   
            filterSequential.add(Dense(1, activation='sigmoid'))
            print("\t dense>>",filterSequential.predict(self.x_Train).shape)
            convBlocks.append(filterSequential.predict(self.x_Train))
            
        for tweetIndex in range(self.x_Train.shape[0]):
            tier1Array = np.array([])
            for filterIndex in range(len(convBlocks)):
                tier2Array = np.array(convBlocks[filterIndex])
                tier1Array = self.populateArray(self, tier1Array, tier2Array[tweetIndex])
            tier1Array = tier1Array.reshape(1,tier1Array.shape[0],tier1Array.shape[1])
            tier1Array = tier1Array.transpose(0,2,1)
            #print("subArray>>",tier1Array.shape)
            transientCNNTraining = self.populateArray(self, transientCNNTraining, tier1Array) 
        
        '''
        print("total Dense shape>>>",transientCNNTraining.shape)        
        modelInput = Input(shape = (transientCNNTraining.shape[1],transientCNNTraining.shape[2]))
        hiddenDenseLayer = Dense(1, activation='sigmoid')(modelInput)
        model = Model(inputs = modelInput, outputs = hiddenDenseLayer, name = 'denseLayer')
        
        transientFeedScores = model.predict(transientCNNTraining)
        print("final Dense shape",transientFeedScores.shape)
        transientCNNTraining = np.array([])
        transientCNNTraining = self.populateArray(self, transientCNNTraining, transientFeedScores)
        '''
        
        '''
        for tier1Index in range(4):
            subTrainArray = np.array(self.x_Train[tier1Index])
            subTrainArray = subTrainArray.reshape(1, subTrainArray.shape[0], subTrainArray.shape[1])
            convBlocks = np.array([])
            for eachFilter in filterSize:
                hybridFeedDimension = int(np.rint(self.TWEET_SIZE/2))
                singleConv = Conv1D(filters=hybridFeedDimension, kernel_size=eachFilter, padding='valid',activation='relu',strides=300)(modelInput)
                singleConv = MaxPool1D(pool_size = 1)(singleConv)
                model = Model(inputs = modelInput, outputs=singleConv)
                
                transientFeedScores = model.predict(subTrainArray)
                transientFeedScores = transientFeedScores[0]
                #print("window %d" % eachFilter,"\t>>>",transientFeedScores.shape)
                convBlocks = self.populateArray(self, convBlocks, transientFeedScores)
            
            convBlocks = convBlocks.reshape(1, convBlocks.shape[0], convBlocks.shape[1])
            #print("index %d" % tier1Index,"\t>>>",convBlocks.shape)
            transientCNNTraining = self.populateArray(self, transientCNNTraining, convBlocks)
        '''
           
        print("CNN output shape>>>>", transientCNNTraining.shape)
        
        '''
        lrAdam = Adam(lr=0.01,decay=0.001)
        model.compile(loss='binary_crossentropy', optimizer=lrAdam, metrics=['accuracy'])
        print(model.summary())
        
        model.fit(self.x_Train, self.y_Train, validation_data=(self.x_Train, self.y_Train), epochs=5, batch_size=int(self.x_Train.shape[0]/2))
        scores = model.evaluate(self.x_Train, self.y_Train, verbose=0)
        print("Accuracy: %.2f%%" % (scores[1]))
        
        for tier1Itr in range(self.x_Validation.shape[0]):
            testExample = self.x_Validation[tier1Itr].reshape(1,self.TWEET_SIZE,self.EMDEDDING_DIM)
            probability = model.predict(testExample, verbose=2)
            print(">>>",probability,"\t actual>>>",self.y_Validation[tier1Itr])
        '''
        
        return(transientCNNTraining)
    
seed(1)
set_random_seed(2)    