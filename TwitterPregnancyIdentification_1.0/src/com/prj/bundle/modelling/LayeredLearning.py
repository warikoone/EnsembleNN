'''
Created on Jun 17, 2018

@author: neha
'''
from src.com.prj.bundle.processing import EmbeddingRepresentation
from src.com.prj.bundle.modelling import Tier1LearningModel
from src.com.prj.bundle.modelling import Tier2LearningModel
from src.com.prj.bundle.modelling import Tier3LearningModel
import numpy as np
from sklearn.model_selection import KFold
from keras.utils import to_categorical
from sklearn import svm
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import classification_report, precision_score, recall_score, f1_score
from collections import Counter
from operator import itemgetter
import sys
from tensorflow import set_random_seed
from numpy.random import seed

class LayeredLearning:
    
    def __init__(self):
        print("initializing Layered Learning")
        self.resourceData = {}
        self.multiClassLabelResource = {}
        self.TWEET_EMBEDDING = {}
        self.kFoldSplitSize = 10
        self.EMDEDDING_DIM = 0
        self.TWEET_SIZE = 0
        self.category_index = {-1:0, 1:1}
        self.multicategory_index = {'N':2,'H':3,'A':4,'D':5}
        self.transientRNN_Training = np.array([])
        self.transientCNN_Training = np.array([])
        self.HYBRIDEMBEDDING_DICTIONARY = {}
        self.CLASSIFICATION_DICTIONARY = {}
        self.defaultLabel = 1
        self.MULTICLASSCLASSIFICATION_DICTIONARY = {}
        
    def loadEmbeddingRepresentation(self):
        embeddingInstance = EmbeddingRepresentation.EmbeddingRepresentation
        embeddingInstance.__init__(embeddingInstance)
        embeddingInstance.loadResource(embeddingInstance) 
        self.resourceData = embeddingInstance.resourceData
        self.TWEET_EMBEDDING = embeddingInstance.TWEET_EMBEDDING
        self.EMDEDDING_DIM = embeddingInstance.EMBEDDING_DIM
        self.TWEET_SIZE = embeddingInstance.tweetSize
        embeddingInstance.readMultiClassLabel(embeddingInstance)
        self.multiClassLabelResource = embeddingInstance.multiClassLabelResource

    def dataStratification(self):
        
        trainPassDictionary = {}
        validationPassDictionary = {}
        for currentKey in self.resourceData.keys():
            print("currentkey>>",currentKey)
            decoyDictionary = self.resourceData.get(currentKey)
            kFoldSplitStrata = KFold(n_splits = self.kFoldSplitSize)
            decoyArray = np.array(list(map(lambda valueItem : int(valueItem), decoyDictionary.keys())))
            passIndex = 0
            for trainIndex, validIndex in kFoldSplitStrata.split(decoyArray):
                #print("passIndex>>",passIndex,"\n train>>",trainIndex,"\t test>>",validIndex)
                decoyDictionary = {}
                if passIndex in trainPassDictionary:
                    #print("dict>>",trainPassDictionary)
                    decoyDictionary = trainPassDictionary.get(passIndex)
                    #print("decoy>>>",decoyDictionary)
                decoyDictionary.update({currentKey:trainIndex})
                trainPassDictionary.update({passIndex:decoyDictionary})
                decoyDictionary = {}
                if passIndex in validationPassDictionary:
                    decoyDictionary = dict(validationPassDictionary.get(passIndex))
                decoyDictionary.update({currentKey:validIndex})
                validationPassDictionary.update({passIndex:decoyDictionary})
                passIndex += 1

        #print("pass dict>>>",trainPassDictionary.get(0))                
        return(trainPassDictionary,validationPassDictionary)
    
    def normalizeDataSize(self, trainPassDictionary):
        
        for strataIndex in trainPassDictionary:
            bufferDictionary = dict(trainPassDictionary.get(strataIndex))
            balancedResourceSize = self.minResourceSize(bufferDictionary)
            print("strat::",strataIndex)
            for tier1MapKey in bufferDictionary:
                tier1MapValue = list(bufferDictionary.get(tier1MapKey))
                tier1MapValue = tier1MapValue[0:balancedResourceSize]
                bufferDictionary.update({tier1MapKey:tier1MapValue})
                print("instance::",tier1MapKey,"\t len>>",len(tier1MapValue))
            trainPassDictionary.update({strataIndex:bufferDictionary})
                
        return(trainPassDictionary)
    
    def populateArray(self, currArray,appendArray):
    
        if currArray.shape[0] == 0:
            currArray = appendArray
        else:
            currArray = np.insert(currArray, currArray.shape[0], appendArray, 0)
        return(currArray)  
    
    def mappingTensor(self, decoyDictionary, instanceType):
        
        '''
        bufferEmbeddingMatrix = np.zeros((1,self.EMDEDDING_DIM))
        bufferOutputMatrix = np.zeros((1,1))
        '''
        bufferEmbeddingMatrix = np.array([])
        bufferOutputMatrix = np.array([])
        blockTensorSize = 0
        for tier1Itr in decoyDictionary.items():
            tier1MapValue = tier1Itr[1]
            tweetEmbedding = np.array(tier1MapValue)
            #print("dict embed>>",tweetEmbedding.shape)
            tweetEmbedding = tweetEmbedding.reshape(self.TWEET_SIZE,self.EMDEDDING_DIM)
            #print("reshape embed>>",tweetEmbedding.shape)
            bufferEmbeddingMatrix = self.populateArray(bufferEmbeddingMatrix, tweetEmbedding)
            #statusList = np.array(to_categorical(self.category_index[instanceType], len(self.category_index))).reshape(1,2)
            statusList = np.array(self.category_index[instanceType]).reshape(1,1)
            bufferOutputMatrix = self.populateArray(bufferOutputMatrix, statusList)
            blockTensorSize += 1
        
        blockTensorDimension = blockTensorSize
        spanTensorDimension = self.TWEET_SIZE
        featureTensorDimension = self.EMDEDDING_DIM 
        bufferEmbeddingMatrix = bufferEmbeddingMatrix.reshape(blockTensorDimension, spanTensorDimension, featureTensorDimension)        
        return(bufferEmbeddingMatrix, bufferOutputMatrix)
        
    def lauchStrataLearning(self):

        ''' RNN '''
        tier1Instance = Tier1LearningModel.Tier1LearningModel
        tier1Instance.TWEET_SIZE = self.TWEET_SIZE
        tier1Instance.EMDEDDING_DIM = self.EMDEDDING_DIM
        
        ''' CNN '''
        tier2Instance = Tier2LearningModel.Tier2LearningModel
        tier2Instance.TWEET_SIZE = self.TWEET_SIZE
        tier2Instance.EMDEDDING_DIM = self.EMDEDDING_DIM
        
        tier3Instance = Tier3LearningModel.Tier3LearningModel
        
        for instanceType in self.TWEET_EMBEDDING:
            print("instancetype>>>",instanceType)
            subEmbeddingDictionary = dict(self.TWEET_EMBEDDING.get(instanceType))
            x_TrainMatrix, y_TrainMatrix = self.mappingTensor(subEmbeddingDictionary, instanceType)
            tier1Instance.x_Train = x_TrainMatrix
            tier2Instance.x_Train = x_TrainMatrix
            tier3Instance.transientRNN_Tensor = tier1Instance.rnnModelCofiguration(tier1Instance)
            tier3Instance.transientCNN_Tensor = tier2Instance.cnnModelCofiguration(tier2Instance)
            transientHybrid_Tensor = tier3Instance.denseModelConfiguration(tier3Instance)
            self.HYBRIDEMBEDDING_DICTIONARY.update({instanceType:transientHybrid_Tensor})
            self.CLASSIFICATION_DICTIONARY.update({instanceType:y_TrainMatrix})
        
        return()
    
    def minResourceSize(self, bufferDictionary):
        
        balancedResourceSize = 100000000
        for tier1MapKey in bufferDictionary:
            currentResourceSize = len(bufferDictionary.get(tier1MapKey))
            if currentResourceSize < balancedResourceSize:
                balancedResourceSize = currentResourceSize
        return(balancedResourceSize)
    
    def getReducedTensorRepresentation(self, bufferDictionary, dataType):

        indexArrayMap = []
        balancedResourceDictionary = {}
        maxSplitSize = 0
        balancedResourceSize = self.minResourceSize(bufferDictionary)
        a=balancedResourceSize
        #balancedResourceSize = balancedResourceSize + int(balancedResourceSize/25)
        print("\n percentage",(balancedResourceSize/(balancedResourceSize+a)))
        for tier1MapKey in bufferDictionary:
            indexArrays = []
            tier1MapValue = list(bufferDictionary.get(tier1MapKey))
            if dataType == 'training':
                while len(tier1MapValue) > balancedResourceSize:
                    #subIndexArray = list(np.random.choice(tier1MapValue, balancedResourceSize, replace=False))
                    #subIndexArray.sort()
                    subIndexArray = list(tier1MapValue[0:balancedResourceSize])
                    halfSize = int(len(subIndexArray)*0.3)
                    for item in subIndexArray[0:halfSize]:
                        tier1MapValue.remove(item)
                    indexArrays.append(subIndexArray)
                if len(tier1MapValue) < balancedResourceSize:
                    tier1MapValue.extend(tier1MapValue)
                indexArrays.append(tier1MapValue[0:balancedResourceSize])
            else:
                indexArrays.append(tier1MapValue)
                
            #indexArrays.append(tier1MapValue)

            if len(indexArrays) > maxSplitSize:
                maxSplitSize = len(indexArrays)
            balancedResourceDictionary.update({tier1MapKey:indexArrays})
            print("instance>>>",tier1MapKey)
            for i in range(len(indexArrays)):
                print("size>>",i,">>>",len(indexArrays[i]))
    
        iterIndex = 0
        x_TrainArray = {}
        y_TrainArray = {}
        while(iterIndex < maxSplitSize):
            x_train = np.array([])
            y_train = np.array([])
            for tier2MapKey in balancedResourceDictionary:
                tier2MapValue = balancedResourceDictionary.get(tier2MapKey)
                if len(tier2MapValue) <= iterIndex:
                    tier3DecoyArray = tier2MapValue[0]
                else:
                    tier3DecoyArray = tier2MapValue[iterIndex]
                
                for termIndex in tier3DecoyArray:
                    x_train = self.populateArray(x_train, self.HYBRIDEMBEDDING_DICTIONARY.get(tier2MapKey)[termIndex])
                    y_train = self.populateArray(y_train, self.CLASSIFICATION_DICTIONARY.get(tier2MapKey)[termIndex])
                    if x_train.shape[0] != 0:
                        decoyIndexArray = {}
                        decoyIndexArray.update({termIndex:tier2MapKey})
                        indexArrayMap.append(decoyIndexArray)
                    

            x_TrainArray.update({iterIndex:x_train})
            y_TrainArray.update({iterIndex:y_train})
            iterIndex += 1
            
        print("x subType>>>",x_TrainArray.keys())
        return(x_TrainArray, y_TrainArray, iterIndex, indexArrayMap)
    
    def lauchSupportVectorClassifier(self, trainPassDictionary, validationPassDictionary):
        
        predictedPositiveInstances = {}
        overAllFoldMetrics = {}
        overPred = []
        overGround = []
        indexValidArray = []
        for strataIndex in range(self.kFoldSplitSize):
            print("\n\t****Passs %d" % strataIndex,"***********")
            bufferIndexDictionary = dict(trainPassDictionary.get(strataIndex))
            validationIndexDictionary = dict(validationPassDictionary.get(strataIndex))
            x_TrainArray, y_TrainArray, iterTrainingIndex, indexTrainMap  = self.getReducedTensorRepresentation(bufferIndexDictionary,'training')
            x_ValidationArray, y_ValidationArray, iterValidationIndex, indexValidMap = self.getReducedTensorRepresentation(validationIndexDictionary, 'validation')
            print("x_validation>>",x_ValidationArray.keys(),"\t>>>",x_ValidationArray.get(0).shape)
            currentRunMetrics = {}
            subPred = {}
            subGround = []
            for iterIndexId in range(iterTrainingIndex):
                print("\n\t****Sub Passs %d" % iterIndexId,"***********")
                X_Train = x_TrainArray.get(iterIndexId)
                Y_Train = y_TrainArray.get(iterIndexId)
                X_Valid = x_ValidationArray.get(0)
                y_Valid = y_ValidationArray.get(0)
                
                ''' SVM '''
                svmKernel = svm.SVC(C=1.0, kernel='rbf',coef0=0.0 , degree=1, gamma=0.001)
                svmKernel.fit(X_Train, Y_Train)
                
                Y_Predict = svmKernel.predict(X_Valid)
                
                ''' Naive Bayes'''
                '''
                gnb = GaussianNB()
                gnb.fit(X_Train, Y_Train)
                y_Valid = y_ValidationArray.get(0)
                Y_Predict = gnb.predict(X_Valid)
                '''
                
                print(" valid Size",len(y_Valid),"\n",list(map(lambda x : x, y_Valid)))
                print(" pred Size",len(Y_Predict),"\n",list(map(lambda x : x, Y_Predict)))
                
                print((set(y_Valid)-set(Y_Predict )))

                for index in range(len(Y_Predict)):
                    bufferArray = []
                    if index in subPred:
                        bufferArray = subPred.get(index)
                    bufferArray.append(Y_Predict[index])
                    subPred.update({index:bufferArray})
                subGround.append(y_Valid)
                print(classification_report(y_Valid, Y_Predict))
                
                precisionScore = [0.0, 0.0]
                recallScore = [0.0, 0.0]
                f1Score = [0.0, 0.0]
                if 'Precision' in currentRunMetrics:
                    precisionScore = currentRunMetrics.get('Precision')
                bufferArray = precision_score(y_Valid, Y_Predict, average=None)
                print("precision>>",bufferArray)
                precisionScore = list(map(lambda value,index : precisionScore[index]+value, bufferArray, range(len(bufferArray))))
                if iterIndexId == (iterTrainingIndex-1):
                    precisionScore = list(map(lambda value : value/iterTrainingIndex,precisionScore))
                currentRunMetrics.update({'Precision':precisionScore})
                
                if 'Recall' in currentRunMetrics:
                    recallScore = currentRunMetrics.get('Recall')
                bufferArray = recall_score(y_Valid, Y_Predict, average=None)
                print("Recall>>",bufferArray)
                recallScore = list(map(lambda value,index : recallScore[index]+value, bufferArray, range(len(bufferArray))))
                if iterIndexId == (iterTrainingIndex-1):
                    recallScore = list(map(lambda value : value/iterTrainingIndex,recallScore))
                currentRunMetrics.update({'Recall':recallScore})
                
                if 'f1' in currentRunMetrics:
                    f1Score = currentRunMetrics.get('f1')
                bufferArray = f1_score(y_Valid, Y_Predict, average=None)
                print("f1>>",bufferArray)
                f1Score = list(map(lambda value,index : f1Score[index]+value, bufferArray, range(len(bufferArray))))
                if iterIndexId == (iterTrainingIndex-1):
                    f1Score = list(map(lambda value : value/iterTrainingIndex,f1Score))
                currentRunMetrics.update({'f1':f1Score})
                
                #sys.exit()
            bufferArray = []
            for index in subPred:
                decoyVoteDictionary = dict(Counter(subPred.get(index)))
                status = int(max(decoyVoteDictionary.items(),key=itemgetter(1))[0])
                bufferArray.append(status)
            
            indexValidArray.extend(indexValidMap)
            overPred.extend(bufferArray)
            overGround.extend(subGround[0])
            
            precisionScore = [0.0, 0.0]
            recallScore = [0.0, 0.0]
            f1Score = [0.0, 0.0]
            if 'Precision' in overAllFoldMetrics:
                precisionScore = overAllFoldMetrics.get('Precision')
            precisionScore = list(map(lambda value,index : precisionScore[index]+value, currentRunMetrics.get('Precision'), range(len(currentRunMetrics.get('Precision')))))
            if strataIndex == (self.kFoldSplitSize-1):
                precisionScore = list(map(lambda value : value/self.kFoldSplitSize,precisionScore))
            overAllFoldMetrics.update({'Precision':precisionScore})
            
            if 'Recall' in overAllFoldMetrics:
                recallScore = overAllFoldMetrics.get('Recall')
            recallScore = list(map(lambda value,index : recallScore[index]+value, currentRunMetrics.get('Recall'), range(len(currentRunMetrics.get('Recall')))))
            if strataIndex == (self.kFoldSplitSize-1):
                recallScore = list(map(lambda value : value/self.kFoldSplitSize,recallScore))
            overAllFoldMetrics.update({'Recall':recallScore})
                
            if 'f1' in overAllFoldMetrics:
                f1Score = overAllFoldMetrics.get('f1')
            f1Score = list(map(lambda value,index : f1Score[index]+value, currentRunMetrics.get('f1'), range(len(currentRunMetrics.get('f1')))))
            if strataIndex == (self.kFoldSplitSize-1):
                f1Score = list(map(lambda value : value/self.kFoldSplitSize,f1Score))
            overAllFoldMetrics.update({'f1':f1Score})
            

        print("precision>>",overAllFoldMetrics.get('Precision'))
        print("Recall>>",overAllFoldMetrics.get('Recall'))
        print("f1>>",overAllFoldMetrics.get('f1'))        
        
        print("total size",len(overPred),"\t>>>",len(overGround))
        print("\n***********",classification_report(overGround, overPred))
        
        
        print("\n>>>",overPred)
        for index, value in enumerate(overPred):
            if (value == 1):
                tempItem = dict(indexValidArray[index]).popitem()
                bufferArray = []
                if tempItem[1] in predictedPositiveInstances:
                    bufferArray = predictedPositiveInstances.get(tempItem[1])
                bufferArray.append(tempItem[0])
                predictedPositiveInstances.update({tempItem[1]:bufferArray}) 
                print("\t>>",index,"::",tempItem)
                            
        return(predictedPositiveInstances)
    
    def generateMultiClassStrata(self, predictedPositiveInstances):
        
        trainPassDictionary = {}
        validationPassDictionary = {}
        for keyLabel in self.multicategory_index.keys():
            decoyArray = self.multiClassLabelResource.get(keyLabel)
            kFoldSplitStrata = KFold(n_splits = self.kFoldSplitSize)
            passIndex = 0
            for trainIndex, validIndex in kFoldSplitStrata.split(decoyArray):
                decoyDictionary = {}
                if passIndex in trainPassDictionary:
                    decoyDictionary = trainPassDictionary.get(passIndex)
                decoyList = list(map(lambda index : decoyArray[index], trainIndex))
                decoyDictionary.update({keyLabel:decoyList})
                trainPassDictionary.update({passIndex:decoyDictionary})
                
                decoyDictionary = {}
                if passIndex in validationPassDictionary:
                    decoyDictionary = dict(validationPassDictionary.get(passIndex))
                decoyList = list(map(lambda index : decoyArray[index], validIndex))
                decoyList = list(set(predictedPositiveInstances.get(self.defaultLabel)).intersection(set(decoyList)))
                decoyDictionary.update({keyLabel:decoyList})
                validationPassDictionary.update({passIndex:decoyDictionary})
                
                passIndex += 1
                
        for keyInstance in trainPassDictionary:
            print("\t pass %d" % keyInstance,"\t:::",trainPassDictionary.get(keyInstance))
            
        return(trainPassDictionary, validationPassDictionary)
    
    def getReducedTensorForMultiClass(self, bufferDictionary, dataType):
        
        balancedResourceDictionary = {}
        maxSplitSize = 0
        balancedResourceSize = self.minResourceSize(bufferDictionary)
        a=balancedResourceSize
        #balancedResourceSize = balancedResourceSize + int(balancedResourceSize/25)
        print("\n percentage",(balancedResourceSize/(balancedResourceSize+a)),"\t dataType>>",dataType)
        for tier1MapKey in bufferDictionary:
            indexArrays = []
            tier1MapValue = list(bufferDictionary.get(tier1MapKey))
            if dataType == 'training':
                while len(tier1MapValue) > balancedResourceSize:
                    #subIndexArray = list(np.random.choice(tier1MapValue, balancedResourceSize, replace=False))
                    #subIndexArray.sort()
                    subIndexArray = list(tier1MapValue[0:balancedResourceSize])
                    halfSize = int(len(subIndexArray)*0.3)
                    for item in subIndexArray[0:halfSize]:
                        tier1MapValue.remove(item)
                    indexArrays.append(subIndexArray)
                if len(tier1MapValue) < balancedResourceSize:
                    tier1MapValue.extend(tier1MapValue)
                indexArrays.append(tier1MapValue[0:balancedResourceSize])
            else:
                indexArrays.append(tier1MapValue)
                
            #indexArrays.append(tier1MapValue)

            if len(indexArrays) > maxSplitSize:
                maxSplitSize = len(indexArrays)
            balancedResourceDictionary.update({tier1MapKey:indexArrays})
            print("instance>>>",tier1MapKey)
            for i in range(len(indexArrays)):
                print("size>>",i,">>>",len(indexArrays[i]))
    
        iterIndex = 0
        x_TrainArray = {}
        y_TrainArray = {}
        while(iterIndex < maxSplitSize):
            x_train = np.array([])
            y_train = np.array([])
            for tier2MapKey in balancedResourceDictionary:
                tier2MapValue = balancedResourceDictionary.get(tier2MapKey)
                if len(tier2MapValue) <= iterIndex:
                    tier3DecoyArray = tier2MapValue[0]
                else:
                    tier3DecoyArray = tier2MapValue[iterIndex]
                
                for termIndex in tier3DecoyArray:
                    x_train = self.populateArray(x_train, self.HYBRIDEMBEDDING_DICTIONARY.get(self.defaultLabel)[termIndex])
                    if tier2MapKey is not self.defaultLabel:
                        statusList = np.array([tier2MapKey])
                    else:
                        statusList = self.CLASSIFICATION_DICTIONARY.get(tier2MapKey)[termIndex]
                        
                    #print("statuslist>>>",statusList.shape)
                    y_train = self.populateArray(y_train, statusList)
                
            y_train = y_train.reshape(y_train.shape[0])
            x_TrainArray.update({iterIndex:x_train})
            y_TrainArray.update({iterIndex:y_train})
            iterIndex += 1
            
        print("x subType>>>",x_TrainArray.keys())
        return(x_TrainArray, y_TrainArray, iterIndex)
    
    def reorganizeTensors(self, keyLabel, decoyDictionary):
        
        bufferDictionary = {}
        negativeInstances = []
        for label in decoyDictionary:
            if label is not keyLabel:
                negativeInstances.extend(decoyDictionary.get(label))
            else:
                bufferDictionary.update({self.multicategory_index.get(keyLabel):decoyDictionary.get(label)}) 
        bufferDictionary.update({self.defaultLabel:negativeInstances})
        return(bufferDictionary)
    
    def launchOneVsAllSupportVectorClassifier(self, trainMultiClassPassDictionary, validationMultiClassPassDictionary):
        
        
        for keyLabel in ['D']:#self.multicategory_index.keys():
            print("\n\t<<<<<<<<<<<<<<Primary Key>>>>>>>>>>>>>>",keyLabel)
            overAllFoldMetrics = {}
            overPred = []
            overGround = []
            for strataIndex in range(self.kFoldSplitSize):
                print("\n\t****Passs %d" % strataIndex,"***********")
                decoyTrainDictionary = dict(trainMultiClassPassDictionary.get(strataIndex))
                decoyValidDictionary = dict(validationMultiClassPassDictionary.get(strataIndex))
                decoyTrainDictionary = self.reorganizeTensors(keyLabel, decoyTrainDictionary)
                decoyValidDictionary = self.reorganizeTensors(keyLabel, decoyValidDictionary)
                
                x_TrainArray, y_TrainArray, iterTrainingIndex = self.getReducedTensorForMultiClass(decoyTrainDictionary, 'training')
                x_ValidationArray, y_ValidationArray, iterValidationIndex = self.getReducedTensorForMultiClass(decoyValidDictionary, 'validation')
                print("\n x_train>>",x_TrainArray.keys(),"\t>>>",x_TrainArray.get(0).shape)
                print("\n y_train>>",y_TrainArray.keys(),"\t>>>",y_TrainArray.get(0).shape)
                print("x_validation>>",x_ValidationArray.keys(),"\t>>>",x_ValidationArray.get(0).shape)
                print("y_validation>>",y_ValidationArray.keys(),"\t>>>",y_ValidationArray.get(0).shape)
                currentRunMetrics = {}
                subPred = {}
                subGround = []
                for iterIndexId in range(iterTrainingIndex):
                    print("\n\t****Sub Passs %d" % iterIndexId,"***********")
                    X_Train = x_TrainArray.get(iterIndexId)
                    Y_Train = y_TrainArray.get(iterIndexId)
                    svmKernel = svm.SVC(C=1.0, kernel='rbf',coef0=0.0 , degree=1, gamma=0.001)
                    svmKernel.fit(X_Train, Y_Train)
                    X_Valid = x_ValidationArray.get(0)
                    y_Valid = y_ValidationArray.get(0)
                    Y_Predict = svmKernel.predict(X_Valid)
                    
                    print(" valid Size",len(y_Valid),"\n",list(map(lambda x : x, y_Valid)))
                    print(" pred Size",len(Y_Predict),"\n",list(map(lambda x : x, Y_Predict)))
                    
                    print((set(y_Valid)-set(Y_Predict)))
    
                    for index in range(len(Y_Predict)):
                        bufferArray = []
                        if index in subPred:
                            bufferArray = subPred.get(index)
                        bufferArray.append(Y_Predict[index])
                        subPred.update({index:bufferArray})
                    subGround.append(y_Valid)
                    print(classification_report(y_Valid, Y_Predict))
                    
                    precisionScore = [0.0, 0.0]
                    recallScore = [0.0, 0.0]
                    f1Score = [0.0, 0.0]
                    if 'Precision' in currentRunMetrics:
                        precisionScore = currentRunMetrics.get('Precision')
                    bufferArray = precision_score(y_Valid, Y_Predict, average=None)
                    print("precision>>",bufferArray)
                    precisionScore = list(map(lambda value,index : precisionScore[index]+value, bufferArray, range(len(bufferArray))))
                    if iterIndexId == (iterTrainingIndex-1):
                        precisionScore = list(map(lambda value : value/iterTrainingIndex,precisionScore))
                    currentRunMetrics.update({'Precision':precisionScore})
                    
                    if 'Recall' in currentRunMetrics:
                        recallScore = currentRunMetrics.get('Recall')
                    bufferArray = recall_score(y_Valid, Y_Predict, average=None)
                    print("Recall>>",bufferArray)
                    recallScore = list(map(lambda value,index : recallScore[index]+value, bufferArray, range(len(bufferArray))))
                    if iterIndexId == (iterTrainingIndex-1):
                        recallScore = list(map(lambda value : value/iterTrainingIndex,recallScore))
                    currentRunMetrics.update({'Recall':recallScore})
                    
                    if 'f1' in currentRunMetrics:
                        f1Score = currentRunMetrics.get('f1')
                    bufferArray = f1_score(y_Valid, Y_Predict, average=None)
                    print("f1>>",bufferArray)
                    f1Score = list(map(lambda value,index : f1Score[index]+value, bufferArray, range(len(bufferArray))))
                    if iterIndexId == (iterTrainingIndex-1):
                        f1Score = list(map(lambda value : value/iterTrainingIndex,f1Score))
                    currentRunMetrics.update({'f1':f1Score})
                
                bufferArray = []
                for index in subPred:
                    decoyVoteDictionary = dict(Counter(subPred.get(index)))
                    status = int(max(decoyVoteDictionary.items(),key=itemgetter(1))[0])
                    bufferArray.append(status)
            
                overPred.extend(bufferArray)
                overGround.extend(subGround[0])
                
                precisionScore = [0.0, 0.0]
                recallScore = [0.0, 0.0]
                f1Score = [0.0, 0.0]
                if 'Precision' in overAllFoldMetrics:
                    precisionScore = overAllFoldMetrics.get('Precision')
                precisionScore = list(map(lambda value,index : precisionScore[index]+value, currentRunMetrics.get('Precision'), range(len(currentRunMetrics.get('Precision')))))
                if strataIndex == (self.kFoldSplitSize-1):
                    precisionScore = list(map(lambda value : value/self.kFoldSplitSize,precisionScore))
                overAllFoldMetrics.update({'Precision':precisionScore})
                
                if 'Recall' in overAllFoldMetrics:
                    recallScore = overAllFoldMetrics.get('Recall')
                recallScore = list(map(lambda value,index : recallScore[index]+value, currentRunMetrics.get('Recall'), range(len(currentRunMetrics.get('Recall')))))
                if strataIndex == (self.kFoldSplitSize-1):
                    recallScore = list(map(lambda value : value/self.kFoldSplitSize,recallScore))
                overAllFoldMetrics.update({'Recall':recallScore})
                    
                if 'f1' in overAllFoldMetrics:
                    f1Score = overAllFoldMetrics.get('f1')
                f1Score = list(map(lambda value,index : f1Score[index]+value, currentRunMetrics.get('f1'), range(len(currentRunMetrics.get('f1')))))
                if strataIndex == (self.kFoldSplitSize-1):
                    f1Score = list(map(lambda value : value/self.kFoldSplitSize,f1Score))
                overAllFoldMetrics.update({'f1':f1Score})
                
            print("precision>>",overAllFoldMetrics.get('Precision'))
            print("Recall>>",overAllFoldMetrics.get('Recall'))
            print("f1>>",overAllFoldMetrics.get('f1'))        
            
            print("total size",len(overPred),"\t>>>",len(overGround))
            print("\n***********",classification_report(overGround, overPred))
            
            #break
        return()

seed(1)
set_random_seed(2)
learningInstance = LayeredLearning()
learningInstance.loadEmbeddingRepresentation()
''' Binary Classification '''
trainPassDictionary,validationPassDictionary = learningInstance.dataStratification()
learningInstance.lauchStrataLearning()
predictedPositiveInstances = learningInstance.lauchSupportVectorClassifier(trainPassDictionary,validationPassDictionary)
''' MultiClass Classification '''
#trainMultiClassPassDictionary, validationMultiClassPassDictionary = learningInstance.generateMultiClassStrata(predictedPositiveInstances)
#learningInstance.launchOneVsAllSupportVectorClassifier(trainMultiClassPassDictionary, validationMultiClassPassDictionary)


