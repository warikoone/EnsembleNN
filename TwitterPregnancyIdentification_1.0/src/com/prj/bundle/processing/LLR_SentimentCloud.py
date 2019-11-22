'''
Created on Jul 17, 2018

@author: iasl
'''
import sys
import re
import numpy as np
from collections import Counter
from src.com.prj.bundle.processing import EmbeddingRepresentation

class LLR_SentimentCloud:
    
    def __init__(self):
        self.multiCLassLabelInput = {}
        self.wordCloudDictionary = {}
        self.wordShade = {'D':1,'A':4,'H':5,'N':7}
        self.vocab = []
        
    def getVocabToken(self, embeddingInstance, sentenceList):
        
        vocabArray = []
        for sentence in sentenceList:
            sentence = embeddingInstance.removeStopWords(embeddingInstance, sentence)
            #print(sentence)
            vocabArray.extend(re.split('\s+', sentence))
        return(vocabArray)
    
    def removeDuplications(self, bufferList):
        
        decoyList=[]
        for listItem in bufferList:
            decoyList.extend(listItem)
        print("preliminary size %d" %len(decoyList))
        decoyDictionary = Counter(decoyList)
        print("after size %d" % len(decoyDictionary))
        return(decoyDictionary)
    
    def populateArray(self, currArray,appendArray):
    
        if currArray.shape[0] == 0:
            currArray = appendArray
        else:
            currArray = np.insert(currArray, currArray.shape[0], appendArray, 0)
        return(currArray) 
    
    def calculateLLR(self, confusionMatrix):
        
        llrModel = np.array([])
        for i in range(confusionMatrix.shape[0]):
            decoyArray = []
            for j in range(confusionMatrix.shape[1]):
                num = confusionMatrix[i][j]/confusionMatrix[i,:].sum()
                #print("num>>>",num)
                denom = sum(confusionMatrix[:,j])/confusionMatrix.sum()
                #print("denom>>>",denom)
                hypothesisProb = (num/denom)
                #print(">>>>",hypothesisProb)
                quadrantScore = 0
                if int(np.ceil(hypothesisProb)) != 0:
                    quadrantScore = (confusionMatrix[i][j] * np.log(hypothesisProb))
                #print("quadrantScore>>>>",quadrantScore)
                decoyArray.append(quadrantScore)
                
            llrModel = self.populateArray(llrModel, np.array([decoyArray]))
        
        #print(llrModel)
        llrScore = (llrModel.sum()*2)
        #print(llrScore)
        #sys.exit()
        return(llrScore)
    
    def generateConfusionMatrix(self, sentimentToken, sentimentComplimentDictionary, sentimentDictionary, colorIndex):
        
        tp=0
        if (sentimentToken in sentimentDictionary):
            tp = sentimentDictionary.get(sentimentToken)
            
        fp=0
        if (sentimentToken in sentimentComplimentDictionary):
            fp = sentimentComplimentDictionary.get(sentimentToken)
            
        fn=0
        tnList = (set(sentimentComplimentDictionary.keys())-set(sentimentToken))
        fnList = (tnList &(set(sentimentDictionary)))
        for item in fnList:
            fn = fn+int(sentimentComplimentDictionary.get(item))
            
        tn=0
        for item in tnList:
            tn = tn+int(sentimentComplimentDictionary.get(item))
            
        confusionMatrix = np.array([tp,fp,fn,tn]).reshape(2,2)
        llrScore = self.calculateLLR(confusionMatrix)
        wordArray = [sentimentToken, llrScore, colorIndex]
        
        return(wordArray)
    
    def screenForStopWords(self, conditionVocabDictionary):
        
        bufferDictionary = {}
        for token in conditionVocabDictionary:
            truePattern = True
            token = str(token).strip()
            completeMatch = re.search('\d+|\W+|month|week|ca|pregn|I|We|me', token, flags=re.RegexFlag.IGNORECASE)
            if not completeMatch:
                for character in list(token):
                    partialMatch = re.findall('[a-zA-Z0-9]', character)
                    if not partialMatch:
                        truePattern = False
                if truePattern:
                    bufferDictionary.update({token:conditionVocabDictionary.get(token)})
        return(bufferDictionary)
    
    def flattenWordDictionary(self, conditionWordArray):
        
        for wordArray in conditionWordArray:
            currWord = wordArray[0]
            tier1List = wordArray[1:]
            if currWord in self.wordCloudDictionary:
                tier2List = self.wordCloudDictionary.get(currWord)
                if tier2List[0] > tier1List[0]:
                    tier1List = tier2List
            self.wordCloudDictionary.update({currWord:tier1List})
        
        return()
        
    def screenTokensForLLR(self):
        embeddingInstance = EmbeddingRepresentation.EmbeddingRepresentation
        embeddingInstance.__init__(embeddingInstance)
        embeddingInstance.readMultiClassLabel(embeddingInstance)
        self.multiCLassLabelInput = embeddingInstance.multiCLassLabelInput
        limitSize = 0
        
        while (limitSize < 3):
            print("Available sentiment types:",self.multiCLassLabelInput.keys(),"\n\t Pick a category>>")
            conInput = input()
            print(self.multiCLassLabelInput.get(conInput))
            
            conditionVocabList = self.getVocabToken(embeddingInstance, self.multiCLassLabelInput.get(conInput))
            conditionVocabDictionary = self.removeDuplications([conditionVocabList])
            print(conditionVocabDictionary)
            
            decoyDictionary = self.multiCLassLabelInput.copy()
            decoyDictionary.pop(conInput)
            vocabList = list(map(lambda dictItems:self.getVocabToken(embeddingInstance, dictItems[1]), decoyDictionary.items()))
            vocabDictionary = self.removeDuplications(vocabList)
            print(vocabDictionary)
            
            conditionVocabDictionary = self.screenForStopWords(conditionVocabDictionary)
            
            conditionWordArray = list(map(lambda vocabToken:self.generateConfusionMatrix(vocabToken, vocabDictionary, conditionVocabDictionary, self.wordShade.get(conInput)),conditionVocabDictionary))
            self.flattenWordDictionary(conditionWordArray)
            
            limitSize += 1
        
        writeFilePath = embeddingInstance.openConfigurationFile(embeddingInstance,'wordCloudDataPath')
        fileWriter = open(writeFilePath, mode='w+')
        for keyWord in self.wordCloudDictionary:
            decoyList = self.wordCloudDictionary.get(keyWord)
            entryText = keyWord+','+str(decoyList[0])+','+str(decoyList[1])+'\n'
            fileWriter.write(entryText)
            
        fileWriter.close()
        
        return()
    
llrSentimentInstance = LLR_SentimentCloud()
llrSentimentInstance.screenTokensForLLR()

