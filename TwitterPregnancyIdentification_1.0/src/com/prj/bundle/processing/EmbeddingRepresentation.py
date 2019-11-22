'''
Created on Jun 16, 2018

@author: neha
'''
import os
import sys
import re
import json
import nltk
from nltk.tokenize import word_tokenize
import numpy as np
from gensim.models import KeyedVectors
from tensorflow import set_random_seed
from numpy.random import seed

class EmbeddingRepresentation:

    def __init__(self):
        print("initializing Embedding Representation")
        self.contextTerms = 'pregnan'
        self.resourceData = {}
        self.multiClassLabelResource = {}
        self.multiCLassLabelInput = {}
        self.tweetSize = 0
        self.engStopwords = set(nltk.corpus.stopwords.words('english'))
        self.EMBEDDING_DIM = 300
        EMBEDDING_FILE = self.openConfigurationFile(self, "wordEmbeddingPath")
        self.PRETRAINED_GLOVE_EMBEDDING = KeyedVectors.load_word2vec_format(EMBEDDING_FILE,binary=True)
        self.ASSIGNED_EMBEDDING = {} 
        self.TWEET_EMBEDDING = {}

    def openConfigurationFile(self,jsonVariable):
        
        path = os.path.dirname(sys.argv[0])
        tokenMatcher = re.search(".*TwitterPregnancyIdentification_1.0\/", path)
        if tokenMatcher:
            configFile = tokenMatcher.group(0)
            configFile="".join([configFile,"config.json"])
        
        jsonVariableValue = None
        with open(configFile, "r") as json_file:
            data = json.load(json_file)
            jsonVariableValue = data[jsonVariable]
            json_file.close()
            
        if jsonVariableValue is not None:
            return(jsonVariableValue)
        else:
            print("\n\t Variable load failure")
            sys.exit()
        return()
    
    def performSentenceSplit(self, sentenceBundle):
        tokenizer = nltk.data.load('nltk:tokenizers/punkt/english.pickle')
        bundledSent = tokenizer.tokenize(sentenceBundle)
        return(bundledSent)
    
    def removeStopWords(self,sentence):
        tokenizedSentence = word_tokenize(sentence)
        filteredSentence = filter(lambda word: word not in self.engStopwords, tokenizedSentence)
        filteredSentence = ' '.join(filteredSentence)
        return(filteredSentence)
    
    def performWordSplit(self, sentence):
        '''return(word_tokenize(sentence))'''
        return(sentence.split(sep=' '))
            
    def readResourceData(self,fileDataPath):
        
        with open(fileDataPath, "r") as bufferFile:
            currentData = bufferFile.readline()
            while len(currentData)!=0:
                currentData = str(currentData).strip()
                decoyList = currentData.split(sep="\t", maxsplit=1)
                if len(decoyList) == 2:
                    instanceKey = int(decoyList[0])
                    tweetInstance = str(decoyList[1])
                    tweetInstance = self.removeStopWords(self, tweetInstance)
                    tweetTokenList = list(filter(lambda token:len(token.strip())>0,self.performWordSplit(self, tweetInstance)))
                    if self.tweetSize < len(tweetTokenList):
                        self.tweetSize = len(tweetTokenList)
                    decoyDictionary = {}
                    if instanceKey in self.resourceData:
                        decoyDictionary = self.resourceData.get(instanceKey)
                    currIndex = len(decoyDictionary)
                    decoyDictionary.update({currIndex:tweetTokenList})
                    self.resourceData.update({instanceKey:decoyDictionary})
                    
                currentData = bufferFile.readline()
        self.tweetSize = self.tweetSize+1
        return()
    
    def readMultiClassResource(self, fileDataPath):
        
        currIndex = 0
        with open(fileDataPath, "r") as bufferFile:
            currentData = bufferFile.readline()
            while len(currentData)!=0:
                currentData = str(currentData).strip()
                decoyList = currentData.split(sep="\t", maxsplit=1)
                if len(decoyList) == 2:
                    instanceKey = str(decoyList[0])
                    instanceValue = str(decoyList[1])
                    decoyArray = []
                    decoyList = []
                    if instanceKey in self.multiClassLabelResource:
                        decoyArray = self.multiClassLabelResource.get(instanceKey)
                        decoyList = self.multiCLassLabelInput.get(instanceKey)
                    decoyArray.append(currIndex)
                    decoyList.append(instanceValue)
                    self.multiClassLabelResource.update({instanceKey:decoyArray})
                    self.multiCLassLabelInput.update({instanceKey:decoyList})
                currIndex += 1    
                currentData = bufferFile.readline()
        return()
    
    def assignRandomEmbedding(self, token):
        
        if token not in self.ASSIGNED_EMBEDDING.keys():
            assignedEmbeddingVector = np.random.rand(1,self.EMBEDDING_DIM)
            #assignedEmbeddingVector = np.ones((1,self.EMBEDDING_DIM))
            self.ASSIGNED_EMBEDDING.update({token:assignedEmbeddingVector})
        return()
    
    def retreiveSeenPreTrainedEmbedding(self,embeddingMatrix, index, token, status):
        
        multiplicationFactor = float(1)
        if status == 1:
            embeddingMatrix[index] = (self.PRETRAINED_GLOVE_EMBEDDING.word_vec(token)[0:self.EMBEDDING_DIM]*multiplicationFactor)
        else:
            self.assignRandomEmbedding(self, token)
            embeddingMatrix[index] = (self.ASSIGNED_EMBEDDING[token]*multiplicationFactor)
        return(embeddingMatrix)
    
    def recursiveTokenIdentification(self, currentToken, remainderToken, wordSubTokens):
        
        startIndex = 0
        endIndex = len(currentToken)
        #print("range>>",startIndex,">>>",endIndex,">>",currentToken,"***>>>",remainderToken)
        termIndex = endIndex
        #print("index",termIndex,"token>>",currentToken[termIndex-1])
        bufferToken = currentToken[startIndex:termIndex]
            
        flag = 0
        if bufferToken in self.PRETRAINED_GLOVE_EMBEDDING.vocab:
            dicIndex = len(wordSubTokens)
            wordSubTokens.update({dicIndex:{1:bufferToken}})
            flag=1
                
        if ((flag == 0) and (termIndex > 1)):
            ''' reducing one letter at a time'''
            remainderToken.append(currentToken[termIndex-1:])
            currentToken = bufferToken[:termIndex-1]
        elif(flag == 1):
            ''' subgroup word structure'''
            if len(remainderToken) > 0:
                remainderToken.reverse()
                currentToken = ''.join(charTerm for charTerm in remainderToken)
                remainderToken = list()
            else:
                currentToken = None
        else:
            ''' for single words not present with embedding'''
            dicIndex = len(wordSubTokens)
            wordSubTokens.update({dicIndex:{-1:bufferToken}})
            currentToken = None
            
        if currentToken is not None:
            self.recursiveTokenIdentification(self, currentToken, remainderToken, wordSubTokens)
        
        return(wordSubTokens)
    
    def retrieveUnseenEmbeddings(self,embeddingMatrix, index, token):
        
        wordSubTokens = {}
        wordSubTokens = self.recursiveTokenIdentification(self, token,list(), wordSubTokens)
        #print("sub>>>",wordSubTokens)
        tempEmbeddingMatrix = np.zeros((len(wordSubTokens),self.EMBEDDING_DIM))
        for subIndex,dictionaryIndex in enumerate(wordSubTokens):
            subTokenItem = wordSubTokens[dictionaryIndex]
            #print(">>>",subTokenItem)
            subTokenStatus = list(subTokenItem.keys())[0]
            subToken = list(subTokenItem.values())[0]
            tempEmbeddingMatrix = self.retreiveSeenPreTrainedEmbedding(self, tempEmbeddingMatrix, subIndex, subToken, subTokenStatus)
            
        #print("tempEmbeddingMatrix>>>",tempEmbeddingMatrix)
        embeddingMatrix[index] = np.array(np.prod(np.array(tempEmbeddingMatrix),axis=0))
        return(embeddingMatrix)
    
    def assimilatePreTrainedEmbeddings(self, sentenceTokens):

        embeddingMatrix = np.zeros((self.tweetSize,self.EMBEDDING_DIM))        
        for index,token in enumerate(sentenceTokens):
            if token in self.PRETRAINED_GLOVE_EMBEDDING.vocab:
                embeddingMatrix = self.retreiveSeenPreTrainedEmbedding(self,embeddingMatrix, index, token, 1)
            else:
                print(token)
                embeddingMatrix = self.retrieveUnseenEmbeddings(self, embeddingMatrix, index, token)
        
        return(embeddingMatrix)
    
    def findTweetVectorEmbeddings(self):
        
        for tier1MapValue in self.resourceData.items():
            instanceType = tier1MapValue[0]
            tier2MapValue = dict(tier1MapValue[1])
            decoyEmbeddingDictionary = {}
            for tier3MapValue in tier2MapValue.items():
                decoyIndex = tier3MapValue[0]
                embeddingMatrix = self.assimilatePreTrainedEmbeddings(self, list(tier3MapValue[1]))
                decoyEmbeddingDictionary.update({decoyIndex:embeddingMatrix})
             
            self.TWEET_EMBEDDING.update({instanceType:decoyEmbeddingDictionary})   
        return()
    
    ''' set config path '''
    def loadResource(self):
            
        trainDataPath = self.openConfigurationFile(self,"trainDataPath")
        self.readResourceData(self,trainDataPath)
        print("max sentence size>>",self.tweetSize)
        self.findTweetVectorEmbeddings(self)
        return()
    
    def readMultiClassLabel(self):
        
        multiClassDataPath = self.openConfigurationFile(self,"multiClassDataPath")
        self.readMultiClassResource(self, multiClassDataPath)
        for labelKeys in self.multiClassLabelResource:
            print("\t class label >>",labelKeys,"\t size>>",len(self.multiClassLabelResource.get(labelKeys)))
            
        return()
        
seed(1)
set_random_seed(2)        