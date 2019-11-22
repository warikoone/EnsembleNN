'''
Created on Jul 16, 2018

@author: iasl
'''
import sys
import re
import numpy as np
from nltk.tokenize import word_tokenize
from gensim.models import KeyedVectors
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from src.com.prj.bundle.processing import EmbeddingRepresentation

class SentimentCloudRepresentation:

    def __init__(self):
        self.multiCLassLabelInput = {}
        self.PRETRAIN_WORD2VEC = KeyedVectors.load_word2vec_format("/home/iasl/Disk_R/Bio_NLP/NeonWorkspace_1.2/TwitterPregnancyIdentification_1.0/src/com/prj/bundle/resource/GoogleNews-vectors-negative300.bin",binary=True)
        
    def screenTokensForLLR(self):
        embeddingInstance = EmbeddingRepresentation.EmbeddingRepresentation
        embeddingInstance.__init__(embeddingInstance)
        embeddingInstance.readMultiClassLabel(embeddingInstance)
        self.multiCLassLabelInput = embeddingInstance.multiCLassLabelInput
        print("Available sentiment types:",self.multiCLassLabelInput.keys(),"\n\t Pick a category>>")
        conInput = input()
        print(self.multiCLassLabelInput.get(conInput))
        inputText ='. '.join(str(item).lower() for item in self.multiCLassLabelInput.get(conInput))
        print(inputText)
        inputText = embeddingInstance.removeStopWords(embeddingInstance, inputText)
        print("after>>>",inputText)
        #tokenizedInput = word_tokenize(inputText)
        tokenizedInput = re.split('\s+', inputText)
        wordDict={}
        for index,value in enumerate(tokenizedInput):
            count=0
            match = re.search('\d+|\W+|month|week|ca', value, flags=re.RegexFlag.IGNORECASE)
            if not match:
                if value in self.PRETRAIN_WORD2VEC.vocab:
                    if value in wordDict:
                        count = wordDict.get(value)
                        
                    cosineSimilarityScore = np.around(self.PRETRAIN_WORD2VEC.similarity('annoyed',value), decimals=3)
                    print("word>>",value,"\t::",cosineSimilarityScore)
                    if cosineSimilarityScore >= 0.2:
                        #count = count + (cosineSimilarityScore*3)
                        count = int((count+1.5)) 
                    else:
                        count += 1
                        
                    wordDict.update({value:int(count)})
        print(wordDict)
        newInput = []
        for item in wordDict:
            if wordDict.get(item) >= 1:
                temp = str(item+' ') * int(wordDict.get(item))
                #temp = str(item)
                newInput.append(temp)
                
        
        inputText = ' '.join(str(value) for value in newInput)
        print("after>>",inputText)
        
        '''
        cloudRep = WordCloud().generate(inputText)
        
        plt.imshow(cloudRep, interpolation='bilinear')
        plt.axis('off')
        '''
        
        wordcloud = WordCloud(width=480,height=480,stopwords=['pregnant'],background_color='steelblue',colormap= 'Reds',max_font_size=65).generate(inputText)
        wordcloud.to_file('/home/iasl/Disk_R/Bio_NLP/NeonWorkspace_1.2/TwitterPregnancyIdentification_1.0/src/com/prj/bundle/resource/image.png')
        plt.figure()
        plt.imshow(wordcloud, interpolation="bilinear")
        plt.axis("off")
        plt.show()
        
        return()

sentimentInstance = SentimentCloudRepresentation()
sentimentInstance.screenTokensForLLR()
