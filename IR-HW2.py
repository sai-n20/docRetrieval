#!/usr/bin/env python
# coding: utf-8

## Information Retrieval HW2

import os
import re
import nltk
import sys
import numpy as np
import pandas as pd
from string import punctuation
from nltk.stem import PorterStemmer
from collections import Counter
from bs4 import BeautifulSoup
from nltk.corpus import stopwords
nltk.download('stopwords')
stop_list = stopwords.words('english')
ps = PorterStemmer()

__location__ = os.path.realpath(os.path.join(
    os.getcwd(), os.path.dirname(__file__)))

docsFolder = "cranfieldDocs"
if(len(sys.argv) > 1):
    docsFolder = sys.argv[1]
print("Currently chosen folder for cranfield docs:", docsFolder)

#Pre process docs by stemming, removing stop words and punctuation
docCorpus = []
for filename in os.listdir(os.path.join(__location__, docsFolder)):
    with open(os.path.join(__location__, docsFolder) + '/' + filename) as contents:
        temp = contents.read()
        temp = temp.split()
        temp = [ps.stem(word) for word in temp]
        temp = [item for item in temp if item not in stop_list]
        temp = [item for item in temp if len(item) > 2]
        temp = (" ").join(temp)
        soup = BeautifulSoup(temp, features="html.parser")

        title = str(soup.find("title"))
        text = str(soup.find("text"))

        text = text.replace("<text>", "").replace("</text>", "").replace("\n", " ").strip().lower()
        for i in punctuation:
            text = text.replace(i, "")
        text = re.sub("\d", "", text)

        title = title.replace("<title>", "").replace("</title>", "").replace("\n", " ").strip().lower()
        for i in punctuation:
            title = title.replace(i, "")
        title = re.sub("\d", "", title)
        text = title + " " + text
        docCorpus.append(text)
    contents.close()


# Preprocessing queries
queries = []
with open(os.path.join(__location__, "queries.txt")) as que:
    text = que.readlines()
    for line in text:
        line = line.split()
        line = [ps.stem(word) for word in line]
        line = [item for item in line if item not in stop_list]
        line = [item for item in line if len(item) > 2]
        line = (" ").join(line)
        for i in punctuation:
            line = line.replace(i, "")
        line = re.sub("\d", "", line)
        queries.append(line.strip())


#Word freq dict and subsequent list
tokenFreq = {}
for doc in docCorpus:
    for word in doc.split():
        if(word in tokenFreq):
            tokenFreq[word] += 1
        else:
            tokenFreq[word] = 1
vocab = [term for term in tokenFreq]


#Create tfIDF dict of tuples
tfIDF = {}
for number, doc in enumerate(docCorpus):
    docTokens = doc.split()
    wordCounts = Counter(docTokens)
    totalWords = len(docTokens)
    for word in np.unique(docTokens):
        tf = wordCounts[word]/totalWords
        df = tokenFreq[word]
        idf = np.log((len(docCorpus))/df)

        tfIDF[number, word] = tf*idf


#Document vector from tfIDF
docVector = np.zeros((len(docCorpus), len(vocab)))
for item in tfIDF:
    index = vocab.index(item[1])
    docVector[item[0]][index] = tfIDF[item]


#Vectorize query in the same shape as document vector
def vectorizeQuery(data):
    vecQ = np.zeros((len(vocab)))
    wordCounts = Counter(data)
    totalWords = len(data)
    
    for token in np.unique(data):
        tf = wordCounts[token]/totalWords
        df = tokenFreq[token] if token in vocab else 0
        idf = np.log((len(docCorpus)+1)/(df+1))
        try:
            ind = vocab.index(token)
            vecQ[ind] = tf*idf
        except:
            pass
    return vecQ


#Calculate cosine similarity and arrange docs in descending order
def cosineSimilarity(query, k):
    tokens = query.split()
    cosines = []
    queryVector = vectorizeQuery(tokens)
    for item in docVector:
        cosines.append(np.dot(queryVector, item)/(np.linalg.norm(queryVector)*np.linalg.norm(item)))
    if k > 0:
        # top k docs in descending order    
        return np.array(cosines).argsort()[-k:][::-1]
    else:
        # consider all docs
        return np.array(cosines).argsort()[::-1]


#Run queries against cosine similarities and return docs
def queryLookup(k):
    queryDoc = []
    for i in range(len(queries)):
        queryDoc.append([i, cosineSimilarity(queries[i], k)])
    return queryDoc


relevance = pd.read_csv(os.path.join(__location__, "relevance.txt"), delim_whitespace=True, names=['qNo', 'relDoc'], header=None)

#Relevant docs per query
queryRelevanceList = []
for query in range(1,11):
    queryRelevanceList.append(relevance[relevance['qNo']==query]['relDoc'].to_list())


#Calc precision as number of correctly retrieved docs/number of retrieved docs
def precisionCalc(k):
    precisionList = []
    for i in range(len(queries)):
        lookup = queryLookup(k)[i][1].tolist()
        correct = [item for item in lookup if item + 1 in queryRelevanceList[i]]
        correct = len(correct)
        pres = correct / k
        precisionList.append(pres)
    return precisionList


#Calc recall as number of correctly retrieved docs/number of relevant docs
def recallCalc(k):
    recallList = []
    for i in range(len(queries)):
        lookup = queryLookup(k)[i][1].tolist()
        correct = [item for item in lookup if item + 1 in queryRelevanceList[i]]
        correct = len(correct)
        recall = correct / len(queryRelevanceList[i])
        recallList.append(recall)
    return recallList


for k in [10, 50, 100, 500]:
    print("For top", k, "documents in the ranking:-")
    pres = precisionCalc(k)
    recall = recallCalc(k)
    for i in range(len(queries)):
        print("Query", i+1, "\tPrecision:", np.round(pres[i], 3), " \t Recall:", np.round(recall[i], 3))
    print("\nAverage precision over", k, "documents in ranking:", np.round(np.mean(pres), 2), "(", np.round(np.mean(pres)*100, 2), "% )")
    print("Average recall over", k, "documents in ranking:", np.round(np.mean(recall), 3), "(", np.round(np.mean(recall)*100, 2), "% )\n")