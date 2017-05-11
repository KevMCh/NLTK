import nltk
import re
import os

from nltk.corpus import brown

nltk.corpus.brown.tagged_words()

string = open('./TextEn.txt', 'r').read()
print(type(string))
print(len(string))

tokens = nltk.word_tokenize(string)
print(type(tokens))
print(len(tokens))

tag = nltk.pos_tag(tokens)
print(tag)

resultFile = open('./result.txt', 'w')
for elem in tag:
    resultFile.write(str(elem) + " ")

resultFile.close()

porter = nltk.PorterStemmer()
porterList =  [porter.stem(t) for t in tokens]
print(porterList)

lancaster = nltk.LancasterStemmer()
lancasterList = [lancaster.stem(t) for t in tokens]
print(lancasterList)

wnl = nltk.WordNetLemmatizer()
wnlList = [wnl.lemmatize(t) for t in tokens]
print(wnlList)
