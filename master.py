#ChatBot
#imports
import nltk
import numpy
import tflearn
import tensorflow
import random
import json
from nltk.stem.lancaster import LancasterStemmer

#Calling LancasterStemmer
stemmer = LancasterStemmer()

#Opening intents file and reading the contents into data
with open("intents.json") as file:
    data = json.load(file)

words = []
labels = []
docs_x = []
docs_y = []

for intent in data['intents']:
    for pattern in intent['patterns']:

        #Split Patterns into words
        wrds = nltk.word_tokenize(pattern)
        words.extend(wrds)

        #Creating list of patterns {docs_x} with list of corresponding tags {docs_y}
        docs_x.append(pattern)
        docs_y.append(intent['tag'])

    #Creating list of unique tags
    if intent["tag"] not in labels:
        labels.append(intent['tag'])

#Finding root word in tokenized words and sorting it
words = [stemmer.stem(w.lower()) for w in words]
words = sorted(list(set(words)))
labels = sorted(labels)

training = []
output = []

#Creating a list of 0's with list length as the no. of lables
out_empty = [0 for _ in range(len(labels))]

for x, doc in enumerate(docs_x):
    bag = []

    #Stem docs_x
    wrds = [stemmer.stem(w) for w in doc]
    
    #Creating a bag of words using One Hot Encoding
    for w in words:
        if w in wrds:
            bag.append(1)
        else:
            bag.append(0)

    #Creating a list with indexs of corresponding labels denoting 1
    output_row = out_empty[:]
    output_row[labels.index(docs_y[x])] = 1

    training.append(bag)
    output.append(output_row)

#Convert into numpy arrays for tflearn
training = numpy.array(training)
output = numpy.array(output)


print(training)
print("---------------------")
print(output)