from nltk.corpus import movie_reviews
from nltk.probability import FreqDist
from nltk.corpus import stopwords
import numpy as np
import string
import os
import sys
import re
import json
import nltk
import random
from sklearn import svm
path = '../data'

class test_pte(object):

	def __init__(self):

		self.train_documents = []
		self.train_labels = []
		self.test_documents = []
		self.test_actual_labels = []
		self.test_predicted_labels = []
		nltk.data.path.append(path)
		self.all_words = json.load(open('word_mapping.json'))
		self.word_embedding = np.load('lookupW.npy')
		self.D = 100

	def load_data(self):
		documents = [(list(w.lower() for w in movie_reviews.words(fileid) if w.lower() not in string.punctuation), category, fileid) for category in movie_reviews.categories() for fileid in movie_reviews.fileids(category)]
		random.shuffle(documents)
		train_set =  documents[0:len(documents)/2]
		test_set =  documents[len(documents)/2:len(documents)]
		return train_set, test_set

	def test(self):

		train_set,test_set = self.load_data()
		for index in range(len(train_set)):
			document_sum = np.zeros(self.D)
			for word_index in range(len(train_set[index][0])):
				word = train_set[index][0][word_index]
				i = self.all_words[word]
				embedding = self.word_embedding[i]
				document_sum = np.add(document_sum, embedding)
			document_average = np.divide(document_sum, len(train_set[index][0]))
			self.train_documents.append(document_average)
			self.train_labels.append(train_set[index][1])
		for index in range(len(test_set)):
			document_sum = np.zeros(self.D)
			for word_index in range(len(test_set[index][0])):
				word = test_set[index][0][word_index]
				i = self.all_words[word]
				embedding = self.word_embedding[i]
				document_sum = np.add(document_sum, embedding)
			document_average = np.divide(document_sum, len(test_set[index][0]))
			self.test_documents.append(document_average)
			self.test_actual_labels.append(train_set[index][1])
		clf = svm.SVC()
		clf.fit(self.train_documents, self.train_labels)
		self.test_predicted_labels = clf.predict(self.test_documents)
		correct = 0
		for i in range(len(self.test_predicted_labels)):
			if self.test_predicted_labels[i] == self.test_actual_labels[i]:
				correct = correct + 1
		accuracy = (correct*100)/len(self.test_predicted_labels)
		print 'Accuracy : ',accuracy
        

if __name__ == "__main__":
    pte = test_pte()
    pte.test()