#! /usr/bin/env python
#*-- coding : utf-8 --*

import math
from textblob import TextBlob as tb
from sklearn.feature_extraction.text import TfidfVectorizer
import nltk
from nltk.corpus import stopwords 
import numpy as np
import os
import codecs
from tqdm import tqdm
from collections import defaultdict
from config.feature_config import TOP_N_TERMS, TOP_N_GRAMS, TERMS_FILE_PATH, TEXT_TYPE

STOP_WORDS = set(stopwords.words('english'))

class ExtractTerms(object):
	
	def __init__(self, corpus, labels):
		self.top_n_terms = TOP_N_TERMS
		self.num_classes = len(np.unique(labels))
		self.ngram = TOP_N_GRAMS
		self.corpus_matrix = []
		for i, instance in enumerate(corpus):
			self.corpus_matrix.append([corpus[i],labels[i]])


	def tf(self, word, blob):
		return blob.words.count(word) / len(blob.words)

	def n_containing(self, word, bloblist):
		return sum(1 for blob in bloblist if word in blob.words)

	def idf(self, word, bloblist):
		return math.log(len(bloblist) / (1 + self.n_containing(word, bloblist)))

	def tfidf(self, word, blob, bloblist):
		return self.tf(word, blob) * self.idf(word, bloblist)

	# def get_top_terms(self):
	# 	term_list = []
	# 	bloblist = []
	# 	for i in range(self.num_classes):
	# 		document = tb(' '.join([doc[0] for doc in self.corpus_matrix if doc[1] == i]))
	# 		bloblist.append(document)

	# 	for i, blob in tqdm(enumerate(bloblist)):
	# 	    scores = {' '.join(word): self.tfidf(' '.join(word), blob, bloblist) 
	# 	    for word in blob.ngrams(n = self.ngram)}
	# 	    sorted_words = sorted(scores.items(), key=lambda x: x[1], reverse=True)
	# 	    term_list.append(set([w for w,_ in sorted_words[:self.top_n_terms]]))
	# 	return term_list


	def get_top_term_list(self):
		term_list = []
		bloblist = []
		vectorizer = TfidfVectorizer(ngram_range=(1,self.ngram), stop_words='english')
		for i in range(self.num_classes):
			document = [doc[0] for doc in self.corpus_matrix if doc[1] == i]		
			bloblist.append(document)
		for i, blob in tqdm(enumerate(bloblist)):
			response = vectorizer.fit_transform(blob)
			#feature_array = np.array(vectorizer.get_feature_names())
			#tf_idf_result = np.asarray(response.sum(axis=0)).ravel()
			#tfidf_sorting = np.argsort(response.toarray()).flatten()[::-1]
			scores = zip(vectorizer.get_feature_names(),
                 np.asarray(response.sum(axis=0)).ravel())
			sorted_scores = sorted(scores, key=lambda x: x[1], reverse=True)
			features_by_gram = defaultdict(list)
			for element in sorted_scores:
				features_by_gram[len(element[0].split(' '))].append(element[0])
			for gram, features in features_by_gram.items():
				term_list.append(set(features[:self.top_n_terms]))
		return term_list


	def get_imp_vector(self, new_corpus):
		terms_file = os.path.join(TERMS_FILE_PATH, 
			'sig_term_vector_top_' + str(self.top_n_terms) + '_gram_' + str(self.ngram) + '_' +TEXT_TYPE + '.npy')
		if not os.path.exists(terms_file):
			extterms = self.get_top_term_list()
			save_exttterms = ['\t'.join(x) for x in extterms]
			np.savetxt(terms_file, save_exttterms, fmt='%s')
		else:
			extterms = []
			with codecs.open(terms_file, 'r', 'utf-8') as term_list_obj:
				for cl_terms in term_list_obj:
					extterms.append(set(cl_terms.replace('\n','').split('\t')))
		imp_vec = np.zeros((len(new_corpus), len(extterms)), float)

		for i, instance in tqdm(enumerate(new_corpus)):
			instance = ' '.join([w for w in instance.split(' ') if w not in STOP_WORDS])
			for j in range(self.ngram):
				for w in tb(instance).ngrams(n = j+1):
					w = ' '.join(w)
					for c in range(len(extterms)):
						if w in extterms[c]:
							imp_vec[i][c] = 1.0

		return imp_vec




if __name__ == '__main__':
	text_corpus = ['i am very happy to be in germany', 'germany is very boring place', 'academically germany is very strong',
	'indians are mean in germany']
	labels = [1, 0, 1, 0]
	print(ExtractTerms(text_corpus,labels).get_imp_vector(text_corpus))
	# extterms = ExtractTerms(text_corpus, 4,2,2).get_top_terms()
	# if 'very boring' in extterms[1]:
	# 	print('true')
	# print(np.array(extterms)[0])






