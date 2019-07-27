#! usr/bin/env python
# -*- coding : utf-8 -*-

'''
input file should be tab separated 2 fields named tweetid and annotation
'''

import codecs
import pandas as pd
import numpy as np
import sys
from embloader import MeanEmbeddingTransformer
from preprocessor_arc import Arc_preprocessor
from sklearn.metrics import classification_report
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.metrics import roc_auc_score
from collections import Counter
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from imblearn.under_sampling import RandomUnderSampler
from sklearn.model_selection import GridSearchCV
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
import os
from time import sleep



def preprocess_tweet_text(tweet_text):
    arc_obj = Arc_preprocessor()
    tokenized_tweet_text = [x.strip('\n\r').lower() for x in arc_obj.tokenizeRawTweetText(tweet_text)]
    tokenized_tweet_text = ['url' if 'http' in str(x) else x for x in tokenized_tweet_text]
    return tokenized_tweet_text


def get_embedding_vector(tweet_text_vector, embedding_file):
	met = MeanEmbeddingTransformer(embedding_file)
	X_transform = met.fit_transform(tweet_text_vector)
	return X_transform


def print_scores(clf, X_train, y_train, X_test, y_test):
    clf.fit(X_train, y_train)
    print(clf.best_params_)
    y_pred = clf.predict(X_test)
    y_pred = [round(value) for value in y_pred]
    target_names = ['News', 'Not News']
    print('Precision score: {:3f}'.format(precision_score(y_test, y_pred, average='macro') ))
    print('Recall score: {:3f}'.format(recall_score(y_test, y_pred, average='macro') ))
    print('F1 score: {:3f}'.format(f1_score(y_test, y_pred, average='macro')))
    print('AUC score: {:3f}'.format(roc_auc_score(y_test, y_pred)))
    print('Confusion Metric : %s' %(confusion_matrix(y_test, y_pred)))
    print(classification_report(y_test, y_pred, target_names=target_names))


def save_scores(clf, X_train, y_train, X_test, y_test, clf_name):
	clf.fit(X_train, y_train)
	y_pred = clf.predict(X_test)
	y_pred = [round(value) for value in y_pred]
	target_names = ['News', 'Not News']
	with codecs.open('score_report.txt', 'a', 'utf-8') as scr_obj:
		scr_obj.write('\n\n====================%s===================\n' %clf_name)
		scr_obj.write(str([str(key)+' : '+str(val) for key,val in clf.best_params_.items()]))
		scr_obj.write('\nPrecision score: {:3f}\n'.format(precision_score(y_test, y_pred, average='macro') ))
		scr_obj.write('Recall score: {:3f}\n'.format(recall_score(y_test, y_pred, average='macro') ))
		scr_obj.write('F1 score: {:3f}\n'.format(f1_score(y_test, y_pred, average='macro')))
		scr_obj.write('AUC score: {:3f}\n'.format(roc_auc_score(y_test, y_pred)))
		scr_obj.write('Confusion Metric : %s\n' %(confusion_matrix(y_test, y_pred)))
		scr_obj.write(classification_report(y_test, y_pred, target_names=target_names))

def main():
	train_file = sys.argv[1]
	test_file = sys.argv[2]
	embedding_file = sys.argv[3]


	train_data = []
	train_label = []
	test_data = []
	test_label = []

	with codecs.open(train_file, 'r', 'utf-8') as in_obj:
		for line in in_obj:
			line = line.strip().strip('\n\r')
			train_data.append(line.split('\t')[0])
			train_label.append(line.split('\t')[1])


	with codecs.open(test_file, 'r', 'utf-8') as in_obj:
		for line in in_obj:
			line = line.strip().strip('\n\r')
			test_data.append(line.split('\t')[0])
			test_label.append(line.split('\t')[1])


	train_data_size = len(train_data)

	parttions_index = round(train_data_size/10)



	in_data = train_data + test_data

	tweet_text_vector = []

	for tweet_text in in_data:
		tweet_text_vector.append(preprocess_tweet_text(tweet_text))

	if not os.path.exists(train_file+'_embed.csv'):

		X_transform = get_embedding_vector(tweet_text_vector, embedding_file)

		np.savetxt(train_file+'_embed.csv', X_transform, delimiter='\t')
	else:
		X_transform = np.loadtxt(train_file+'_embed.csv', delimiter='\t')



	
	scaler = MinMaxScaler()

	X_transform_scaled = scaler.fit_transform(X_transform)

	X_transform_scaled_train = X_transform_scaled[:train_data_size]

	print(len(X_transform_scaled_train))

	X_transform_scaled_test = X_transform_scaled[train_data_size:]





	le = LabelEncoder()
	y_train = le.fit_transform(train_label)
	print(len(y_train))



	rus = RandomUnderSampler(random_state=0)

	param_grid1 = {'C': [10], 'kernel' : ['rbf'], 'gamma' : [ 'scale']}
	gs1 = GridSearchCV(SVC(), param_grid=param_grid1, scoring="f1_macro", cv=5)

	y_test = le.fit_transform(test_label)



	for i in range(10):
		if i < 9:
			#X_resample, y_resample = rus.fit_sample(X_transform_scaled_train[:(i+1)*parttions_index], y_train[:(i+1)*parttions_index])
			X_resample, y_resample = X_transform_scaled_train[:(i+1)*parttions_index], y_train[:(i+1)*parttions_index]
		else:
			X_resample, y_resample = X_transform_scaled_train, y_train
		print(len(X_resample), len(y_resample))

		print("=============%s train data==========================" %((i+1)*10))

		print_scores(gs1, X_resample, y_resample, X_transform_scaled_test, y_test)


if __name__ == '__main__':
	main()











