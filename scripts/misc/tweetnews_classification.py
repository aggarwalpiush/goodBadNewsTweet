#! usr/bin/env python
# -*- coding : utf-8 -*-

'''
input file should be tab separated 2 fields named tweetid and annotation
'''

import codecs
import pandas as pd
import numpy as np
import sys
import pymongo
from pymongo import MongoClient
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


CLIENT = MongoClient('mongodb://127.0.0.1:27017')
DB = CLIENT.mttweetlabelsDB
COLLECTIONS = ['Ebola', 'agu17','harvey_hurricane','hiv','iot','irma','macerata','nintendo','stockholm','swachhbharat']


def filter_tweets(input_data):
	'''
	every tweet should be unique with either support or deny label
	query tweets are also filtered out
	'''
	output_data = []
	tweet_ann = {}
	for token in input_data:
		if token[0] not in tweet_ann.keys():
			tweet_ann[token[0]] = [token[1]]
		else:
			if token[1] not in tweet_ann[token[0]]:
				tweet_ann[token[0]].append(token[1])
	for key,val in tweet_ann.items():
		if len(val) == 1:
			if val[0] != 'query':
				output_data.append([key,val[0]])
	return output_data

def get_tweet_text(tweetid):
	print(tweetid)
	for col in COLLECTIONS:
		collec = DB[col]
		tweet_text = collec.find_one({"id_str":str(int(tweetid)), "$or": [ { "full_text": {"$exists": True} }, { "text": {"$exists": True} } ]},{'full_text':1,'text':1,"_id":0})
		if not tweet_text is None:
			if 'text' in tweet_text.keys() and 'full_text' not in tweet_text.keys():
				tweet_text['full_text'] = tweet_text['text']
			if tweet_text['full_text'] != '':
				tweet_emb_text = collec.find_one({"id_str":str(int(tweetid)), "is_quote_status":True, "quoted_status": {"$exists": True}},{'quoted_status.text':1,'quoted_status.full_text':1,"_id":0})
				tweet_content = tweet_text['full_text'].rstrip('\n\r''"')
				if not tweet_emb_text is None:
					if 'text' in tweet_emb_text['quoted_status'].keys() and 'full_text' not in tweet_emb_text['quoted_status'].keys():
						tweet_emb_text['quoted_status']['full_text'] = tweet_emb_text['quoted_status']['text']
					tweet_content = tweet_text['full_text'].rstrip('\n\r''"') + ' tweeted on ' + tweet_emb_text['quoted_status']['full_text'].rstrip('\n\r''"')
				break

	if tweet_text is None:
		raise NameError('Bad Tweet id %s' %tweetid)	
	else:
		tweet_content = ' '.join(tweet_content.split('\n'))
		tweet_content = ' '.join(tweet_content.split('\r\n'))
		tweet_content = ' '.join(tweet_content.split('\r'))
		tweet_content = ' '.join(tweet_content.split('\n\r'))
		return tweet_content.rstrip('\n\r''"')

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
    for i in range(len(y_pred)):
    	if y_pred[i] != y_test[i]:
    		print("%s\%s" %(y_pred[i], X_test[i]))
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
	input_file = sys.argv[1]
	embedding_file = sys.argv[2]
	is_TFIDF = sys.argv[3]

	input_data = []

	with codecs.open(input_file, 'r', 'utf-8') as in_obj:
		for line in in_obj:
			line = line.strip().strip('\n\r')
			input_data.append(line.split('\t'))

	filtered_input_data = filter_tweets(input_data)

	tweet_text_vector = []
	labels = []
	for i in range(len(filtered_input_data)):
		labels.append(filtered_input_data[i][1])

	if is_TFIDF.strip().lower() == 'true' and not os.path.exists(input_file+'_tfidf.csv'):
		tweet_text_list = []
		if os.path.exists(input_file+'_class_inp.tsv'):
			with codecs.open(input_file+'_class_inp.tsv', 'r', 'utf-8') as class_file_obj:
				for line in class_file_obj:
					line = line.split('\t')
					tweet_text_vector.append(' '.join(preprocess_tweet_text(line[0])))
			vectorizer = TfidfVectorizer(ngram_range=(1, 3))
			X_transform  = vectorizer.fit_transform(tweet_text_vector).toarray()
			print(X_transform[:10])
			np.savetxt(input_file+'_tfidf.csv', X_transform, delimiter='\t')
		else:
			raise NameError('please provide label annotated texts file named %s_class_inp.tsv' %input_file)




	if not os.path.exists(input_file+'_embed.csv') and is_TFIDF.strip().lower() != 'true':
		tweet_text_list = []
		for i in range(len(filtered_input_data)):
			tweet_text = get_tweet_text(filtered_input_data[i][0])
			tweet_text_list.append(tweet_text)
			tweet_text_vector.append(preprocess_tweet_text(tweet_text))

		X_transform = get_embedding_vector(tweet_text_vector, embedding_file)

		out_data = np.append(np.transpose([np.array(tweet_text_list)]), np.transpose([labels]), axis=1)
		np.savetxt(input_file+'_class_inp.tsv', out_data, fmt='%s	%s', delimiter='\t')

		np.savetxt(input_file+'_embed.csv', X_transform, delimiter='\t')


	if is_TFIDF.strip().lower() != 'true':
		X_transform = np.loadtxt(input_file+'_embed.csv', delimiter='\t')
	else:
		X_transform = np.loadtxt(input_file+'_tfidf.csv', delimiter='\t')
	
	scaler = MinMaxScaler()

	X_transform_scaled = scaler.fit_transform(X_transform)
	print(len(X_transform_scaled))

	np.savetxt(input_file+'_embed_scaled.csv', X_transform_scaled, delimiter='\t')

	le = LabelEncoder()
	y = le.fit_transform(labels)
	print(len(y))

	rus = RandomUnderSampler(random_state=0)
	X_resample, y_resample = rus.fit_sample(X_transform_scaled, y)

	print(len(X_resample), len(y_resample))

	X_train, X_test, y_train, y_test = train_test_split(X_resample,
                                                    y_resample, stratify=y_resample, random_state=0)



	param_grid1 = {'activation': ['logistic'],'alpha': [0.001],'learning_rate': ['adaptive'], 'tol': [ 0.1]}
	gs1 = GridSearchCV(MLPClassifier(), 
             param_grid=param_grid1, scoring="f1_macro", cv=5)

	param_grid2 = {'C': [0.1, 1, 10], 'kernel' : ['rbf'], 'gamma' : [ 'scale']}
	gs2 = GridSearchCV(SVC(), 
             param_grid=param_grid2, scoring="f1_macro", cv=5)

	param_grid3 = {'C': [0.1, 1.0, 10.0], 'penalty' : [ 'l2', 'l1']}
	gs3 = GridSearchCV(LogisticRegression(), 
             param_grid=param_grid3, scoring="f1_macro", cv=5)

	param_grid4 = {'n_neighbors': [15,20,25,30,35], 'weights' : [ 'uniform', 'distance']}
	gs4 = GridSearchCV(KNeighborsClassifier(), 
             param_grid=param_grid4, scoring="f1_macro", cv=5)

	param_grid5 = {'n_estimators':[150,500,1000], 'max_depth':[4,5,10,15,20]}
	gs5 = GridSearchCV(RandomForestClassifier(), 
             param_grid=param_grid5, scoring="f1_macro", cv=5)

	param_grid6 = {'C': [0.1, 1.0, 10.0], 'penalty' : [ 'l2']}
	gs6 = GridSearchCV(LinearSVC(), 
             param_grid=param_grid6, scoring="f1_macro", cv=5)

	param_grid7 = {'min_samples_split':[2,3,4,5], 'min_samples_leaf':[1,2,3,4,5]}
	gs7 = GridSearchCV(DecisionTreeClassifier(), 
             param_grid=param_grid7, scoring="f1_macro", cv=5)

	param_grid8 = {}
	gs8 = GridSearchCV(XGBClassifier(), 
             param_grid=param_grid8, scoring="f1_macro", cv=5)


	save_scores(gs1, X_train, y_train, X_test, y_test, 'MLPClassifier')
	save_scores(gs2, X_train, y_train, X_test, y_test, 'SVC')
	save_scores(gs3, X_train, y_train, X_test, y_test, 'Logistic Regression')
	save_scores(gs4, X_train, y_train, X_test, y_test, 'K Nearest Neighbours')
	save_scores(gs5, X_train, y_train, X_test, y_test, 'Random Forest')
	save_scores(gs6, X_train, y_train, X_test, y_test, 'LinearSVC')
	save_scores(gs7, X_train, y_train, X_test, y_test, 'Descision Tree Classifier')
	save_scores(gs8, X_train, y_train, X_test, y_test, 'XGBoost Classifier')






	# print('\n\n==============MLPClassifier===============\n')
	# print_scores(gs1, X_train, y_train, X_test, y_test)
	# print('\n\n==============SVC===============\n')
	# print_scores(gs2, X_train, y_train, X_test, y_test)
	# print('\n\n==============Logistic Regression===============\n')
	# print_scores(gs3, X_train, y_train, X_test, y_test)
	# print('\n\n==============K Nearest Neighbours===============\n')
	# print_scores(gs4, X_train, y_train, X_test, y_test)
	# print('\n\n==============Random Forest===============\n')
	# print_scores(gs5, X_train, y_train, X_test, y_test)
	# print('\n\n==============LinearSVC===============\n')
	# print_scores(gs6, X_train, y_train, X_test, y_test)
	# print('\n\n==============Descision Tree Classifier===============\n')
	# print_scores(gs7, X_train, y_train, X_test, y_test)
	# print('\n\n==============XGBoost Classifier===============\n')
	# print_scores(gs8, X_train, y_train, X_test, y_test)

if __name__ == '__main__':
	main()











