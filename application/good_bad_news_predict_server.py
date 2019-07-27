#! usr/bin/env python
# *-- coding : utf-8 --*


from werkzeug.wrappers import Request, Response
from werkzeug.serving import run_simple

from jsonrpc import JSONRPCResponseManager, dispatcher

from preprocessor_arc import Arc_preprocessor
from joblib import dump, load
from sklearn import svm
from tqdm import tqdm
import os
import pandas as pd
import numpy as np
import re


def remove_num(str):
	string_no_numbers = re.sub("\d+", " ", str)
	return string_no_numbers
	

def get_embedding_dictionary(embedding_file):
	E = {}
	with codecs.open(embedding_file, 'r', encoding="utf8") as file:
		for i, line in tqdm(enumerate(file)):
			if i == 0:
				continue
			l = line.split(' ')
			if l[0].isalpha():
				v = [float(i) for i in l[1:]]
				E[l[0]] = np.array(v)
	return E   



def get_embedding_vector(tweet, embedding_file):
	word_array = []
	embeddings =  get_embedding_dictionary(embedding_file)
	for token in tweet:
		if  token.lower().strip() in embeddings.keys():
			word_array.append(embeddings[token.lower().strip()])
		else:
			word_array.append(np.zeros([len(v) for v in embeddings.values()][0]))

	return np.mean(np.array(word_array), axis=0)





@dispatcher.add_method
def get_request(tweet):

	result_dict = {}

	# preprocessing
	arc_obj = Arc_preprocessor()
	tokenized_tweet_text = [x.strip('\n\r') for x in arc_obj.tokenizeRawTweetText(tweet)]
	tokenized_tweet_text = ['url' if 'http' in str(x) else x for x in tokenized_tweet_text]
	tokenized_tweet_text = [remove_num(x) for x in tokenized_tweet_text]
	tokenized_tweet_text = [x for x in tokenized_tweet_text if not x == '']

	tweet_transform = get_embedding_vector(tokenized_tweet_text, 'crawl-300d-2M.vec').reshape(1,-1)



	# prediction - news or not
	clf_news_not = load('tweet_news_not_model.joblib') 
	clf_good_bad = load('tweet_good_bad_model.joblib')
	result_dict['notnews_vs_news'] = clf_news_not.predict_proba(tweet_transform)[0][0]
	result_dict['bad_vs_good'] = clf_good_bad.predict_proba(tweet_transform)[0][0]

	print(result_dict)
	return result_dict


@Request.application
def application(request):
    response = JSONRPCResponseManager.handle(
        request.data, dispatcher)
    return Response(response.json, mimetype='application/json')


if __name__ == '__main__':
    run_simple('localhost', 4000, application)