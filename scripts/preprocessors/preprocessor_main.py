	#! usr/bin/env python
# * -- coding: utf-8 --*

import codecs
import os
import glob
from scripts.preprocessors.preprocessor_arc import Arc_preprocessor
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from config.feature_config import TEST_SIZE
from config.feature_config import DATA_PATH, LABELS_PATH, TEXT_TYPE
from config.feature_config import DATA_DOMAIN_PATH
from tqdm import tqdm
import re


class Corpus_Preprocessor(object):
	def __init__(self, test_domain=False):
		self.data_path = DATA_PATH
		self.label_path = LABELS_PATH
		self.test_domain = test_domain
		self.domain_path = DATA_DOMAIN_PATH
		if not test_domain:
			self.save_path = os.path.join(os.path.dirname(DATA_PATH), TEXT_TYPE + '_exp_data')
		else:
			self.save_path = os.path.join(DATA_DOMAIN_PATH, TEXT_TYPE + '_exp_data')
		if not os.path.exists(self.save_path):
			os.makedirs(self.save_path)

	def _generate_corpus(self):
		corpus = []
		labels = []
		tweet_list = []
		all_labels = {}
		with codecs.open(self.label_path, 'r', 'utf-8') as label_obj:
			for label_line in label_obj:
				all_labels[int(label_line.split('\t')[0])] = int(label_line.split('\t')[1].strip().rstrip('\r\n').
					replace('\n', '').replace('\r', ''))

		for textfile in glob.glob(self.data_path+'/[0-9]*.txt'):
			file_text = []
			tweetid = int(os.path.basename(textfile).replace('.txt', ''))
			if int(tweetid) in all_labels.keys():
				labels.append(all_labels[tweetid])
				tweet_list.append(int(tweetid))
			with codecs.open(textfile, 'r', 'utf-8') as txt_obj:
				for textline in txt_obj:
					file_text.append(textline.strip().lower().rstrip('\r\n').replace('\n', '').replace('\r', ''))
			corpus.append(' '.join(file_text))
			assert len(labels) == len(corpus)
		return corpus, labels, tweet_list

	def remove_num(self, str):
	    string_no_numbers = re.sub("\d+", " ", str)
	    return string_no_numbers

	def _apply_ark_tokenization(self):
		tokenized_corpus = []
		arc_obj = Arc_preprocessor()
		corpus, labels, tweet_list = self._generate_corpus()
		for i, tweet_text in tqdm(enumerate(corpus)):
			tokenized_tweet_text = [x.strip('\n\r') for x in arc_obj.tokenizeRawTweetText(tweet_text)]
			tokenized_tweet_text = ['url' if 'http' in str(x) else x for x in tokenized_tweet_text]
			tokenized_tweet_text = [self.remove_num(x) for x in tokenized_tweet_text]
			tokenized_tweet_text = [x for x in tokenized_tweet_text if not x == '']
			tokenized_corpus.append(str(tweet_list[i]) + '\t' +' '.join(tokenized_tweet_text))
		return tokenized_corpus, labels, tweet_list



	def _load_file(self, input_path):
		input_tweets = []
		for fil in glob.glob(str(input_path)+'/**/*.txt', recursive=True):
			input_tweets.append(int(os.path.basename(fil).rstrip('\r\n').replace('\n','').replace('.txt','')))
		return input_tweets


	def train_test_domain_split(self, corpus, labels, tweet_list):
		domain_tweets = self._load_file(self.domain_path)
		set_domain_tweets = set(domain_tweets)
		X_train = []
		X_test = []
		y_train = []
		y_test = []
		for i,tw in enumerate(tweet_list):
			if not tw in set_domain_tweets:
				X_train.append(corpus[i])
				y_train.append(labels[i])
			else:
				X_test.append(corpus[i])
				y_test.append(labels[i])
		return X_train, X_test, y_train, y_test


	def split_dataset(self):
		corpus, labels, tweet_list = self._apply_ark_tokenization()
		if not self.test_domain:
			X_train, X_test, y_train, y_test = train_test_split(corpus, labels, stratify=labels, 
				test_size=TEST_SIZE, random_state=42)
		else:
			X_train, X_test, y_train, y_test = self.train_test_domain_split(corpus, labels, tweet_list)
		return X_train, X_test, y_train, y_test


	def save_train_test_files(self):
		X_train, X_test, y_train, y_test = self.split_dataset()
		with codecs.open(os.path.join(self.save_path, 'train_data.tsv'), 'w', 'utf-8') as train_obj:
			for i, txt in enumerate(X_train):
				train_obj.write('%s\t%s\n' %(txt, y_train[i]))
		with codecs.open(os.path.join(self.save_path,'test_data.tsv'), 'w', 'utf-8') as test_obj:
			for i, txt in enumerate(X_test):
				test_obj.write('%s\t%s\n' %(txt, y_test[i]))	




def main():
	cp = Corpus_Preprocessor(test_domain=True)
	cp.save_train_test_files()

    #print(pv.get_tweet_pos)


if __name__ == '__main__':
    main()










