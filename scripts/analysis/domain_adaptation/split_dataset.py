#! usr/bin/env python
# -*- coding : utf-8 -*-

import codecs
import glob
import os
import sys
from config.feature_config import DATA_PATH, DATA_DOMAIN_PATH, LABELS_PATH

def load_file(input_path):
	input_tweets = []
	for fil in glob.glob(str(input_path)+'/**/*.txt', recursive=True):
		input_tweets.append(int(os.path.basename(fil).rstrip('\r\n').replace('\n','').replace('.txt','')))
	return input_tweets


def load_labels(label_path):
	labels = {}
	with codecs.open(label_path, 'r', 'utf-8') as label_obj:
		for line in label_obj:
			line = line.split('\t')
			labels[int(line[0])] = int(line[1].strip().rstrip('\r\n').replace('\n', ''))
	return labels


def save_split_trainingset(total_tweets, domain_tweets, labels):
	set_domain_tweets = set(domain_tweets)
	X_train = []
	X_test = []
	y_train = []
	y_test = []
	split_path = os.path.join(DATA_DOMAIN_PATH, 'split_dataset')
	if not os.path.exists(split_path):
		os.mkdir(split_path)
		os.chmod(split_path, 0o777)
	for tw in total_tweets:
		if not tw in set_domain_tweets:
			with codecs.open(os.path.join(DATA_PATH,str(tw)+'.txt'),
				'r', 'utf-8') as tw_obj:
				X_train.append(' '.join([txt for txt in tw_obj]))
			y_train.append(labels[tw])
		else:
			with codecs.open(os.path.join(DATA_PATH,str(tw)+'.txt'), 'r', 'utf-8') as tw_obj:
				X_test.append(' '.join([txt for txt in tw_obj]))
			y_test.append(labels[tw])

	with codecs.open(os.path.join(split_path, 'train_data.tsv'), 'w', 'utf-8') as train_obj:
		for i, txt in enumerate(X_train):
			train_obj.write('%s\t%s\n' %(txt, y_train[i]))
	with codecs.open(os.path.join(split_path,'test_data.tsv'), 'w', 'utf-8') as test_obj:
		for i, txt in enumerate(X_test):
			test_obj.write('%s\t%s\n' %(txt, y_test[i]))	

def main():
	save_split_trainingset(load_file(DATA_PATH), load_file(DATA_DOMAIN_PATH), load_labels(LABELS_PATH))	


if __name__ == '__main__':
	main()

