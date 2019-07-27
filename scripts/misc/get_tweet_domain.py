#! usr/bin/env python
# -*- coding : utf-8 -*-

import codecs
import pymongo
from pymongo import MongoClient
import sys
import os
import glob
import stat
from shutil import copyfile, rmtree
from tqdm import tqdm

CLIENT = MongoClient('mongodb://127.0.0.1:27017')
DB = CLIENT.mttweetlabelsDB
COLLECTIONS = ['swachhbharat', 'swachhbharat_v2','hiv_v2',  'macerata', 'harvey_hurricane', 'iot','irma','nintendo','stockholm']
DONE= ['Ebola', 'agu17', 'hiv']


def get_tweet_domain(tweets, col):
	collec = DB[col]
	domain_tweets = []
	for tweetid in tqdm(tweets):
		tweet_text = collec.find({"id":int(tweetid)}).limit(1).count()
		if int(tweet_text) > 0:
			domain_tweets.append(tweetid)
	return domain_tweets


def main():
	tweets_file_path = sys.argv[1]

	tweets = []
	for f in glob.glob(os.path.join(tweets_file_path,'*.txt')):
		tweets.append(int(os.path.basename(f).strip().replace('.txt', '')))
	assert len(tweets) == 6853
	collection_dict = {}
	for col in tqdm(COLLECTIONS):
		collection_dict[col] = get_tweet_domain(tweets, col)
		dom_dir = os.path.join(os.path.dirname(tweets_file_path), col)
		if os.path.exists(dom_dir):
			rmtree(dom_dir, ignore_errors=False, onerror=None)
		os.mkdir(dom_dir)
		os.chmod(dom_dir, 0o777)
		for tw in collection_dict[col]:
			copyfile(os.path.join(tweets_file_path, str(tw)+'.txt'),  
				os.path.join(dom_dir, str(tw)+'.txt') )


if __name__ == '__main__':
	main()







