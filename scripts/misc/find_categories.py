#! usr/bin/env python
# -*- coding : utf-8 -*-

import codecs
import os


def main():
	ann_tweets = []
	ann_file = '/Users/aggarwalpiush/github_repos/tweetnewsdetection/data/step_two/labelled/v2/tweets.txt'
	with codecs.open(ann_file, 'r', 'utf-8') as ann_obj:
		for line in ann_obj:
			ann_tweets.append(line.strip().rstrip().replace('\n','').replace('\r',''))
	print(ann_tweets[:10])

	cats = ['Ebola', 'harvey_hurricane', 'hiv_v2', 'iot', 'irma', 'nintendo', 'stockholm', 'swachhbharat_v2']
	cat_tweets = {}
	
	cat_file = '/Users/aggarwalpiush/github_repos/tweetnewsdetection/data/step_two/unlabelled/v2'
	for catg in cats:
		cat_tweets[catg] = []
		with codecs.open(os.path.join(cat_file, catg+'.tsv_results_f8.tsv'), 'r', 'utf-8') as cat_obj:
			for line in cat_obj:
				cat_tweets[catg].append(line.split('\t')[0].strip())


	cat_tweet_count = {}

	for catg in cats:
		cat_tweet_count[catg] = 0
		for tw in ann_tweets:
			if tw in cat_tweets[catg]:
				cat_tweet_count[catg] += 1

	print(cat_tweet_count)
	print(sum(cat_tweet_count.values()))



if __name__ == '__main__':
	main()



