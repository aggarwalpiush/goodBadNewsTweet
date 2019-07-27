#! usr/bin/env python
# -*- coding : utf-8 -*- 

from __future__ import division
import sys
import codecs
from collections import Counter
from sklearn.metrics import cohen_kappa_score
from sklearn.metrics import matthews_corrcoef
import numpy as np



def generate_annotator_tweet(input_file):
	annotator = {}
	tweet = {}
	with codecs.open(input_file, 'r', 'utf-8') as in_obj:
		for i,line in enumerate(in_obj):
			if i == 0:
				continue
			line = line.split(',')
			line = [x.strip('\n\r') for x in line]
			if line[3] == 'query':
				continue
			if line[-1] and line[1] and line[3]:
				if line[-1] not in annotator.keys():
				 	annotator[line[-1]] = [[line[1], line[3]]]
				else:
				 	annotator[line[-1]].append([line[1], line[3]])
				if line[1] not in tweet.keys():
				 	tweet[line[1]] = [line[3]]
				else:
				 	tweet[line[1]].append(line[3])
	return annotator, tweet


def main():
	input_file = sys.argv[1]
	annotator, tweet = generate_annotator_tweet(input_file)
	annotator_kappa_score = {}
	for key ,value in annotator.items():
		majority_vote = []
		annotator_vote = []
		for ann_tweet,annotation in value:
			majority_vote.append(1 if max(Counter(tweet[ann_tweet]), key=lambda k: Counter(tweet[ann_tweet])[k])=='support' else 0)
			annotator_vote.append(1 if annotation=='support' else 0)
		if not np.isnan(cohen_kappa_score(majority_vote, annotator_vote)):
			annotator_kappa_score[key] = cohen_kappa_score(majority_vote, annotator_vote)
	inter_annotator_aggreement = np.average(np.array([float(x) for x in list(annotator_kappa_score.values())]))	
	Total_support = len([1  for key in tweet.keys() if max(Counter(tweet[key]), key=lambda k: Counter(tweet[key])[k])=='support'])
	Total_deny = len([1 for  key in tweet.keys() if max(Counter(tweet[key]), key=lambda k: Counter(tweet[key])[k])=='deny' ])
	print('Total annotators: %s' %(len(list(annotator.keys()))))
	print('Total tweets: %s' %(len(list(tweet.keys()))))
	print('Total support: %s' %Total_support)
	print('Total deny: %s' %Total_deny)
	print('Inter Annotator Agreement based on Majority voting : %s' %inter_annotator_aggreement)

if __name__ == '__main__':
	main()







