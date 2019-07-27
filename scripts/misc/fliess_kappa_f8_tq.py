#! usr/bin/env python
# -*- coding : utf-8 -*- 

from __future__ import division
import sys
import codecs
from collections import Counter
from sklearn.metrics import cohen_kappa_score
from sklearn.metrics import matthews_corrcoef
import numpy as np
from time import sleep
import json
from statsmodels.stats.inter_rater import fleiss_kappa
from get_tweet_text import TweetReader


def take_labels(filename):
	labels = []
	count = 100
	with codecs.open(filename, 'r', 'utf-8') as in_file_obj:
		for line in in_file_obj:
			count += 1
			if count < 200:
				flag = 2
				if line.split('\t')[-1].strip('\n\r') == 'b':
					flag = 0
				elif line.split('\t')[-1].strip('\n\r') == 'g':
					flag = 1
				labels.append(flag)
	return labels

def generate_fliess_table(input_table):
	fleiss_table = []
	annotators = len(input_table)
	for i in range(len(input_table[0])):
		subject_annotation = []
		for j in range(annotators):
			subject_annotation.append(input_table[j][i])
		cnt = Counter(subject_annotation)
		fleiss_table.append([cnt[0] if 0 in cnt.keys() else 0, cnt[1] if 1 in cnt.keys() else 0, cnt[2] if 2 in cnt.keys() else 0] )
		# print(fleiss_table[i])
		# sleep(0.5)
	return fleiss_table


def main():
	fliess_input_table = []
	response = ''
	while str(response.encode('utf-8')).strip().lower() not in ['p'] :
		response = input("type annotator filename or press p to proceed if no more files available: ")
		if response.strip() == 'p':
			break
		fliess_input_table.append(take_labels(response))
	fliess_table = generate_fliess_table(fliess_input_table)
	print(fleiss_kappa(fliess_table, method='fleiss'))

	# tweets_text = []
	# with codecs.open('f8_gb_test_hydrated.jsonl','r','utf-8') as in_obj:
	# 	for line in in_obj:
	# 		tweets_desc = json.loads(line)
	# 		if tweets_desc['full_text'] != '':
	# 			tweets_text.append(tweets_desc['full_text'].replace('\n', ' ').replace('|', ' ').replace('\t', ' ').replace('\r', ''))

	# with codecs.open('gb_f8_tq.tsv' ,'w', 'utf-8') as tq_out_obj:
	# 	tq_out_obj.write('GoodNews_or_BadNews_gold\tTWEET_ID\tTEXT\t_golden\tGoodNews_or_BadNews_gold_reason\n')
	# 	with codecs.open('test_questions_tweets.tsv' ,'r', 'utf-8') as ref_obj:
	# 		tr = TweetReader()
	# 		for i, line in enumerate(ref_obj):
	# 			flag = 'unsure'
	# 			if np.argmax(fliess_table[i]) == 0:
	# 				flag = 'bad'
	# 			elif np.argmax(fliess_table[i]) == 1:
	# 				flag = 'good'
	# 			if flag != 'unsure':
	# 				print(int(line.rstrip('\n\r')))
	# 				tq_out_obj.write(flag+'\t'+line.rstrip('\n\r')+'\t'+tweets_text[i]+'\tTRUE\t0.66 support\n')



		


if __name__ == '__main__':
	main()
