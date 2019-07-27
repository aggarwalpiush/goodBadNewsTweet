#! usr/bin/env python
# -*- coding : utf-8 -*-

import codecs
import sys
import numpy as np
from collections import Counter
from sklearn.metrics import cohen_kappa_score
from statsmodels.stats.inter_rater import fleiss_kappa
from time import sleep

def load_file(infile):
	subject_response = {}
	confident_sub_resp = {}
	with codecs.open(infile, 'r', 'utf-8') as in_obj:
		for i,line in enumerate(in_obj):
			if i == 0:
				continue
			line = line.rstrip('\r\n')
			line = line.split(',')
			# changes done on 02.05.2019 to compensate file invalid confidence scores
			#start
			current_confidence_score = line[2].replace('-','0')
			current_confidence_score = current_confidence_score.replace('\\','')
			current_confidence_score = current_confidence_score.replace('p','0')
			if not current_confidence_score.isdigit():
				current_confidence_score = 50
			#end
			if line[1] == 'deny':
				flag = 0
				conf_flag = [0, int(current_confidence_score)]
			elif line[1] == 'support':
				flag = 1
				conf_flag = [1, int(current_confidence_score)]
			else:
				print(line[1])
				flag = 2
			if line[0] in subject_response.keys():
				subject_response[line[0]].append(flag)
				confident_sub_resp[line[0]].append(conf_flag)
			else:
				subject_response[line[0]] = [flag]
				confident_sub_resp[line[0]] = [conf_flag]
	return subject_response, confident_sub_resp


def calc_majority_vote(subject_response):
	majority_vote = []
	for key,value in subject_response.items():
		if len(value) >= 3:
			val = np.max(value)
			majority_vote.append(val)
	return majority_vote

def calc_annotator_reponse(subject_response):
	max_length = max([len(val) for key,val in subject_response.items()])
	annotators = {}
	majority_vote = []
	annotators[0] = []
	annotators[1] = []
	annotators[2] = []
		
	for key,val in subject_response.items():
		if len(val) >= 3:
			val_maj = np.max(val)
			majority_vote.append(val_maj)
		for i in range(3):
			if i < len(val) and len(val) >= 3:
				annotators[i].append(val[i])
			else:
				print(len(val), key)
				#annotators[i].append(3)
	return annotators, majority_vote

def generate_fliess_table(subject_response):
	fliess_table = []
	for key,val in subject_response.items():
		cnt = Counter(val)
		# print(max([float(x) for k,x in cnt.items()]))
		# print(sum([float(x) for k,x in cnt.items()]))
		if max([float(x) for k,x in cnt.items()])/sum([float(x) for k,x in cnt.items()]) >= 0.0:
			cnt = Counter(val[:3])
			#print(cnt)
			if not 2 in cnt.keys():
				if sum([cnt[0] if 0 in cnt.keys() else 0, cnt[1] if 1 in cnt.keys() else 0]) == 3:
					fliess_table.append([cnt[0] if 0 in cnt.keys() else 0, cnt[1] if 1 in cnt.keys() else 0])
	return fliess_table



def main():
	response_file = sys.argv[1]
	best_three = bool(sys.argv[2])
	subject_response, confident_sub_resp = load_file(response_file)
	fliess_table = generate_fliess_table(subject_response)
	print(len(fliess_table))
	#majority_vote = calc_majority_vote(subject_response)
	annotators, majority_vote = calc_annotator_reponse(subject_response)
	cohen_kappa = []
	print(len(majority_vote))
	with codecs.open('annotation_scores.tsv', 'w', 'utf-8') as ann_obj:
		for keys in annotators.keys():
			temp_annotator_response = []
			temp_majority_vote = []
			for i in range(len(majority_vote)):
				if annotators[keys][i] != 3:
					temp_annotator_response.append(annotators[keys][i])
					if keys == 2:
						ann_obj.write(str(annotators[keys][i])+'\t')
					temp_majority_vote.append(majority_vote[i])
					if keys == 2:
						ann_obj.write(str(majority_vote[i])+'\n')
			print(len(temp_annotator_response))
			print(len(temp_majority_vote))
			print(cohen_kappa_score(temp_majority_vote, temp_annotator_response))
			cohen_kappa.append(cohen_kappa_score(temp_majority_vote, temp_annotator_response))
	print(fleiss_kappa(fliess_table, method='fleiss'))
	print(np.average(np.array(cohen_kappa)))


if __name__ == '__main__':
	main()





