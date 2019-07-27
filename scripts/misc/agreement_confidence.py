#! usr/bin/env python
# -*- coding : utf-8 -*-

'''
This class is designed to get inter annotators agreement with each subject is annotated by at least 3 annotator
along with confidence percentage for each annotation. 

class input:

2 dimensional array with subject_id,support,confidence as tuple 

'''


from sklearn.metrics import cohen_kappa_score
from statsmodels.stats.inter_rater import fleiss_kappa
import numpy as np
import sys
from collections import Counter


class ConfidentIAA(object):
	def __init__(self, data_object, best_n):
		self.data_object = data_object
		self.best_n = best_n


	def get_annotators(self):
		annotators = {}
		for i in self.data_object:
			if i[0] in annotators.keys():
				annotators[i[0]].append([i[1:]])
			else:
				annotators[i[0]] = [[i[1:]]]
		return annotators

	def get_best_annotators(self):
		annotators = {}
		temp_annotators = self.get_annotators()
		for keys, value in temp_annotators.items():
			annotators[keys] = []
			confidence = []
			for i in value:
				confidence.append(i[1])
			for i in np.argsort(np.array(confidence))[-self.best_n:]:
				annotators[keys].append(value[i][0])
		return annotators

	def get_fliess_table(self):
		fliess_table = []
		for key,val in self.get_best_annotators().items():
			cnt = Counter(val)
			if sum([cnt[0] if 0 in cnt.keys() else 0, cnt[1] if 1 in cnt.keys() else 0]) == self.best_n:
				fliess_table.append([cnt[0] if 0 in cnt.keys() else 0, cnt[1] if 1 in cnt.keys() else 0])
		table = 1.0 * np.asarray(fliess_table)   #avoid integer division
		n_sub, n_cat =  fliess_table.shape
		n_total = table.sum()
		n_rater = table.sum(1)
		n_rat = n_rater.max()
	    #assume fully ranked
		assert n_total == n_sub * n_rat
		return fliess_table

	def get_kappas(self):
		majority_vote = []
		individual_vote = []
		cohen_kappas= []
		for key,val in self.get_best_annotators().items():
			cnt = Counter(val)
			majority_vote.append(max(cnt, key=lambda k: cnt[k]))
			individual_vote.append(val)
		individual_vote = np.asarray(individual_vote)
		for i in len(self.best_n):
			cohen_kappas.append(cohen_kappa_score(majority_vote,individual_vote[:,i]))

		avg_cohen_kappas = np.average(np.asarray(cohen_kappas))
		fleiss_kappa_score = fleiss_kappa(get_fliess_table(), method='fleiss')
		scores = {}
		scores['Fleiss Kappa Score'] = fleiss_kappa_score
		scores['Average Cohen Kappa'] = avg_cohen_kappas
		for i in cohen_kappas:
			scores['Cohen kappa Annotator '+str(i)] = cohen_kappas[i]
		return scores
		










