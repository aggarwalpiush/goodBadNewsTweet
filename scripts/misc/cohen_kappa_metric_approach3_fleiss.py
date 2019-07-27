#! usr/bin/env python
# -*- coding : utf-8 -*- 

from __future__ import division
import sys
import codecs
from collections import Counter
from sklearn.metrics import cohen_kappa_score
from sklearn.metrics import matthews_corrcoef
import numpy as np
from statsmodels.stats.inter_rater import fleiss_kappa


def generate_annotate_table(input_file):
	tweet_table = {}
	#category = {'support':[1,0,0],'deny':[0,1,0], 'query':[0,0,1]}
	category = {'support':[1,0],'deny':[0,1]}
	with codecs.open(input_file, 'r', 'utf-8') as in_obj:
		for i,line in enumerate(in_obj):
			annotation_dist = [0,0,0]
			if i == 0:
				continue
			line = line.split(',')
			line = [x.strip('\n\r') for x in line]
			if line[3] == 'query':
				continue
			if line[1] not in tweet_table.keys():
				tweet_table[line[1]] = category[line[3]]
			else:
				tweet_table[line[1]] = [sum(x) for x in zip(tweet_table[line[1]],category[line[3]])]
	table = []
	for row in list(tweet_table.values()):
		#if float(max(row))/float(sum(row)) > 0.67:
		if sum(row) > 1:
			table.append(row)
	#table = list(tweet_table.values())
	print(len(table))
	return table

def calculate_po(table):
	pi_arr = []
	for i in table:
		pi = 0
		n = sum(i)
		for j in i:
			pi += j*(j-1)
		pi = pi / (n * (n-1))
		pi_arr.append(pi)
	return np.average(np.array(pi_arr))

def calculate_pe(table):
	pe_array = []
	N = len(table)
	avg_n = np.average([sum(ann) for ann in table])
	inverse_total_ann = 1 / (N * avg_n)
	for j in np.array(table).transpose() :
		pj = sum(j)
		pj =  inverse_total_ann * pj
		pe_array.append(pj)
	pe_sq_sum = np.sum(np.array(pe_array)**2)
	return pe_sq_sum

def calculate_kappa(po,pe):
	k = (po - pe)/(1 - pe)
	return k




def main():
	input_file = sys.argv[1]
	table = generate_annotate_table(input_file)
	#print(table)
	table = 1.0 * np.asarray(table)
	print('pe:%s' %calculate_po(table))
	print('po:%s' %calculate_pe(table))
	print(calculate_kappa(calculate_po(table), calculate_pe(table)))


if __name__ == '__main__':
	main()
