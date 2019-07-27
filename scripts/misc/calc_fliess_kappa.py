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



def generate_fliess_table(input_file):
	fleiss_table = []
	with codecs.open(input_file, 'r', 'utf-8') as in_obj:
		for line in in_obj:
			line = line.split('\t')
			if int(line[0]) >= 3 or int(line[1]) >= 3:
				fleiss_table.append([int(line[0]), int(line[1]), int(line[2].rstrip('\r\n'))])
	return fleiss_table


def main():
	input_file = sys.argv[1]
	fliess_table = generate_fliess_table(input_file)
	print(len(fliess_table))
	print(fleiss_kappa(fliess_table, method='fleiss'))


if __name__ == '__main__':
	main()
