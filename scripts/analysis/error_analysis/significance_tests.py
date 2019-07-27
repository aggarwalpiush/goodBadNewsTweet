#! usr/bin/env python
# -*- coding : utf-8 -*-

import codecs
from scipy import stats
import glob
import os
import numpy as np
from statsmodels.stats.contingency_tables import mcnemar


class FindSignificance(object):
	def __init__(self):
		pass


	def paired_t_test(self, metA, metB):
		return stats.ttest_ind(metA, metB)

	def mcnemar_test(self, metA, metB, gold):
		metAmetB_correct = 0
		metAmetB_incorrect = 0
		metA_correct_metB_not = 0
		metB_correct_metA_not = 0

		for i,lab in enumerate(gold):
			if lab == metA[i] and lab == metB[i]:
				metAmetB_correct += 1
			elif lab == metA[i] and not lab == metB[i]:
				metA_correct_metB_not += 1
			elif lab == metB[i] and not lab == metA[i]:
				metB_correct_metA_not += 1
			else:
				metAmetB_incorrect += 1
		table = [[metAmetB_correct, metA_correct_metB_not],[metB_correct_metA_not, metAmetB_incorrect]]

		return mcnemar(table, exact=False, correction=True)

def filetolist(file):
	results = []
	with codecs.open(file, 'r', 'utf-8') as infile_obj:
		for line in infile_obj:
			results.append(int(line.split('\t')[0]))
	print(results[:10])
	return results

def filetogold(file):
	results = []
	with codecs.open(file, 'r', 'utf-8') as infile_obj:
		for line in infile_obj:
			results.append(int(line.split('\t')[1].strip().rstrip().replace('\r\n','')))
	return results

def save_ttest_results(directory_path, ref_path):
	fs = FindSignificance()
	outfile = os.path.join(os.path.dirname(directory_path),os.path.basename(directory_path)+'_significant_terms_ttest.tsv')
	with codecs.open(outfile, 'w', 'utf-8') as out_obj:
		out_obj.write('%s\t%s\t%s\n' %('System', 't-stat', 'p-value'))
		for file in glob.glob(directory_path+'/*test_results.txt'):
			system_name = '_'.join(os.path.basename(file).split('_')[:-2])
			t_test = fs.paired_t_test(filetolist(ref_path), filetolist(file))
			out_obj.write('%s\t%.3f\t%.6f\n' %(system_name, t_test[0], t_test[1]))
		t_test = fs.paired_t_test(filetolist(ref_path), np.zeros(len(filetolist(file)), int))
		out_obj.write('%s\t%.3f\t%.6f\n' %('baseline_all_bad', t_test[0], t_test[1]))
	return None




def main():
	# file1 = '/Users/aggarwalpiush/github_repos/tweetnewsdetection/test_results_text_wreply/BERT_bert_text_wreply_test_results.txt'
	# file2 = '/Users/aggarwalpiush/github_repos/tweetnewsdetection/test_results_text/LSVC_emb_tfidf_text_test_results.txt'
	# fs = FindSignificance()
	# t_test = fs.paired_t_test(filetolist(file1), filetolist(file2 ))
	# for i in t_test:
	# 	print(i)
	# print(fs.mcnemar_test(filetolist(file1), filetolist(file2 ), filetogold(file1)))
	dir_path = '/Users/aggarwalpiush/github_repos/tweetnewsdetection/test_results_text'
	ref_path = '/Users/aggarwalpiush/github_repos/tweetnewsdetection/test_results_text/SVC_imp_text_test_results.txt'
	dt = '/Users/aggarwalpiush/github_repos/tweetnewsdetection/test_results_text/DT_tfidf_text_test_results.txt'
	LSVC = '/Users/aggarwalpiush/github_repos/tweetnewsdetection/test_results_text/LSVC_lexicon_text_test_results.txt'
	bert = '/Users/aggarwalpiush/github_repos/tweetnewsdetection/test_results_text/BERT_bert_text_test_results.txt'
	SVC = '/Users/aggarwalpiush/github_repos/tweetnewsdetection/test_results_text/SVC_characterstic_text_test_results.txt'
	# heath = '/Users/aggarwalpiush/github_repos/tweetnewsdetection/test_results_newsnotnews_health_text/SVC_emb_text_test_results.txt'
	# scitech = '/Users/aggarwalpiush/github_repos/tweetnewsdetection/test_results_newsnotnews_scitech_text/SVC_emb_scitech_text_test_results.txt'
	# geoenv = '/Users/aggarwalpiush/github_repos/tweetnewsdetection/test_results_newsnotnews_geo_env_text/SVC_emb_text_test_results.txt'
	# terrorism = '/Users/aggarwalpiush/github_repos/tweetnewsdetection/test_results_newsnotnews_geo_env_text/SVC_emb_text_test_results.txt'
	fs = FindSignificance()
	print(fs.paired_t_test(filetolist(LSVC),filetolist(SVC)))
	#save_ttest_results(dir_path, ref_path)

if __name__ == '__main__':
	main()

	

