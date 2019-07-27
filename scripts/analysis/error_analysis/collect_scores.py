#! usr/bin/env python
# * -- codng : utf-8 -- *

import codecs
import os
import numpy as np
import glob
import sys
from tqdm import tqdm
from textblob import TextBlob as tb
from sklearn.metrics import classification_report
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.metrics import roc_auc_score
from config.feature_config import TEST_OUTPUT_PATH, TEXT_TYPE, DATA_PATH


def load_file(input_file):
        in_data = []
        with codecs.open(input_file, 'r', 'utf-8') as in_file_obj:
                for line in in_file_obj:
                        in_data.append(line)

        return in_data


def save_scores(y_pred, y_test, model_name):
	target_names = ['Bad News', 'Good News']
	with codecs.open(os.path.join(TEST_OUTPUT_PATH, model_name + '_' +  TEXT_TYPE
		+  '_model_evaluation.txt'), 'w', 'utf-8') as scr_obj:
		scr_obj.write('\nPrecision score: {:3f}\n'.format(precision_score(y_test, y_pred, average='macro') ))
		scr_obj.write('Recall score: {:3f}\n'.format(recall_score(y_test, y_pred, average='macro') ))
		scr_obj.write('F1 score: {:3f}\n'.format(f1_score(y_test, y_pred, average='macro')))
		scr_obj.write('AUC score: {:3f}\n'.format(roc_auc_score(y_test, y_pred)))
		scr_obj.write('Confusion Metric : %s\n' %(confusion_matrix(y_test, y_pred)))
		scr_obj.write(classification_report(y_test, y_pred, target_names=target_names))


def process_bert_results():
	for f in os.listdir(TEST_OUTPUT_PATH):
		with codecs.open(os.path.join(TEST_OUTPUT_PATH,f), 'r', 'utf-8') as infile_obj:
			file_parts = f.split('_')
			if file_parts[0] == 'BERT' and file_parts[-1] == 'probability.txt':
				y_pred = []
				y_test = []
				pred_results = load_file(os.path.join(TEST_OUTPUT_PATH,f))
				for x in pred_results:
					y_pred.append(0 if float(x.split('\t')[0]) > float(x.split('\t')[1]) else 1)

				test_results = load_file(os.path.join(os.path.dirname(DATA_PATH), TEXT_TYPE + '_exp_data', 
					'test_data.tsv'))
				for x in test_results:
					y_test.append(int(x.split('\t')[-1].strip('\r\n')))

				print(y_test[:10])
				print(y_pred[:10])

				save_scores(y_pred, y_test, 'BERT_bert')
				# please comment below code not necessary
				### comment start
				with codecs.open(os.path.join(TEST_OUTPUT_PATH, 'BERT_bert_' + TEXT_TYPE + 
			'_fn_fps.txt'), 'w', 'utf-8') as fn_fn_obj:
					for i in range(len(y_pred)):
						if not y_pred[i] == y_test[i]:
							if y_test[i] == 0:
								fn_fn_obj.write("False positive : %s\n" %(i+1) )
							elif y_test[i] == 1:
								fn_fn_obj.write("False negative : %s\n" %(i+1))

				### comment end
				out_data = np.append(np.transpose([np.array(y_pred)]), np.transpose([y_test]), axis=1)
				np.savetxt(os.path.join(TEST_OUTPUT_PATH, 'BERT_bert_' + TEXT_TYPE + 
			'_test_results.txt'), out_data, fmt='%s	%s', delimiter='\t')

def collect_f1():
	score_card_path = os.path.join(os.path.dirname(DATA_PATH), TEXT_TYPE + '_scorecards')
	if not os.path.exists(score_card_path):
		os.makedirs(score_card_path)
	with codecs.open(os.path.join(score_card_path, 'f1_scorecard.tsv'), 'w', 'utf-8') as scorecard_obj:
		for f in glob.glob(TEST_OUTPUT_PATH+'/*_model_evaluation.txt'):
			with codecs.open(f, 'r', 'utf-8') as infile_obj:
				basename = os.path.basename(f)
				for line in infile_obj:
					line = line.split(':')
					if line[0] == 'F1 score':
						scorecard_obj.write(basename.split('_')[0]+'\t'+' '.join(basename.split('_')[1:-2]) + '\t' 
							+ line[1].strip().rstrip('\n\r') + '\n')


def majority_voting(top_n, y_true):
	print(len(y_true))
	if not top_n > 2 and not top_n % 2 == 0:
		assert('number of votes should be greater than 2 and odd')
		sys.exit(-1)
	scores = []
	score_card_path = os.path.join(os.path.dirname(DATA_PATH), TEXT_TYPE + '_scorecards')
	with codecs.open(os.path.join(score_card_path, 'f1_scorecard.tsv'), 'r', 'utf-8') as scorecard_obj:
		for line in scorecard_obj:
			line = line.strip().rstrip('\r\n')
			scores.append(line.split('\t'))
	sorted_score = sorted(scores,key=lambda x: x[2], reverse = True)
	majority_voting = []
	instance_votes = []
	system_used = []
	for i in range(top_n):
		vote_file = os.path.join(TEST_OUTPUT_PATH,'_'.join([sorted_score[i][0],'_'.join(sorted_score[i][1].split(' ')),
		 'test_results.txt']))
		system_used.append('_'.join([sorted_score[i][0],'_'.join(sorted_score[i][1].split(' '))]))
		vote = []
		print(vote_file)
		with codecs.open(vote_file, 'r', 'utf-8') as vf_obj:
			for line in vf_obj:
				vote.append(int(line.split('\t')[0].strip().rstrip('\r\n')))
		print(len(vote))
		instance_votes.append(vote)
	instance_votes_sum = np.sum(instance_votes, axis=0)
	for each_vote in instance_votes_sum:
		if each_vote > float(top_n/2):
			majority_voting.append(1)
		else:
			majority_voting.append(0)
	print(len(majority_voting))
	majority_vote_path = os.path.join(os.path.dirname(DATA_PATH), TEXT_TYPE + 'majority_vote')
	if not os.path.exists(majority_vote_path):
		os.makedirs(majority_vote_path)
	out_data = np.append(np.transpose([np.array(majority_voting)]), np.transpose([y_true]), axis=1)
	np.savetxt(os.path.join(majority_vote_path, '_'.join(system_used) + 'test_results.txt'), out_data,
	 fmt='%s	%s', delimiter='\t')
	save_scores(majority_voting, y_true, '_'.join(system_used) + '_majority_voting_' + str(top_n))









def compare_sentiment_analysis(X_test, y_test):
	y_pred = []
	for i, instance in tqdm(enumerate(X_test)):
		senti = tb(instance).sentiment
		if senti[0] > 0:
			y_pred.append(1)
		else:
			y_pred.append(0)
	compare_sentiment_path = os.path.join(os.path.dirname(DATA_PATH), TEXT_TYPE + 'compare_sentiment')
	if not os.path.exists(compare_sentiment_path):
		os.makedirs(compare_sentiment_path)
	out_data = np.append(np.transpose([np.array(y_pred)]), np.transpose([y_test]), axis=1)
	np.savetxt(os.path.join(compare_sentiment_path, 'textblob_compare_sentiment_' + 'test_results.txt'), out_data,
	 fmt='%s	%s', delimiter='\t')
	save_scores(y_pred, y_test, 'textblob_compare_sentiment')



def compare_sentiment_analysis_80_boundary(X_test, y_test):
	y_pred = []
	y_test_boundary = []
	for i, instance in tqdm(enumerate(X_test)):
		senti = tb(instance).sentiment
		if senti[0] > 0 and senti[1] < 0.5:
			y_pred.append(1)
			y_test_boundary.append(y_test[i])
		elif senti[0] < 0  and senti[1] < 0.5:
			y_pred.append(0)
			y_test_boundary.append(y_test[i])
	print('total instances unattended %s out of %s' %((len(y_test)-len(y_test_boundary)), len(y_test)))
	compare_sentiment_path = os.path.join(os.path.dirname(DATA_PATH), TEXT_TYPE + 'compare_sentiment_80_boundary')
	if not os.path.exists(compare_sentiment_path):
		os.makedirs(compare_sentiment_path)
	out_data = np.append(np.transpose([np.array(y_pred)]), np.transpose([y_test_boundary]), axis=1)
	np.savetxt(os.path.join(compare_sentiment_path, 'textblob_compare_sentiment_80_boundary' + 'test_results.txt'), out_data,
	 fmt='%s	%s', delimiter='\t')
	save_scores(y_pred, y_test_boundary, 'textblob_compare_sentiment_80_boundary')




def score_layout():
	features = []
	instances = []
	ml_algos = []
	score_layout = {}
	score_card = os.path.join(os.path.dirname(DATA_PATH), TEXT_TYPE + '_scorecards', 'f1_scorecard.tsv')
	with codecs.open(score_card, 'r', 'utf-8') as scorecard_obj:
		for line in scorecard_obj:
			line = line.rstrip('\r\n')
			features.append(line.split('\t')[1].strip().rstrip('\r\n'))
			ml_algos.append(line.split('\t')[0].strip().rstrip('\r\n'))
			instances.append(line.strip().rstrip('\r\n').split('\t'))

	features = list(set(features))
	ml_algos = list(set(ml_algos))

	for mla in ml_algos:
		temp_list = [str(None)]*len(features)
		for inst in instances:
			if inst[0] == mla:
				temp_list[features.index(inst[1])] = inst[2]
		score_layout[mla] = temp_list

	# save file

	score_card_path = os.path.join(os.path.dirname(DATA_PATH), TEXT_TYPE + '_scorecards')
	if not os.path.exists(score_card_path):
		os.makedirs(score_card_path)

	with codecs.open(os.path.join(score_card_path, 'layout_f1_scorecard.tsv'), 'w', 'utf-8') as scorecard_obj:
		scorecard_obj.write('\t'+ '\t'.join(features) + '\n')
		for key, values in score_layout.items():
			scorecard_obj.write(key+'\t')
			print(values)
			scorecard_obj.write('\t'.join(values)+'\n')








def heatmap_correlation(metric):
	pass


def heatmap_correlation(model):
	pass



def top_pos_tags(top_n):
	pass



def main():
	#process_bert_results()
	#collect_f1()
	# y_test = []
	# X_test = []
	# with codecs.open(os.path.join(os.path.dirname(DATA_PATH), TEXT_TYPE + '_exp_data',
	#  'test_data.tsv'), 'r', 'utf-8') as test_obj:
	# 	for line in test_obj:
	# 		y_test.append(int(line.split('\t')[-1].strip().rstrip('\r\n')))
	# 		X_test.append(line.split('\t')[1].strip().rstrip('\r\n'))
	# compare_sentiment_analysis(X_test, y_test)

	# majority_voting(3, y_true)
	score_layout()


if __name__ == '__main__':
	main()




