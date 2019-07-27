#!usr/bin/env python
# -*- coding : utf-8 -*-


import codecs
import os
import json
import sys
import numpy as np, pandas as pd 
import zipfile

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from scipy import stats
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt 
import seaborn as sns
#%matplotlib inline


def load_bert_embeddings(json_input):
	bert_output = pd.read_json(json_input, lines = True)
	with codecs.open(json_input, "r", 'utf-8') as read_file:
		for line in read_file:
			data = json.loads(line)
			print(data['features'][1]['layers'][0]['values'])

def convert_embedding_2D(json_input, position_dict, fp_fn_position_dict):
	bert_output = pd.read_json(json_input, lines = True)
	emb_2d = {}
	emb12 = {}
	nrows = 100
	#nrows = bert_output.shape[0]  
	texts = []
	for row in range(nrows):
		if not row+1 in position_dict.keys():
			continue

		# Get the BERT embeddings for the current line in the data file
		features = pd.DataFrame(bert_output.loc[row,"features"]) 

		span = range(1,len(features)-1)

		# Make a list with the text of each token, to be used in the plots

		for j in span:
			word_array = []
		    #token = features.loc[j,'token']
		    #texts.append(token)
			word_array.append(np.array(features.loc[j,'layers'][0]['values']))
		emb12[row] = np.mean(np.array(word_array), axis=0)
		texts.append(str(row+1))


	X1 = np.array(list(emb12.values()))
	print("Shape of embedding matrix: ", X1.shape)

	# Use PCA to reduce dimensions to a number that's manageable for t-SNE
	pca = PCA(n_components = 50, random_state = 7)
	X1 = pca.fit_transform(X1)
	print("Shape after PCA: ", X1.shape)

	# Reduce dimensionality to 2 with t-SNE.
	# Perplexity is roughly the number of close neighbors you expect a
	# point to have. Our data is sparse, so we chose a small value, 10.
	# The KL divergence objective is non-convex, so the result is different
	# depending on the seed used.
	tsne = TSNE(n_components = 2, perplexity = 10, random_state = 6, 
	            learning_rate = 100, n_iter = 1500)
	X1 = tsne.fit_transform(X1)
	print("Shape after t-SNE: ", X1.shape)

	# Recording the position of the tokens, to be used in the plot
	position = np.array(list(range(1,nrows+1))) 
	position = position.reshape(-1,1)
	print(position.shape)
	print(np.array(texts).reshape(-1,1).shape)
	labels = []
	fp=0
	fn=0
	tp=0
	tn=0
	for i in range(100):
		if i+1 in fp_fn_position_dict.keys() and fp_fn_position_dict[i+1] == 'False negative':
			labels.append('False Negative')
			fn += 1
		elif i+1 in fp_fn_position_dict.keys() and fp_fn_position_dict[i+1] == 'False positive':
			labels.append('False Positive')
			fp += 1
		elif int(position_dict[i+1]) == 0:
			labels.append('Bad News(True Negative)')
			tn += 1
		elif int(position_dict[i+1]) == 1:
			labels.append('Good News(True Positive)')
			tp += 1
	# Pie chart
	labels2 = ['False Negative', 'False Positive', 'True Negative', 'True Positive']
	sizes = [fn, fp, tn, tp]
	# only "explode" the 2nd slice (i.e. 'Hogs')
	explode = (0.2, 0.1, 0, 0)
	colour = ['blue', 'red', 'orange', 'green' ]  
	fig1, ax1 = plt.subplots()
	ax1.pie(sizes, explode=explode, labels=labels2, autopct='%1.1f%%',
	        shadow=True, startangle=90, colors = colour)
	# Equal aspect ratio ensures that pie is drawn as a circle
	ax1.axis('equal')  
	plt.tight_layout()
	plt.savefig('/Users/aggarwalpiush/github_repos/tweetnewsdetection/data/bert_embeddings/text_test_data/t-SNE_bert_text_100_pie.png')
	# for i in range(len(position_dict.keys())):
		#labels.append(position_dict[i+1])

	X = pd.DataFrame(np.concatenate([X1, position, np.array(texts).reshape(-1,1), np.array(labels).reshape(-1,1)], axis = 1), 
	                 columns = ["x1", "y1", "position", "texts", "labels"])
	X = X.astype({"x1": float, "y1": float, "position": float, "texts": object, "labels":object})

	# Remove a few outliers based on zscore
	X = X[(np.abs(stats.zscore(X[["x1", "y1"]])) < 3).all(axis=1)]
	# emb_2d[row] = X
	# for row in range(nrows):
	# 	X = emb_2d[row]

	# Plot for layer -1
	plt.figure(figsize = (15,12))
	p1 = sns.scatterplot(x = X["x1"], y = X["y1"], s=100,  style=X["labels"], hue = X["labels"])
	p1.set_title('BERT Classifier on t-SNE distribution of Tweets')
	#red_patch = mpatches.Patch(color='red', label='False Postive')
	#plt.legend(handles=[red_patch])

#Label each datapoint with the word it corresponds to
	# for line in X.index:
	# 	text = X.loc[line,"texts"]
	# 	if int(text) in fp_fn_position_dict.keys() and fp_fn_position_dict[int(text)] == 'False negative':
	# 		text = 'FN'
	# 		p1.text(X.loc[line,"x1"]+0.1, X.loc[line,"y1"], text, horizontalalignment='left', 
	# 	            size='large', color='blue', weight='semibold')
	# 	elif int(text) in fp_fn_position_dict.keys() and fp_fn_position_dict[int(text)] == 'False positive':
	# 		text = 'FP'
	# 		p1.text(X.loc[line,"x1"]+0.1, X.loc[line,"y1"], text, horizontalalignment='left', 
	# 	            size='large', color='red', weight='semibold')
		# else:
		# 	text = 1
		# 	p1.text(X.loc[line,"x1"]+0.1, X.loc[line,"y1"], text, horizontalalignment='left', 
		#             size='small', color='red', weight='semibold')
	plt.savefig('/Users/aggarwalpiush/github_repos/tweetnewsdetection/data/bert_embeddings/text_test_data/t-SNE_bert_text_100.png')


def convert_embedding_2D_reply(json_input1, json_input2, fp_fn_position_dict1, fp_fn_position_dict2, position_dict):
	bert_output = pd.read_json(json_input1, lines = True)
	emb_2d = {}
	emb12 = {}
	#nrows1 = bert_output.shape[0]
	nrows1 = 100  
	texts = []
	for row in range(nrows1):
		#if row+1 not in fp_fn_position_dict1.keys():
		#	continue

		# Get the BERT embeddings for the current line in the data file
		features = pd.DataFrame(bert_output.loc[row,"features"]) 

		span = range(1,len(features)-1)

		# Make a list with the text of each token, to be used in the plots

		for j in span:
			word_array = []
		    #token = features.loc[j,'token']
		    #texts.append(token)
			word_array.append(np.array(features.loc[j,'layers'][0]['values']))
		emb12[row] = np.mean(np.array(word_array), axis=0)
		texts.append(str(row+1))
	bert_output = pd.read_json(json_input2, lines = True)
	nrows2 = 100
	#nrows2 =  bert_output.shape[0]  
	for row in range(nrows2):
		#if row+1 not in fp_fn_position_dict2.keys():
		#	continue

		# Get the BERT embeddings for the current line in the data file
		features = pd.DataFrame(bert_output.loc[row,"features"]) 

		span = range(1,len(features)-1)

		# Make a list with the text of each token, to be used in the plots

		for j in span:
			word_array = []
		    #token = features.loc[j,'token']
		    #texts.append(token)
			word_array.append(np.array(features.loc[j,'layers'][0]['values']))
		emb12[row + nrows1] = np.mean(np.array(word_array), axis=0)
		texts.append(str(nrows1+row+1))


	X1 = np.array(list(emb12.values()))
	print("Shape of embedding matrix: ", X1.shape)

	# Use PCA to reduce dimensions to a number that's manageable for t-SNE
	pca = PCA(n_components = 50, random_state = 7)
	X1 = pca.fit_transform(X1)
	print("Shape after PCA: ", X1.shape)

	# Reduce dimensionality to 2 with t-SNE.
	# Perplexity is roughly the number of close neighbors you expect a
	# point to have. Our data is sparse, so we chose a small value, 10.
	# The KL divergence objective is non-convex, so the result is different
	# depending on the seed used.
	tsne = TSNE(n_components = 2, perplexity = 10, random_state = 6, 
	            learning_rate = 100, n_iter = 1500)
	X1 = tsne.fit_transform(X1)
	print("Shape after t-SNE: ", X1.shape)

	# Recording the position of the tokens, to be used in the plot
	#position = np.array(list(range(1,len(fp_fn_position_dict1.keys())+len(fp_fn_position_dict2.keys())+1)))
	position = np.array(list(range(1,nrows1+nrows2+1)))  
	position = position.reshape(-1,1)
	print(position.shape)
	print(np.array(texts).reshape(-1,1).shape)
	labels = []
	#for i in range(len(position_dict.keys())):
	for i in range(100):
		labels.append(position_dict[i+1])
	#for i in range(len(position_dict.keys())):
	for i in range(100):
		labels.append(position_dict[i+1])


	# adding pie chart and change label names to make t-SNE more meaningful
	fp=0
	fn=0
	tp=0
	tn=0
	for i in range(100):
		if i+1 in fp_fn_position_dict1.keys():
			labels.append('Text Corpus Type-I,II Error')
			fn += 1
		elif i+1 in fp_fn_position_dict.keys() and fp_fn_position_dict[i+1] == 'False positive':
			labels.append('False Positive')
			fp += 1
		elif int(position_dict[i+1]) == 0:
			labels.append('Bad News(True Negative)')
			tn += 1
		elif int(position_dict[i+1]) == 1:
			labels.append('Good News(True Positive)')
			tp += 1
	# Pie chart
	labels2 = ['False Negative', 'False Positive', 'True Negative', 'True Positive']
	sizes = [fn, fp, tn, tp]
	# only "explode" the 2nd slice (i.e. 'Hogs')
	explode = (0.2, 0.1, 0, 0)
	colour = ['blue', 'red', 'orange', 'green' ]  
	fig1, ax1 = plt.subplots()
	ax1.pie(sizes, explode=explode, labels=labels2, autopct='%1.1f%%',
	        shadow=True, startangle=90, colors = colour)
	# Equal aspect ratio ensures that pie is drawn as a circle
	ax1.axis('equal')  
	plt.tight_layout()
	plt.savefig('/Users/aggarwalpiush/github_repos/tweetnewsdetection/data/bert_embeddings/text_test_data/t-SNE_bert_text_100_pie.png')


	X = pd.DataFrame(np.concatenate([X1, position, np.array(texts).reshape(-1,1), np.array(labels).reshape(-1,1)], axis = 1), 
	                 columns = ["x1", "y1", "position", "texts", "labels"])
	X = X.astype({"x1": float, "y1": float, "position": float, "texts": object, "labels":int})

	# Remove a few outliers based on zscore
	X = X[(np.abs(stats.zscore(X[["x1", "y1"]])) < 3).all(axis=1)]
	# emb_2d[row] = X
	# for row in range(nrows):
	# 	X = emb_2d[row]

	# Plot for layer -1
	plt.figure(figsize = (15,12))
	p1 = sns.scatterplot(x = X["x1"], y = X["y1"],  s=100, hue = X["labels"], style=X["labels"])#, palette = "coolwarm")
	p1.set_title(os.path.basename('BERT Classifier on t-SNE on distribution of Tweets (tweet vs tweet_with replies)'))

#Label each datapoint with the word it corresponds to
	count_tot = 0
	count1 = 0
	count2 = 0
	for line in X.index:
		text = X.loc[line,"texts"]
		# if (int(text) in fp_fn_position_dict1.keys() and  int(text) not in fp_fn_position_dict2.keys()) or (int(text)-nrows1 not in fp_fn_position_dict1.keys() and int(text)-nrows1 in fp_fn_position_dict2.keys()):
		# 	print(text)
		# 	print(int(text)-nrows1)
		# 	count_tot += 1
		if int(text) <= nrows1:
			count1 += 1
			if int(text) in fp_fn_position_dict1.keys() and fp_fn_position_dict1[int(text)] == 'False negative':
				text = 'FN'
				p1.text(X.loc[line,"x1"]+0.2, X.loc[line,"y1"], text, horizontalalignment='left', 
			            size='large', color='blue', weight='semibold')
			elif int(text) in fp_fn_position_dict1.keys() and fp_fn_position_dict1[int(text)] == 'False positive':
				text = 'FP'
				p1.text(X.loc[line,"x1"]+0.2, X.loc[line,"y1"], text, horizontalalignment='left', 
			            size='large', color='black', weight='semibold')
		else:
			count2 += 1
			if int(text)-nrows1 in fp_fn_position_dict2.keys() and fp_fn_position_dict2[int(text)-nrows1] == 'False negative':
				text = 'FN'
				p1.text(X.loc[line,"x1"]+0.2, X.loc[line,"y1"], text, horizontalalignment='left', 
			            size='large', color='red', weight='bold')
			elif int(text)-nrows1 in fp_fn_position_dict2.keys() and fp_fn_position_dict2[int(text)-nrows1] == 'False positive':
				text = 'FP'
				p1.text(X.loc[line,"x1"]+0.2, X.loc[line,"y1"], text, horizontalalignment='left', 
			            size='large', color='brown', weight='bold')
	print(count_tot)
	print(count1)
	print(count2)
		# 	text = 1
		# 	p1.text(X.loc[line,"x1"]+0.1, X.loc[line,"y1"], text, horizontalalignment='left', 
		#             size='small', color='red', weight='semibold')
	plt.savefig('/Users/aggarwalpiush/github_repos/tweetnewsdetection/data/bert_embeddings/tSNE_bert_vs_reply.png')
	plt.figure()
	plt.pie()



def load_fp_fn(infile):
	fp_fn_dict = {}
	with codecs.open(infile, 'r', 'utf-8') as infile_obj:
		for line in infile_obj:
			line = line.split(':')
			fp_fn_dict[int(line[1].strip().rstrip('\r\n').replace('\n',''))] = line[0].strip()
	return fp_fn_dict

def load_test_file(infile):
	test_dict = {}
	row_number= 1
	with codecs.open(infile, 'r', 'utf-8') as infile_obj:
		for line in infile_obj:
			line = line.split('\t')
			test_dict[row_number] = int(line[2].strip().rstrip('\r\n').replace('\n',''))
			row_number += 1
	return test_dict


def get_tweet_vector(tweet_text, bert_embeddings):
	pass

def main():
	bert_file = '/Users/aggarwalpiush/github_repos/tweetnewsdetection/data/bert_embeddings/text_test_data/output.jsonl'
	bert_file_wrply = '/Users/aggarwalpiush/github_repos/tweetnewsdetection/data/bert_embeddings/textwreply_test_data/output.jsonl'
	fp_fn_infile = '/Users/aggarwalpiush/github_repos/tweetnewsdetection/test_results_text/BERT_bert_text_fn_fps.txt'
	fp_fn_infile_wreply = '/Users/aggarwalpiush/github_repos/tweetnewsdetection/test_results_text_wreply/BERT_bert_text_wreply_fn_fps.txt'
	test_file = '/Users/aggarwalpiush/github_repos/tweetnewsdetection/data/step_two/classification_data/text_wreply_exp_data/test_data.tsv'
	fp_fn_position_dict1 = load_fp_fn(fp_fn_infile)
	fp_fn_position_dict2 = load_fp_fn(fp_fn_infile_wreply)
	position_dict = load_test_file(test_file)
	#convert_embedding_2D(bert_file, position_dict, fp_fn_position_dict1)
	convert_embedding_2D_reply(bert_file, bert_file_wrply, fp_fn_position_dict1,fp_fn_position_dict2,position_dict)

if __name__ == '__main__':
	main()

