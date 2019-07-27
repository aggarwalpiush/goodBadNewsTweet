#! usr/bin/env python
# *-- coding : utf-8 --*


import numpy as np
import codecs
from tqdm import tqdm
from config.feature_config import POSITIVE_LEXICON_FILE, NEGATIVE_LEXICON_FILE


LEXICON_VECTOR = ['positive', 'negative']




class Lex_vector(object):
	def __init__(self, corpus):
		self.corpus = corpus

	def load_words(self, infile):
		out_list = []
		with codecs.open(infile, 'r', 'utf-8') as load_obj:
			for line in load_obj:
				if line[0] != ';' and line.strip().rstrip('\r\n').replace('\n','') != '':
					out_list.append(line.strip().rstrip('\r\n').replace('\n','')) 
		#for i, d in enumerate(out_list):
			# print(d)
			# if i == 5:
			# 	break
		return set(out_list)


	def get_tweet_lexi(self):
		lexi_vec = np.zeros((len(self.corpus), len(LEXICON_VECTOR)), float)
		pos_list = self.load_words(POSITIVE_LEXICON_FILE)
		neg_list = self.load_words(NEGATIVE_LEXICON_FILE)
		for i, instance in tqdm(enumerate(self.corpus)):
			for w in instance.split(' '):
				if w in pos_list:
					lexi_vec[i][0] = 1
				if w in neg_list:
					lexi_vec[i][1] = 1
		return lexi_vec

def main():
	text_corpus = ['i am very happy to be in germany', 'germany is very boring place', 'Academically germany is very strong',
	'Indians are mean in germany']
	print(Lex_vector(text_corpus).get_tweet_lexi())

	#print(pv.get_tweet_pos)


if __name__ == '__main__':
	main()

