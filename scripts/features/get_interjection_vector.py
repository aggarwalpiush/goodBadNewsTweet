#! usr/bin/env python
# *-- coding : utf-8 --*


import numpy as np
import codecs
from tqdm import tqdm
from config.feature_config import POSITIVE_INTERJ_FILE, NEGATIVE_INTERJ_FILE


INTERJECTION_VECTOR = ['positive', 'negative']



class InterJ_vector(object):
	def __init__(self, corpus):
		self.corpus = corpus

	def load_words(self, infile):
		out_list = []
		with codecs.open(infile, 'r', 'utf-8') as load_obj:
			for line in load_obj:
				if line[0] != ';' and line.strip().rstrip('\r\n').replace('\n','') != '':
					out_list.append(line.strip().rstrip('\r\n').replace('\n','')) 
		return set(out_list)


	def get_tweet_interj(self):
		interj_vec = np.zeros((len(self.corpus), len(INTERJECTION_VECTOR)), float)
		pos_list = self.load_words(POSITIVE_INTERJ_FILE)
		neg_list = self.load_words(NEGATIVE_INTERJ_FILE)
		for i, instance in tqdm(enumerate(self.corpus)):
			for w in pos_list:
				if w in instance:
					interj_vec[i][0] = 1
			for w in neg_list:
				if w in instance:
					interj_vec[i][1] = 1
		return interj_vec

def main():
	text_corpus = ['i am very happy to be in germany boo-hoo', 'germany is very  yahoo boring place', 'Academically yahoo germany is very strong',
	'Indians are mean in germany boo-hoo']
	print(InterJ_vector(text_corpus).get_tweet_interj())

	#print(pv.get_tweet_pos)


if __name__ == '__main__':
	main()

