#! usr/bin/env python
# *-- coding : utf-8 --*


from textblob import TextBlob as tb
import numpy as np
from tqdm import tqdm


#  https://pythonprogramming.net/natural-language-toolkit-nltk-part-speech-tagging/
POS_LIST = ['CC',	'CD',	'DT',	'EX',	'FW',	'IN',	'JJ',	'JJR',	'JJS',	'LS',	'MD',	'NN',	
'NNS',	'NNP',	'NNPS',	'PDT',	'POS',	'PRP',	'PRP$',	'RB',	'RBR',	'RBS',	'RP',	'TO',	'UH',	'VB',	
'VBD',	'VBG',	'VBN',	'VBP',	'VBZ',	'WDT',	'WP',	'WP$', 'WRB', 'SYM']


class POS_vector(object):
	def __init__(self, corpus):
		self.corpus = corpus

	def get_tweet_pos(self):
		pos_vec = np.zeros((len(self.corpus), len(POS_LIST)), float)
		for i, instance in tqdm(enumerate(self.corpus)):
			for tag in tb(instance).tags:
				pos_vec[i][POS_LIST.index(tag[1])] = 1.0
		return pos_vec

def main():
	text_corpus = ['Ebola virus outbreak threatens west Africa.', 'germany is very boring place', 'Academically germany is very strong',
	'Indians are mean in germany']
	print(POS_vector(text_corpus).get_tweet_pos())

	#print(pv.get_tweet_pos)


if __name__ == '__main__':
	main()




