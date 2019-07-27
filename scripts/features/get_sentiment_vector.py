#! usr/bin/env python
# *-- coding : utf-8 --*


from textblob import TextBlob as tb
import numpy as np
from tqdm import tqdm


SENTIMENT_VECTOR = ['polarity', 'subjectivity']


class Sent_vector(object):
	def __init__(self, corpus):
		self.corpus = corpus

	def get_tweet_senti(self):
		senti_vec = np.zeros((len(self.corpus), len(SENTIMENT_VECTOR)), float)
		for i, instance in tqdm(enumerate(self.corpus)):
			senti = tb(instance).sentiment
			senti_vec[i][0] = senti[0]
			senti_vec[i][1] = senti[1]
		return senti_vec

def main():
	text_corpus = ['i am very happy to be in germany', 'germany is very boring place', 'Academically germany is very strong',
	'Indians are mean in germany']
	print(Sent_vector(text_corpus).get_tweet_senti())

	#print(pv.get_tweet_pos)


if __name__ == '__main__':
	main()

