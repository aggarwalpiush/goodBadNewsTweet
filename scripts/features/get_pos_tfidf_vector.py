#!usr/bin/env python
# *--coding : utf-8 --*



from textblob import TextBlob as tb
import numpy as np
from tqdm import tqdm
from sklearn.feature_extraction.text import TfidfVectorizer
from config.feature_config import POS_TF_IDF_MAX_FEATURES, POS_TF_IDF_MAX_NGRAM_RANGE





class POS_TFIDF_vector(object):
	def __init__(self, corpus):
		self.corpus = corpus

	def generate_pos_corpus(self, incorpus):
		pos_corpus = []
		for i, instance in tqdm(enumerate(incorpus)):
			pos_corpus.append(' '.join([x[1] for x in tb(instance).tags]))
		return pos_corpus


	def get_tweet_pos_tfidf(self, new_corpus):
		pos_corpus = self.generate_pos_corpus(self.corpus)
		new_pos_corpus = self.generate_pos_corpus(new_corpus)
		vectorizer = TfidfVectorizer(ngram_range=(1, POS_TF_IDF_MAX_NGRAM_RANGE), max_features=POS_TF_IDF_MAX_FEATURES)
		vectorizer.fit(pos_corpus)
		pos_tf_idf_vec = np.zeros((len(new_pos_corpus), POS_TF_IDF_MAX_FEATURES), float)
		for i, instance in tqdm(enumerate(new_pos_corpus)):
			vec_out = vectorizer.transform([instance]).toarray()[0]
			for j,val in enumerate(vec_out):
				pos_tf_idf_vec[i][j] = val
			#tf_idf_vec.append(np.array(vectorizer.transform([instance]).toarray()[0]))
		return pos_tf_idf_vec


def main():
	text_corpus = ['i am very happy to be in germany', 'germany is very boring place', 'Academically germany is very strong',
	'Indians are mean in germany']
	print(POS_TFIDF_vector(text_corpus).get_tweet_pos_tfidf(text_corpus))

	#print(pv.get_tweet_pos)


if __name__ == '__main__':
	main()




