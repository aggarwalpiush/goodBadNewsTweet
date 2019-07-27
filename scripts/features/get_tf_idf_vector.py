#! usr/bin/env python
# -*- coding : utf-8 -*-


from sklearn.feature_extraction.text import TfidfVectorizer
from tqdm import tqdm
import numpy as np
from config.feature_config import TF_IDF_MAX_FEATURES, TF_IDF_MAX_NGRAM_RANGE




class TFIDF_vector(object):
	def __init__(self, corpus):
		self.corpus = corpus

	def get_tweet_tfidf(self, new_corpus):
		vectorizer = TfidfVectorizer(ngram_range=(1, TF_IDF_MAX_NGRAM_RANGE), max_features=TF_IDF_MAX_FEATURES)
		vectorizer.fit(self.corpus)
		tf_idf_vec = np.zeros((len(new_corpus), TF_IDF_MAX_FEATURES), float)
		for i, instance in tqdm(enumerate(new_corpus)):
			vec_out = vectorizer.transform([instance]).toarray()[0]
			for j,val in enumerate(vec_out):
				tf_idf_vec[i][j] = val
			#tf_idf_vec.append(np.array(vectorizer.transform([instance]).toarray()[0]))
		return tf_idf_vec

def main():
	text_corpus = ['suspected ebola patient negative for deadly disease', 'visualise wave data from a buoy in a physical installation', 'Academically germany is very strong',
	'dragon ball fighterz confirmed for switch']
	print(TFIDF_vector(text_corpus).get_tweet_tfidf(text_corpus))

	#print(pv.get_tweet_pos)


if __name__ == '__main__':
	main()
