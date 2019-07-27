#! /usr/bin/env python
#*-- coding : utf-8 --*

import emoji
import numpy as np
import os
import codecs
from tqdm import tqdm
from config.feature_config import EMOJI_PATH



class EmojiVector(object):
	
	def __init__(self, corpus, labels):
		self.corpus = corpus
		self.labels = labels
		self.num_classes = len(np.unique(labels))

	def get_emoji_lists(self):
		emoji_list = []
		for n in range(self.num_classes):
			emoji_list.append([])
		for i, instance in enumerate(self.corpus):
			for j in emoji.UNICODE_EMOJI:
				if j in instance:
					for k in range(self.num_classes):
						if k == self.labels[i]:
							if j not in set(emoji_list[k]):
								emoji_list[k].append(j)
		return emoji_list


	def get_emoji_vector(self, new_corpus):
		if not os.path.exists(EMOJI_PATH):
			extemoji = self.get_emoji_lists()
			np.savetxt(EMOJI_PATH, extemoji, fmt='%s')
		else:
			extemoji = []
			with codecs.open(EMOJI_PATH, 'r', 'utf-8') as emoji_file_obj:
				for cl_emojis in emoji_file_obj:
					cl_emojis = cl_emojis.strip().replace(' ','').replace('[','')
					cl_emojis = cl_emojis.replace(']', '').replace('\n','').replace('\'','')
					extemoji.append(cl_emojis.split(','))
		emj_vec = np.zeros((len(new_corpus), self.num_classes), float)

		for c in tqdm(range(self.num_classes)):		
			for i, instance in enumerate(new_corpus):
				for j in extemoji[c]:
					if j in instance:
						emj_vec[i][c] = 1.0
		return emj_vec










if __name__ == '__main__':
	text_corpus = ['i am very happy to be in germany 😀 You', 'germany is very 😟  boring place', 'Academically 👉 You germany is very strong',
	'Indians are mean in You 😟  germany']
	labels = [1, 0, 1, 0]
	print(EmojiVector(text_corpus,labels).get_emoji_vector(text_corpus))
	# extterms = ExtractTerms(text_corpus, 4,2,2).get_top_terms()
	# if 'very boring' in extterms[1]:
	# 	print('true')
	# print(np.array(extterms)[0])






