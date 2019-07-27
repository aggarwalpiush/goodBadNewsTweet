#! usr/bin/env python
# *-- coding : utf-8 --*


import numpy as np
import os
import json
from config.feature_config import CHARACTERSTICS_PATH
from tqdm import tqdm
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
CHARACTERSTICS_VECTOR = ['retweets', 'likes', 'comments']


class Char_vector(object):
	def __init__(self, tweet_list):
		self.tweet_list = tweet_list

	def get_tweet_char(self):
		instances_found = 0
		char_vec = np.zeros((len(self.tweet_list), len(CHARACTERSTICS_VECTOR)), float)
		for i, instance in tqdm(enumerate(self.tweet_list)):
			if os.path.exists(os.path.join(CHARACTERSTICS_PATH,str(instance),'source-tweets',str(instance)+'.json')):
				instances_found += 1
				with open(os.path.join(CHARACTERSTICS_PATH,str(instance),'source-tweets',str(instance)+'.json')) as fh:
					json_obj = json.load(fh)
				char_vec[i][0] =  json_obj['retweet_count']
				char_vec[i][1] =  json_obj['favorite_count']
				char_vec[i][2] =  sum([len(files) for r, d, files in os.walk(os.path.join(CHARACTERSTICS_PATH,str(instance),'reactions'))]) - 1 
			else:
				char_vec[i] = [.0,.0,.0]
		if instances_found == 0:
			logger.info('NO Characterstics found for the provided dataset')

		return char_vec

def main():
	tweet_ids = ['100533604869881856', '100580305185939456', '1234', '1006569270745190400', '1070694847500222464']
	text_corpus = ['i am very happy to be in germany', 'germany is very boring place', 'Academically germany is very strong',
	'Indians are mean in germany']
	print(Char_vector(tweet_ids).get_tweet_char())

	#print(pv.get_tweet_pos)


if __name__ == '__main__':
	main()

