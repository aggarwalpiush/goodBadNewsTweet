#! usr/bin/env python
# -*- coding : utf-8 -*-

import codecs
import pymongo
from pymongo import MongoClient
import sys

CLIENT = MongoClient('mongodb://127.0.0.1:27017')
DB = CLIENT.mttweetlabelsDB
COLLECTIONS = ['Ebola', 'agu17','harvey_hurricane','hiv','iot','irma','macerata','nintendo','stockholm','swachhbharat']


def main():
	tweetids_file = sys.argv[1]

	tweets = []
	annotated_count = {}
	with codecs.open(tweetids_file, 'r', 'utf-8') as in_obj:
		for line in in_obj:
			line = int(line.strip('\r\n'))
			tweets.append(line)

		for col in COLLECTIONS:
			collec = DB[col]
			annotated_count[col] = 0
			for tw in tweets:
				doc_id = collec.find_one({"id_str":str(int(tw))},{"_id":1})
				if not doc_id is None:
					annotated_count[col] += 1
	print(annotated_count)

if __name__ == '__main__':
	main()







