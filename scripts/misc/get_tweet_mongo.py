#! usr/bin/env python
# -*- coding : utf-8 -*-

import codecs
import pymongo
from pymongo import MongoClient
import sys
import os

CLIENT = MongoClient('mongodb://127.0.0.1:27017')
DB = CLIENT.mttweetlabelsDB
COLLECTIONS = ['Ebola', 'agu17','harvey_hurricane','hiv','iot','irma','macerata','nintendo','stockholm','swachhbharat', 'swachhbharat_v2', 'hiv_v2']


def get_tweet_text(tweetid):
	print(tweetid)
	for col in COLLECTIONS:
		collec = DB[col]
		tweet_text = collec.find_one({"id_str":str(int(tweetid)), "$or": [ { "full_text": {"$exists": True} }, { "text": {"$exists": True} } ]},{'full_text':1,'text':1,"_id":0})
		if not tweet_text is None:
			if 'text' in tweet_text.keys() and 'full_text' not in tweet_text.keys():
				tweet_text['full_text'] = tweet_text['text']
			if tweet_text['full_text'] != '':
				tweet_emb_text = collec.find_one({"id_str":str(int(tweetid)), "is_quote_status":True, "quoted_status": {"$exists": True}},{'quoted_status.text':1,'quoted_status.full_text':1,"_id":0})
				tweet_content = tweet_text['full_text'].rstrip('\n\r''"')
				if not tweet_emb_text is None:
					if 'text' in tweet_emb_text['quoted_status'].keys() and 'full_text' not in tweet_emb_text['quoted_status'].keys():
						tweet_emb_text['quoted_status']['full_text'] = tweet_emb_text['quoted_status']['text']
					tweet_content = tweet_text['full_text'].rstrip('\n\r''"') + ' tweeted on ' + tweet_emb_text['quoted_status']['full_text'].rstrip('\n\r''"')
				break

	if tweet_text is None:
		raise NameError('Bad Tweet id %s' %tweetid)	
	else:
		tweet_content = ' '.join(tweet_content.split('\n'))
		tweet_content = ' '.join(tweet_content.split('\r\n'))
		tweet_content = ' '.join(tweet_content.split('\r'))
		tweet_content = ' '.join(tweet_content.split('\n\r'))
		return tweet_content.rstrip('\n\r''"')


def main():
	tweetids_file = sys.argv[1]
	save_site = sys.argv[2]

	tweets = []
	with codecs.open(tweetids_file, 'r', 'utf-8') as in_obj:
		for line in in_obj:
			line = int(line.strip('\r\n'))
			tweets.append(line)

		for tw in tweets:
			tweet_text = get_tweet_text(tw)
			with codecs.open(os.path.join(save_site,str(int(tw)) + '.txt'), 'w', 'utf-8') as tw_obj:
				tw_obj.write(tweet_text)


if __name__ == '__main__':
	main()







