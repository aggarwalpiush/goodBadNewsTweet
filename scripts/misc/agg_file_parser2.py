#! usr/bin/env python
# -*- coding : utf-8 -*-

'''
The main pupose of this file to fix issue available in figure8 aggregation report where report is generated with 
altered tweetids having ending with '00' 
This scripts will take 2 downloaded csv files from figure8 crowdsourcing 
1. aggregation report having following comma columns

- id
- supportradiogroup
- supportradiogroup:confidence

2. Data file having following columns

- _unit_id
- _created_at
- _updated_at
- _golden
- gold output column
- gold output column reason
- text
- tweet_id
- justification_gold
- justification_gold_reason
- supportradiogroup_gold
- supportradiogroup_gold_reason

Output file contain tab separated following columns
- id
- tweetid
- supportradiogroup
- supportradiogroup:confidence

'''

import codecs
import sys


def main():
	agg_file = sys.argv[1]
	ctl_file = sys.argv[2] 
	output_file = agg_file + '.out'
	ctl_tweet = {} 
	count = 0

	with codecs.open(ctl_file, 'r', 'utf-8') as ctl_obj:
		for i, line in enumerate(ctl_obj):
			if i == 0:
				continue
			line = line.split(',')
			if line[0].isdigit():
				if line[0] in  ctl_tweet.keys():
					raise NameError('Duplicate id found in ctl data %s' % line[0])
				else:
					ctl_tweet[line[0]] = line[-5]
			else:
				raise NameError('wrong id provided %s' %line[0])
		if len(list(ctl_tweet.keys())) != i:
			raise NameError('Records missed')
	print(len(list(ctl_tweet.keys())))


	with codecs.open(output_file, 'w', 'utf-8') as out_file:
		with codecs.open(agg_file, 'r', 'utf-8') as agg_obj:
			for i,line in enumerate(agg_obj):
				if i == 0:
					continue
				line = line.split(',')
				if line[0].isdigit():
					if line[0] in ctl_tweet.keys():
						if ctl_tweet[line[0]].isdigit():
							out_file.write('\t'.join([line[0], ctl_tweet[line[0]], line[1], line[2].strip('\n\r')])+ '\n')
							count += 1
					else:
						raise NameError('input data and aggregated report mismatch id %s' %line[0])
				else:
					raise NameError('invalid id format %s' %line[0])
			if count != i:
				print('some Records missed because of invalid tweetid format')

if __name__ == '__main__':
	main()
