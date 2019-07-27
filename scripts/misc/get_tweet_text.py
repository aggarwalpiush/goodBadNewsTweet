# /urr/bin/env python
# *-- coding : utf-8 --*

import tweepy

consumer_key = 'KQIA4UKnXvLfQZX8ycTCXxT1V'
consumer_secret = 'fUjoxxJoyZSe0KjMkPrtvnlQ3LPBSNOrokjC9Si7YrVFGy5OX6'
access_token = '1006645590992785408-OKIM9hFF41PTrXdXfX62LVS4UYa2Yv'
access_token_secret = 'lyunnQNPeBPHHnlWBFmj5fKgS6OeX58O1UF3iNP54KDax'


class TweetReader(object):
	def __init__(self):

		self.auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
		self.auth.set_access_token(access_token, access_token_secret)
		self.api = tweepy.API(self.auth, wait_on_rate_limit=True, wait_on_rate_limit_notify=True)

	def get_tweet_text(self, tweetid):
		try:
			results = self.api.statuses_lookup([tweetid])[0].text.rstrip('\n\r').replace('\n','').replace('\t','')
			return results
		except tweepy.TweepError:
			print('Something went wrong, quitting...')


# do whatever it is to get por.TweetID - the list of all IDs to look up

def main():
	tr = TweetReader()
	print(tr.get_tweet_text(1100017904421126145))


if __name__ == '__main__':
	main()
