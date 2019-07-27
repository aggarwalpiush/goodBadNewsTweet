#! usr/bin/env python
# *--coding: utf-8 --*

import os
import numpy as np
from scripts.preprocessors.preprocessor_main import Corpus_Preprocessor
from scripts.features.get_characterstic_vector import Char_vector
from scripts.features.get_embedding_vector import MeanEmbeddingTransformer
from scripts.features.get_emoji_vector import EmojiVector
from scripts.features.get_interjection_vector import InterJ_vector
from scripts.features.get_lexicon_vector import Lex_vector
from scripts.features.get_most_common_terms_vector import ExtractTerms
from scripts.features.get_pos_vector import POS_vector
from scripts.features.get_sentiment_vector import Sent_vector
from scripts.features.get_tf_idf_vector import TFIDF_vector
from scripts.features.get_pos_tfidf_vector import POS_TFIDF_vector
from scripts.models.generate_models import Train_Models
from config.feature_config import EMBEDDING_PATH, MODEL_NAME, TEXT_TYPE, TEST_DOMAIN_ADAPTATION
from sklearn.preprocessing import MinMaxScaler


def main():
	scaler = MinMaxScaler()
	print('Preprocessing and split.........')
	cp = Corpus_Preprocessor(test_domain = TEST_DOMAIN_ADAPTATION)
	X_train, X_test, y_train, y_test = cp.split_dataset()
	cp.save_train_test_files()
	train_tweet_ids = []
	for instance in X_train:
		train_tweet_ids.append(instance.split('\t')[0])

	test_tweet_ids = []
	for instance in X_test:
		test_tweet_ids.append(instance.split('\t')[0])


	X_train = [x.split('\t')[1] for x in X_train]

	X_test = [x.split('\t')[1] for x in X_test]

	print('generating Characteterstics vector.........')

	print('train_vectors')
	train_char_features = Char_vector(train_tweet_ids).get_tweet_char()
	train_char_features = scaler.fit_transform(train_char_features)
	print('test_vectors')
	test_char_features = Char_vector(test_tweet_ids).get_tweet_char()
	test_char_features = scaler.transform(test_char_features)

	assert len(train_char_features) == len(X_train)
	assert len(test_char_features) == len(X_test)

	print(train_char_features[:2])

	print('generating Embedding vectors.........')

	if not os.path.exists(EMBEDDING_PATH +'.tmp'):
		print('Embedding file pruning')
		MeanEmbeddingTransformer(EMBEDDING_PATH).generate_temp_embfile(X_train + X_test)
	print('train_vectors')
	train_emb_features = MeanEmbeddingTransformer(EMBEDDING_PATH +'.tmp').fit_transform(X_train)
	print('test_vectors')
	test_emb_features = MeanEmbeddingTransformer(EMBEDDING_PATH +'.tmp').fit_transform(X_test)


	assert len(train_emb_features) == len(X_train)
	assert len(test_emb_features) == len(X_test)

	print(test_emb_features[:2])

	print('generating Emoticon vectors.........')

	print('train_vectors')
	train_emoji_features = EmojiVector(X_train, y_train).get_emoji_vector(X_train)
	print('test_vectors')
	test_emoji_features = EmojiVector(X_train, y_train).get_emoji_vector(X_test)

	assert len(train_emoji_features) == len(X_train)
	assert len(test_emoji_features) == len(X_test)

	print(train_emoji_features[:2])


	print('generating Interjections vectors.........')

	train_interj_features = InterJ_vector(X_train).get_tweet_interj()
	test_interj_features = InterJ_vector(X_test).get_tweet_interj()

	assert len(train_interj_features) == len(X_train)
	assert len(test_interj_features) == len(X_test)

	print(train_interj_features[:2])

	print('generating Lexicons vectors.........')

	train_lexical_features = Lex_vector(X_train).get_tweet_lexi()
	test_lexical_features = Lex_vector(X_test).get_tweet_lexi()

	assert len(train_lexical_features) == len(X_train)
	assert len(test_lexical_features) == len(X_test)


	print(train_lexical_features[:2])

	print('generating Pos vectors.........')

	train_pos_features = POS_vector(X_train).get_tweet_pos()
	test_pos_features = POS_vector(X_test).get_tweet_pos()

	assert len(train_pos_features) == len(X_train)
	assert len(test_pos_features) == len(X_test)

	print(train_pos_features[:2])

	print('generating sentiment vectors.........')

	train_senti_features = Sent_vector(X_train).get_tweet_senti()
	test_senti_features = Sent_vector(X_test).get_tweet_senti()

	assert len(train_senti_features) == len(X_train)
	assert len(test_senti_features) == len(X_test)

	print(train_senti_features[:2])


	print('generating Important terms vectors.........')

	train_imp_features = ExtractTerms(X_train,y_train).get_imp_vector(X_train)
	test_imp_features = ExtractTerms(X_train,y_train).get_imp_vector(X_test)

	assert len(train_imp_features) == len(X_train)
	assert len(test_imp_features) == len(X_test)

	print(train_imp_features[:2])

	print('generating TFIDF vectors.........')

	train_tfidf_features = TFIDF_vector(X_train).get_tweet_tfidf(X_train)
	test_tfidf_features = TFIDF_vector(X_train).get_tweet_tfidf(X_test)

	assert len(train_tfidf_features) == len(X_train)
	assert len(test_tfidf_features) == len(X_test)

	print(train_tfidf_features[:2])




	print('generating POS_TFIDF vectors.........')

	train_pos_tfidf_features = POS_TFIDF_vector(X_train).get_tweet_pos_tfidf(X_train)
	test_pos_tfidf_features = POS_TFIDF_vector(X_train).get_tweet_pos_tfidf(X_test)

	assert len(train_pos_tfidf_features) == len(X_train)
	assert len(test_pos_tfidf_features) == len(X_test)

	print(train_pos_tfidf_features[:2])




	# lets combine features

	# embedding and tf-idf

	print('generating embedding and tf-idf vectors.........')
	
	train_emb_tfidf = np.concatenate((train_emb_features, train_tfidf_features), 1)
	test_emb_tfidf = np.concatenate((test_emb_features, test_tfidf_features), 1)

	assert len(train_emb_tfidf) == len(X_train)
	print(train_emb_tfidf[:2])


	# embedding and pos_tf-idf

	print('generating embedding and pos tf-idf vectors.........')
	
	train_emb_pos_tfidf = np.concatenate((train_emb_features, train_pos_tfidf_features), 1)
	test_emb_pos_tfidf = np.concatenate((test_emb_features, test_pos_tfidf_features), 1)

	assert len(train_emb_pos_tfidf) == len(X_train)
	print(train_emb_pos_tfidf[:2])

	# local features

	print('generating local features..............')
	train_local = np.concatenate((train_char_features, train_emoji_features, train_interj_features,
		train_lexical_features, train_pos_features, train_senti_features, train_imp_features), 1)
	test_local = np.concatenate((test_char_features, test_emoji_features, test_interj_features,
		test_lexical_features, test_pos_features, test_senti_features, test_imp_features), 1)


	assert len(train_local) == len(X_train)
	print(train_local[:2])

	# local with tf_idf

	print('generating local and tf-idf features')
	train_local_tfidf = np.concatenate((train_local, train_tfidf_features), 1)
	test_local_tfidf = np.concatenate((test_local, test_tfidf_features), 1)


	assert len(train_local_tfidf) == len(X_train)
	print(train_local_tfidf[:2])


	# local with pos tf_idf

	print('generating local and pos tf-idf features')
	train_local_pos_tfidf = np.concatenate((train_local, train_pos_tfidf_features), 1)
	test_local_pos_tfidf = np.concatenate((test_local, test_pos_tfidf_features), 1)


	assert len(train_local_pos_tfidf) == len(X_train)
	print(train_local_pos_tfidf[:2])



	# local with embeddings

	print('generating local with embeddings features......')
	train_local_emb = np.concatenate((train_local, train_emb_features), 1)
	test_local_emb = np.concatenate((test_local, test_emb_features), 1)


	assert len(train_local_emb) == len(X_train)
	print(train_local_emb[:2])

	# total

	print('generating all features......')
	train_total = np.concatenate((train_local, train_emb_tfidf, train_pos_tfidf_features), 1)
	test_total = np.concatenate((test_local, test_emb_tfidf, test_pos_tfidf_features), 1)


	assert len(train_total) == len(X_train)
	print(train_local_emb[:2])





	for my_model in MODEL_NAME:
		# Train_Models(my_model, 'characterstic', TEXT_TYPE, train_char_features, y_train, test_char_features, y_test).evaluate_model()
		Train_Models(my_model, 'emb', TEXT_TYPE, train_emb_features, y_train, test_emb_features, y_test).evaluate_model()
		# Train_Models(my_model, 'emoji', TEXT_TYPE, train_emoji_features, y_train, test_emoji_features, y_test).evaluate_model()
		# Train_Models(my_model, 'interj', TEXT_TYPE, train_interj_features, y_train, test_interj_features, y_test).evaluate_model()
		# Train_Models(my_model, 'lexicon', TEXT_TYPE, train_lexical_features, y_train, test_lexical_features, y_test).evaluate_model()
		# Train_Models(my_model, 'pos', TEXT_TYPE, train_pos_features, y_train, test_pos_features, y_test).evaluate_model()
		# Train_Models(my_model, 'senti', TEXT_TYPE, train_senti_features, y_train, test_senti_features, y_test).evaluate_model()
		# Train_Models(my_model, 'imp', TEXT_TYPE, train_imp_features, y_train, test_imp_features, y_test).evaluate_model()
		# Train_Models(my_model, 'tfidf', TEXT_TYPE, train_tfidf_features, y_train, test_tfidf_features, y_test).evaluate_model()
		#Train_Models(my_model, 'emb_tfidf', TEXT_TYPE, train_emb_tfidf, y_train, test_emb_tfidf, y_test).evaluate_model()
		# Train_Models(my_model, 'local', TEXT_TYPE, train_local, y_train, test_local, y_test).evaluate_model()
		# Train_Models(my_model, 'local_tfidf', TEXT_TYPE, train_local_tfidf, y_train, test_local_tfidf, y_test).evaluate_model()
		# Train_Models(my_model, 'local_emb', TEXT_TYPE, train_local_emb, y_train, test_local_emb, y_test).evaluate_model()
		# Train_Models(my_model, 'total', TEXT_TYPE, train_total, y_train, test_total, y_test).evaluate_model()
		# Train_Models(my_model, 'local_pos_tfidf', TEXT_TYPE, train_local_pos_tfidf, y_train, test_local_pos_tfidf, y_test).evaluate_model()
		# Train_Models(my_model, 'emb_pos_tfidf', TEXT_TYPE, train_emb_pos_tfidf, y_train, test_emb_pos_tfidf, y_test).evaluate_model()
		# Train_Models(my_model, 'pos_tf_idf', TEXT_TYPE, train_pos_tfidf_features, y_train, test_pos_tfidf_features, y_test).evaluate_model()

if __name__ == '__main__':
	main()



	




