#! usr/bin/env python
# *--coding : utf-8 --*

import codecs
import os
import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from sklearn.metrics import classification_report
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.metrics import roc_auc_score
from config.feature_config import PARAM_GRID_MLPClassifier, PARAM_GRID_SVC, PARAM_GRID_LR, PARAM_GRID_KNN
from config.feature_config import PARAM_GRID_RF, PARAM_GRID_LSVM, PARAM_GRID_DT, PARAM_GRID_XGB, MODEL_NAME, NUM_SPLITS
from config.feature_config import TEST_OUTPUT_PATH


class Train_Models(object):
	def __init__(self, model_name, feature_name, corpus_name, train_features, train_gold, test_features, test_gold):
		self.model_name = model_name
		self.train_features = train_features
		self.train_gold = train_gold
		self.test_features = test_features
		self.test_gold = test_gold
		self.feature_name = feature_name
		self.corpus_name = corpus_name
		self.test_output_path = TEST_OUTPUT_PATH
		if not os.path.exists(self.test_output_path):
			os.makedirs(self.test_output_path)


	def _generate_results(self, clf):
		clf.fit(self.train_features, self.train_gold)
		y_pred = clf.predict(self.test_features)
		y_pred = [round(value) for value in y_pred]
		y_test = self.test_gold
		out_data = np.append(np.transpose([np.array(y_pred)]), np.transpose([y_test]), axis=1)
		np.savetxt(os.path.join(self.test_output_path, self.model_name + '_' + self.feature_name + '_' + self.corpus_name + 
			'_test_results.txt'), out_data, fmt='%s	%s', delimiter='\t')
		return y_pred, y_test, clf.best_params_

	def _save_scores(self, clf):
		y_pred, y_test, best_params = self._generate_results(clf)
		target_names = ['Bad News', 'Good News']
		with codecs.open(os.path.join(self.test_output_path, self.model_name + '_' + self.feature_name + '_' + self.corpus_name 
			+  '_model_evaluation.txt'), 'w', 'utf-8') as scr_obj:
			scr_obj.write(str([str(key)+' : '+str(val) for key,val in best_params.items()]))
			scr_obj.write('\nPrecision score: {:3f}\n'.format(precision_score(y_test, y_pred, average='macro') ))
			scr_obj.write('Recall score: {:3f}\n'.format(recall_score(y_test, y_pred, average='macro') ))
			scr_obj.write('F1 score: {:3f}\n'.format(f1_score(y_test, y_pred, average='macro')))
			scr_obj.write('AUC score: {:3f}\n'.format(roc_auc_score(y_test, y_pred)))
			scr_obj.write('Confusion Metric : %s\n' %(confusion_matrix(y_test, y_pred)))
			scr_obj.write(classification_report(y_test, y_pred, target_names=target_names))


	def evaluate_model(self):
		'''
		Naming conventions for ML classifiers:

		MLPClassifier = MLPC
		Support Vector rbf = SVC
		Logistic regression = LR
		KNeighborsClassifier = KNN
		RandomForestClassifier = RF
		LinearSVC = LSVC
		DecisionTreeClassifier = DT
		XGBClassifier = XGB

		if adding or deleting any model kindly change the config file too
		'''

		if self.model_name == 'MLPC':
			#param_grid = {'activation': ['logistic'],'alpha': [0.001],'learning_rate': ['adaptive'], 'tol': [ 0.1]}
			gs = GridSearchCV(MLPClassifier(), 
		             param_grid = PARAM_GRID_MLPClassifier, scoring="f1_macro", cv = NUM_SPLITS)
			return self._save_scores(gs)

		if self.model_name == 'SVC':
			#param_grid2 = {'C': [0.1, 1, 10], 'kernel' : ['rbf'], 'gamma' : [ 'scale']}
			gs = GridSearchCV(SVC(), 
		             param_grid=PARAM_GRID_SVC, scoring="f1_macro", cv = NUM_SPLITS)
			return self._save_scores(gs)

		if self.model_name == 'LR':
			#param_grid3 = {'C': [0.1, 1.0, 10.0], 'penalty' : [ 'l2', 'l1']}
			gs = GridSearchCV(LogisticRegression(), 
		             param_grid=PARAM_GRID_LR, scoring="f1_macro", cv = NUM_SPLITS)
			return self._save_scores(gs)

		if self.model_name == 'KNN':
			#param_grid4 = {'n_neighbors': [15,20,25,30,35], 'weights' : [ 'uniform', 'distance']}
			gs = GridSearchCV(KNeighborsClassifier(), 
		             param_grid=PARAM_GRID_KNN, scoring="f1_macro", cv = NUM_SPLITS)
			return self._save_scores(gs)

		if self.model_name == 'RF':
			#param_grid5 = {'n_estimators':[150,500,1000], 'max_depth':[4,5,10,15,20]}
			gs = GridSearchCV(RandomForestClassifier(), 
		             param_grid=PARAM_GRID_RF, scoring="f1_macro", cv = NUM_SPLITS)
			return self._save_scores(gs)

		if self.model_name == 'LSVC':
			#param_grid6 = {'C': [0.1, 1.0, 10.0], 'penalty' : [ 'l2']}
			gs = GridSearchCV(LinearSVC(), 
		             param_grid=PARAM_GRID_LSVM, scoring="f1_macro", cv = NUM_SPLITS)
			return self._save_scores(gs)

		if self.model_name == 'DT':
			#param_grid7 = {'min_samples_split':[2,3,4,5], 'min_samples_leaf':[1,2,3,4,5]}
			gs = GridSearchCV(DecisionTreeClassifier(), 
		             param_grid=PARAM_GRID_DT, scoring="f1_macro", cv = NUM_SPLITS)
			return self._save_scores(gs)

		if self.model_name == 'XGB':
			#param_grid8 = {}
			gs = GridSearchCV(XGBClassifier(), 
		             param_grid=PARAM_GRID_XGB, scoring="f1_macro", cv = NUM_SPLITS)
			return self._save_scores(gs)

def main():
	from scripts.features.get_pos_vector import POS_vector
	tweet_ids = ['100533604869881856', '100580305185939456', '1234', '1006569270745190400', '1070694847500222464', '222222']
	text_corpus = ['i am very happy to be in germany', 'germany is very boring place', 'Academically germany is very strong',
	'Indians are mean in germany', 'hello how are you buddy ooo', 'I am getting something']
	labels = [0,1,0,1,0, 1]
	train_features = POS_vector(text_corpus).get_tweet_pos()
	test_features = POS_vector(text_corpus).get_tweet_pos()
	Train_Models(MODEL_NAME, 'pos', 'abc', train_features, labels, test_features, labels).evaluate_model()


if __name__ == '__main__':
	main()




	





