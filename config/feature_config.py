## WARNING: PLEASE TAKE CARE OF TEST OUTPUT PATH, CHANGE IF YOU RUNNING THE MODDELS IF NECESSARY

DATA_PATH = '/Users/aggarwalpiush/github_repos/tweetnewsdetection/data/step_one/classification_data/text'

LABELS_PATH = '/Users/aggarwalpiush/github_repos/tweetnewsdetection/data/step_one/classification_data/labels/labels.txt'

TEXT_TYPE = 'text_newsnotnews_natdisaster_mix'

TEST_DOMAIN_ADAPTATION = True

DATA_DOMAIN_PATH = '/Users/aggarwalpiush/github_repos/tweetnewsdetection/data/step_one/classification_data/domain_dataset/natdisaster_mix'

TOP_N_TERMS = 300

TOP_N_GRAMS = 2



TERMS_FILE_PATH = '/Users/aggarwalpiush/github_repos/tweetnewsdetection/intermediate'

CHARACTERSTICS_PATH = '/Users/aggarwalpiush/github_repos/tweetnewsdetection/intermediate/characterstics'

FEATURE_PATH = '/Users/aggarwalpiush/github_repos/tweetnewsdetection/feature_vectors'

###################################################
# PLEASE change following path before running main#
# file.											  #
#												  #
###################################################

TEST_OUTPUT_PATH = '/Users/aggarwalpiush/github_repos/tweetnewsdetection/test_results_newsnotnews_'+TEXT_TYPE

EMBEDDING_PATH = '/Users/aggarwalpiush/github_repos/tweetnewsdetection/intermediate/crawl-300d-2M.vec'

EMOJI_PATH = '/Users/aggarwalpiush/github_repos/tweetnewsdetection/intermediate/sig_emoji_vector_newsnotnews_'+TEXT_TYPE+'.npy'

POSITIVE_INTERJ_FILE = '/Users/aggarwalpiush/github_repos/tweetnewsdetection/intermediate/positive-interjections.txt'

NEGATIVE_INTERJ_FILE = '/Users/aggarwalpiush/github_repos/tweetnewsdetection/intermediate/negative-interjections.txt'

POSITIVE_LEXICON_FILE = '/Users/aggarwalpiush/github_repos/tweetnewsdetection/intermediate/positive-words.txt'

NEGATIVE_LEXICON_FILE = '/Users/aggarwalpiush/github_repos/tweetnewsdetection/intermediate/negative-words.txt'


TF_IDF_MAX_FEATURES = 300

TF_IDF_MAX_NGRAM_RANGE = 3

POS_TF_IDF_MAX_FEATURES = 300

POS_TF_IDF_MAX_NGRAM_RANGE = 2


TEST_SIZE = 0.15

PARAM_GRID_MLPClassifier = {'activation': ['logistic'],'alpha': [0.001],'learning_rate': ['adaptive'], 'tol': [ 0.1]}

PARAM_GRID_SVC = {'C': [0.1, 1, 10], 'kernel' : ['rbf'], 'gamma' : [ 'scale']}

PARAM_GRID_LR = {'C': [0.1, 1.0, 10.0], 'penalty' : [ 'l2', 'l1']}

PARAM_GRID_KNN = {'n_neighbors': [15,20,25,30,35], 'weights' : [ 'uniform', 'distance']}

PARAM_GRID_RF = {'n_estimators':[150,500,1000], 'max_depth':[4,5,10,15,20]}

PARAM_GRID_LSVM = {'C': [0.1, 1.0, 10.0], 'penalty' : [ 'l2']}

PARAM_GRID_DT = {'min_samples_split':[2,3,4,5], 'min_samples_leaf':[1,2,3,4,5]}

PARAM_GRID_XGB = {}

'''

Naming convention for evelaute ML classifiers

Naming conventions for ML classifiers:
MLPClassifier = MLPC
Support Vector rbf = SVC
Logistic regression = LR
KNeighborsClassifier = KNN
RandomForestClassifier = RF
LinearSVC = LSVC
DecisionTreeClassifier = DT
XGBClassifier = XGB

'''

MODEL_NAME = [ 'SVC']

NUM_SPLITS = 5

