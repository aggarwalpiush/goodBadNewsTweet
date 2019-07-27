from sklearn.base import TransformerMixin
import numpy as np
import codecs
import os
import pickle
from tqdm import tqdm




class MeanEmbeddingTransformer(TransformerMixin):
    
    def __init__(self, embedding_file):
        self._vocab, self._E = self._load_words(embedding_file)
        self.embedding_file = embedding_file
        
    
    def _load_words(self, embedding_file):
        E = {}
        vocab = []

        with codecs.open(embedding_file, 'r', encoding="utf8") as file:
            for i, line in tqdm(enumerate(file)):
                if i == 0:
                    continue
                l = line.split(' ')
                if l[0].isalpha():
                    v = [float(i) for i in l[1:]]
                    E[l[0]] = np.array(v)
                    vocab.append(l[0])
        return np.array(vocab), E            

    
    def _get_word(self, v):
        for i, emb in enumerate(self._E):
            if np.array_equal(emb, v):
                return self._vocab[i]
        return None
    
    def _doc_mean(self, doc):
        word_array = []
        for w in doc.split():
            if  w.lower().strip() in self._E:
                word_array.append(self._E[w.lower().strip()])
            else:
                word_array.append(np.zeros([len(v) for v in self._E.values()][0]))

        return np.mean(np.array(word_array), axis=0)
                
    def generate_temp_embfile(self, corpus):
        seen = []
        with codecs.open(self.embedding_file+'.tmp', 'w', 'utf-8') as tmp_write:
            for instance in corpus:
                for w in instance.replace('\n', '').split(' '):
                    if  w.lower().strip() in self._E:
                        if w not in set(seen):
                            tmp_write.write("%s %s\n" %(w.lower().strip() , ' '.join([str(i) for i in self._E[w]])))
                            seen.append(w)

    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        return np.array([self._doc_mean(doc) for doc in tqdm(X)])
    
    def fit_transform(self, X, y=None):
        return self.fit(X).transform(X)

def get_emb_vector(embedding_file, corpus):
    if os.path.exists(embedding_file +'.tmp'):
        met = MeanEmbeddingTransformer(embedding_file+'.tmp')
        X_transform = met.fit_transform(corpus)
        return X_transform
    else:
        met = MeanEmbeddingTransformer(embedding_file)
        met.generate_temp_embfile(corpus)
        X_transform = met.fit_transform(corpus)
        return X_transform

def main():
    from config.feature_config import EMBEDDING_PATH
    text_corpus = ['i am very happy to be in germany', 'germany is very boring place', 'Academically germany is very strong',
    'Indians are mean in germany']
    print(get_emb_vector(EMBEDDING_PATH, text_corpus))

    #print(pv.get_tweet_pos)


if __name__ == '__main__':
    main()

