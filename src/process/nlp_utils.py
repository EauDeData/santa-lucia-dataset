import string
import nltk
import re
from typing import *
from gensim.utils import simple_preprocess
from nltk.stem import SnowballStemmer
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from sentence_transformers import SentenceTransformer, util
from flair.data import Sentence
from flair.models import SequenceTagger
from scipy.optimize import linear_sum_assignment
from Levenshtein import distance
import numpy as np

nltk.download('punkt')

class StringCleanAndTrim:
    def __init__(self, stemm = True, lang = 'spanish') -> None:
        self.stemm = stemm
        self.lang = lang
        self.stopwords = nltk.corpus.stopwords.words(lang) # TODO, wtf hardcoded this, for real???


    def __call__(self, batch, *args: Any, **kwds: Any) -> Any:
        '''
        Call function for string cleaner and trimmer. Receives a batch of strings and cleanses them.
        Steps: 
            0. Lower Case
            1. Lemmatization
            2. Punctation

        Args:
            batch: list fo strings to clean
        returns:
            list of cleansed strings of format [[w1, w2, ..., wn], d2, ..., dn].
                w: words
                d: documents of batch
        '''

        shorter = SnowballStemmer(self.lang).stem if self.stemm else WordNetLemmatizer(self.lang).lemmatize
        lemma = [shorter(re.sub('[^A-Za-z0-9]+', '', x)) for x in batch.lower().split() if not x in self.stopwords]
        return lemma    

spanish = 'distiluse-base-multilingual-cased-v1'
TAGGER = SequenceTagger.load("flair/ner-spanish-large")

## Get Distance 2 Sentences ###
def sentence_proximity(query, sentences, model):

    sentence_embeddings = model.encode([query] + sentences)
    sims = []
    for emb in sentence_embeddings[1:]:
        sims.append(util.cos_sim(sentence_embeddings[0], emb))
    
    return sims


## clean sentence ###
def clean_sentence(s1, cleaner):
    return cleaner(s1)

def ner_detection(s1, tagger):
    # make example sentence
    sentence = Sentence(s1)
    tagger.predict(sentence)
    tokens = [[y.replace('"', '').lower() for y in x.shortstring.split('/')] for x in sentence.get_labels()]
    splitted = []

    for token in tokens:
        words = token[0].split()
        cat = [token[1]] * len(words)
        splitted.extend(zip(words, cat))

    return splitted
        
        


def edit_distance(w1, w2):
    return distance(w1, w2)

def hungarian_distance(s1, s2, word_distance = edit_distance):
    s1_split, s2_split = s1.split(), s2.split()
    weights_matrix = np.zeros((len(s1_split), len(s2_split)))
    
    for n, word_s1 in enumerate(s1_split):
        for m, word_s2 in enumerate(s2_split):
            weights_matrix[n, m] = word_distance(word_s1, word_s2)

    row_ind, col_ind = linear_sum_assignment(weights_matrix)
    return weights_matrix[row_ind, col_ind].sum()
    

                              