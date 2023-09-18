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
from typing import Callable
import gensim
import gensim.downloader as api
import gensim.corpora as corpora
import numpy as np
from typing import *
from gensim.models import Word2Vec, KeyedVectors
from gensim.test.utils import common_texts
import os
from transformers import BertTokenizer, BertModel
import math

# https://open.spotify.com/track/2QLrYSnATJYuRThUZtdhH3?si=faed86630309414a

def yieldify_dataiter(dataiter: Callable, function: Callable):
    '''
    Maybe we should just use map() Python function.
    But I'm afraid I want to be able to control threads given the opportunity.
    '''
    for data in dataiter:
        yield function([data])

def _train_precondition(obj) -> None:
    if isinstance(obj.model, int): raise NotImplementedError

class BaseMapper:
    model_instance = None
    popfitted = None
    def __init__(self, dataset, string_preprocess: Callable = StringCleanAndTrim(), ntopics = 224, *args, **kwargs) -> None:
        self.dataset = dataset
        self.prep = string_preprocess
        self.model = 0 # IF it's an int, a not trained error will be rised
        
        self.ntopics = ntopics

    def __getitem__(self, index: int) -> np.ndarray:
        _train_precondition(self)
        instance = self.model[self.corpus[index]]
        gensim_vector = gensim.matutils.sparse2full(instance, self.vector_size)
        return gensim_vector
    def pop_fit(self, dict):
        return {x: dict[x] for x in dict if x in self.popfitted}
    
    def fit(self) -> None:
        
        sentences = [self.prep(x) for x in self.dataset]
        print("Dataset processed...", sentences[-1])
        self.dct = gensim.corpora.Dictionary(documents=sentences)
        print('Creating Corpus...')
        self.corpus = [self.dct.doc2bow(line) for line in sentences]
        print('Fitting...', self.corpus[-1])
        self.model = self.model_instance(**self.pop_fit({
                                    "corpus":self.corpus,
                                    "id2word":self.dct,
                                    "num_topics":self.ntopics
                                    }))
        self.vector_size = len(self.dct) if self.name == 'tf-idf_mapper' else self.ntopics

    def predict(self, sentence):
        sentence_cleaned = self.prep(sentence)
        new_text_corpus =  self.dct.doc2bow(sentence_cleaned)
        return gensim.matutils.sparse2full(self.model[new_text_corpus], self.vector_size)

    def infer(self, index: int) -> Dict:

        return {
            "result": self[index]
        }

class TF_IDFLoader(BaseMapper):
    '''
    Given a dataset; loads its TF-IDF representation.
    self.fit: Builds the TF-IDF model.
    self.infer: Infers the TF-IDF representation of a new text.
    '''

    name = 'tf-idf_mapper'
    model_instance = gensim.models.TfidfModel
    popfitted = ['corpus'] # Accepted keywords in Fit function
     
class LDALoader(BaseMapper):
    # https://en.wikipedia.org/wiki/Latent_Dirichlet_allocation
    name = 'LDA_mapper'
    model_instance = gensim.models.LdaMulticore
    popfitted = ['corpus', 'id2word', 'num_topics'] # Accepted keywords in Fit function

class EnsembleLDALoader:
    name = 'EnsembleLDA_mapper'
    model_instance = gensim.models.EnsembleLda
    popfitted = ['corpus', 'id2word', 'num_topics'] # Accepted keywords in Fit function

class BOWLoader:
    name = 'BOW_mapper'
    def __init__(self, dataset, *args, **kwargs) -> None:
        pass

class LSALoader(BaseMapper):
    name = 'LSA_mapper'
    model_instance = gensim.models.LsiModel
    popfitted = ['corpus', 'id2word', 'num_topics'] # Accepted keywords in Fit function


class BertTextEncoder:
    def __init__(self, pretrained = 'bert-base-multilingual-cased') -> None:
        self.tokenizer = BertTokenizer.from_pretrained(pretrained)
        self.model = BertModel.from_pretrained(pretrained)

    def predict(self, batch):
        encoded_input = self.tokenizer(batch, return_tensors='pt', padding=True).to(self.model.device)
        return self.model(**encoded_input).pooler_output # (BS, 768) 
    
    def to(self, device):
        self.model = self.model.to(device)
        return self

class TextTokenizer:
    bos = '< BOS >'
    eos = '< EOS >'
    unk = '< UNK >'
    pad = '< PAD >'

    def __init__(self, cleaner) -> None:
        self.cleaner = cleaner
        self.tokens = None

    def predict(self, text: str, padding = 'default'):
        tokens = self.cleaner(text)
        tokens = [self.bos] + tokens + [self.eos]
        num_tokens = len(tokens) # TODO: Implement padding

        vector = [None for _ in tokens]
        for n, token in enumerate(tokens): vector[n] = self.tokens[token] if token in self.tokens else self.tokens[self.unk]

        return vector

    def fit(self, dataset):
        freqs = {}
        self.min_leng = 1
        for sntc in dataset.iter_text():
            tokens = self.cleaner(sntc)
            if len(tokens) > self.min_leng: self.min_leng = len(tokens)
            for token in tokens: 
                if token not in freqs: freqs[token] = 0
                freqs[token] += 1
        
        freqs[self.bos] = np.inf
        freqs[self.eos] = np.inf
        freqs[self.unk] = np.inf
        
        self.tokens = {y: n for n, y in enumerate(sorted(freqs, key = lambda x: -freqs[x]))}

spanish = 'distiluse-base-multilingual-cased-v1'
# TAGGER = SequenceTagger.load("flair/ner-spanish-large")

## Get Distance 2 Sentences ###
def sentence_proximity(query, sentences, model):

    sentence_embeddings = model.encode([query] + sentences)
    sims = []
    for emb in sentence_embeddings[1:]:
        sims.append(util.cos_sim(sentence_embeddings[0], emb).squeeze().item())
    
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
    

                              