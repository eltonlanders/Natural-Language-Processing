# -*- coding: utf-8 -*-
"""
Created on Thu May 20 19:05:11 2021

@author: elton
"""
import numpy as np
import pandas as pd
import nltk
import re
nltk.download('stopwords')
nltk.download('wordnet')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer 
from nltk.stem.snowball import SnowballStemmer
from nltk.stem.wordnet import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer



# The text data
text = ('''There was so much to read, for one thing, and so much fine health to
be pulled down out of the young breath-giving air.''', '''I bought a dozen
volumes on banking and credit and investment securities, and they
stood on my shelf in red and gold like new money from the mint,
promising to unfold the shining secrets that only Midas and Morgan and
Maecenas knew.''', '''And I had the high intention of reading many other
books besides.''', '''I was rather literary in college—one year I wrote a
series of very solemn and obvious editorials for the Yale News—and now
I was going to bring back all such things into my life and become
again that most limited of all specialists, the “well-rounded man.”''',
'''This isn’t just an epigram—life is much more successfully looked at
from a single window, after all.''')

corpus = pd.Series(text)


# The preprocessing pipeline
def text_clean(corpus, keep_list):
    '''
    Purpose : Function to keep only alphabets, digits and certain words (punctuations, qmarks, tabs etc. removed)
    
    Input : Takes a text corpus, 'corpus' to be cleaned along with a list of words, 'keep_list', which have to be retained
            even after the cleaning process
    
    Output : Returns the cleaned text corpus
    
    '''
    cleaned_corpus = pd.Series()
    for row in corpus:
        qs = []
        for word in row.split():
            if word not in keep_list:
                p1 = re.sub(pattern='[^a-zA-Z0-9]',repl=' ',string=word)
                p1 = p1.lower()
                qs.append(p1)
            else : qs.append(word)
        cleaned_corpus = cleaned_corpus.append(pd.Series(' '.join(qs)))
    return cleaned_corpus

def stopwords_removal(corpus):
    wh_words = ['who', 'what', 'when', 'why', 'how', 'which', 'where', 'whom']
    stop = set(stopwords.words('english'))
    for word in wh_words:
        stop.remove(word)
    corpus = [[x for x in x.split() if x not in stop] for x in corpus]
    return corpus

def lemmatize(corpus):
    lem = WordNetLemmatizer()
    corpus = [[lem.lemmatize(x, pos = 'v') for x in x] for x in corpus]
    return corpus

def stem(corpus, stem_type = None):
    if stem_type == 'snowball':
        stemmer = SnowballStemmer(language = 'english')
        corpus = [[stemmer.stem(x) for x in x] for x in corpus]
    else :
        stemmer = PorterStemmer()
        corpus = [[stemmer.stem(x) for x in x] for x in corpus]
    return corpus

def preprocess(corpus, keep_list, cleaning = True, stemming = False, stem_type = None, lemmatization = False, remove_stopwords = True):
    '''
    Purpose : Function to perform all pre-processing tasks (cleaning, stemming, lemmatization, stopwords removal etc.)
    
    Input : 
    'corpus' - Text corpus on which pre-processing tasks will be performed
    'keep_list' - List of words to be retained during cleaning process
    'cleaning', 'stemming', 'lemmatization', 'remove_stopwords' - Boolean variables indicating whether a particular task should 
                                                                  be performed or not
    'stem_type' - Choose between Porter stemmer or Snowball(Porter2) stemmer. Default is "None", which corresponds to Porter
                  Stemmer. 'snowball' corresponds to Snowball Stemmer
    
    Note : Either stemming or lemmatization should be used. There's no benefit of using both of them together
    
    Output : Returns the processed text corpus
    
    '''
    
    if cleaning == True:
        corpus = text_clean(corpus, keep_list)
    
    if remove_stopwords == True:
        corpus = stopwords_removal(corpus)
    else :
        corpus = [[x for x in x.split()] for x in corpus]
    
    if lemmatization == True:
        corpus = lemmatize(corpus)
        
        
    if stemming == True:
        corpus = stem(corpus, stem_type)
    
    corpus = [' '.join(x) for x in corpus]        

    return corpus

common_dot_words = ['books', 'literary']

# Preprocessing with Lemmatization here
preprocessed_corpus = preprocess(corpus, keep_list = common_dot_words, 
                                 stemming = False, stem_type = None,
                                lemmatization = True, remove_stopwords = True)


# Building a BoW model using CountVectorizer 
vectorizer = CountVectorizer()
bow_matrix = vectorizer.fit_transform(preprocessed_corpus)

# Obtaining the features corresponding to the BoW model
feature_names = vectorizer.get_feature_names()
feature_array = bow_matrix.toarray()
print(bow_matrix.toarray().shape) # no of sentences (documents) * len of vocab

# Including bigrams and trigrams
vectorizer_ngram_range = CountVectorizer(analyzer='word', ngram_range=(1,3))
bow_matrix_ngram = vectorizer_ngram_range.fit_transform(preprocessed_corpus)
n_gram_feature_names = (vectorizer_ngram_range.get_feature_names())
n_gram_feature_array = bow_matrix_ngram.toarray()

# Specifying Max Features
# Limiting the BoW model to the number of features specified my max_features
vectorizer_max_features = CountVectorizer(analyzer='word', 
                                          ngram_range=(1,3), 
                                          max_features = 50)
bow_matrix_max_features = vectorizer_max_features.fit_transform(preprocessed_corpus)
n_gram_max_feature_names = vectorizer_max_features.get_feature_names()
n_gram_max_feature_array = bow_matrix_max_features.toarray()

# Thresholding using Max_df and Min_df
# Max_df will not consider frequent words in a document above the specified threshold
# Min_df will not consider rare words in a document below the specified threshold
vectorizer_max_features = CountVectorizer(analyzer='word', 
                                          ngram_range=(1,3), 
                                          max_df = 3, 
                                          min_df = 2)
bow_matrix_max_features = vectorizer_max_features.fit_transform(preprocessed_corpus)
n_gram_thresh_feature_names = vectorizer_max_features.get_feature_names()
n_gram_thresh_feature_array = bow_matrix_max_features.toarray()