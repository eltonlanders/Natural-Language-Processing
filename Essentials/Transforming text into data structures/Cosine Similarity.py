# -*- coding: utf-8 -*-
"""
Created on Thu May 20 20:32:38 2021

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

# Preprocessing with Lemmatization
preprocessed_corpus = preprocess(corpus, keep_list = common_dot_words, 
                                 stemming = False, stem_type = None,
                                lemmatization = True, remove_stopwords = True)


# Cosine Similarity Calculation
def cosine_similarity(vector1, vector2):
    vector1 = np.array(vector1)
    vector2 = np.array(vector2)
    return np.dot(vector1, vector2) / (np.sqrt(np.sum(vector1**2)) * np.sqrt(np.sum(vector2**2)))


# Built using CountVectorizer
vectorizer = CountVectorizer()
bow_matrix = vectorizer.fit_transform(preprocessed_corpus)
feature_names_count = vectorizer.get_feature_names()
features_array_count = bow_matrix.toarray()

# Calculating cosine similarity between a document vector and all other document vectors in the corpus
for i in range(bow_matrix.shape[0]):
    for j in range(i + 1, bow_matrix.shape[0]):
        print("The cosine similarity between the documents ", i, "and", j, "is: ",
              cosine_similarity(bow_matrix.toarray()[i], bow_matrix.toarray()[j]))


# Built using TfidfVectorizer
vectorizer = TfidfVectorizer()
tf_idf_matrix = vectorizer.fit_transform(preprocessed_corpus)
feature_names_tfidf = vectorizer.get_feature_names()
features_array_tfidf = tf_idf_matrix.toarray()
print("\nThe shape of the TF-IDF matrix is: ", tf_idf_matrix.shape)

# Calculating cosine similarity between a document vector and all other document vectors in the corpus
for i in range(tf_idf_matrix.shape[0]):
    for j in range(i + 1, tf_idf_matrix.shape[0]):
        print("The cosine similarity between the documents ", i, "and", j, "is: ",
              cosine_similarity(tf_idf_matrix.toarray()[i], tf_idf_matrix.toarray()[j]))
