#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 27 12:58:53 2021

@author: elton
"""
import re
import nltk
import numpy as np
import pandas as pd
nltk.download('wordnet')
nltk.download('stopwords')
#!pip install torch==1.8.1
#!pip install transformers==4.6.1
from nltk.corpus import stopwords
#!pip install sentence-transformers==1.2.0
from nltk.stem.porter import PorterStemmer 
from nltk.stem.snowball import SnowballStemmer
from sentence_transformers import models, util
from nltk.stem.wordnet import WordNetLemmatizer
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer



from google.colab import drive
drive.mount('/content/drive')

# Loading data
data = pd.read_csv("/content/drive/MyDrive/Data/Text_Similarity_Dataset.csv")

list_1 = data['text1']
res = max(list_1, key=len) #length of maximum string
len(res)

list_2 = data['text2']
res = max(list_2, key=len)
len(res)

list_series_1 = pd.Series(list_1) 
list_series_2 = pd.Series(list_2)

# Function for data preprocessing
def text_clean(corpus):
    '''
    Purpose : Function to keep only alphabets, digits and certain words (punctuations, qmarks, tabs etc. removed)
    
    Input : Takes a text corpus, 'corpus' to be cleaned along with a list of words, 'keep_list', which have to be retained
            even after the cleaning process
    
    Output : Returns the cleaned text corpus
    
    '''
    cleaned_corpus = []
    for row in corpus:
        qs = []
        for word in row.split():
            p1 = re.sub(pattern='[^a-zA-Z0-9]',repl=' ',string=word)
            p1 = p1.lower()
            qs.append(p1)
        cleaned_corpus.append(' '.join(qs))
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

def preprocess(corpus, cleaning = True, stemming = False, stem_type = None, lemmatization = False, remove_stopwords = True):
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
        corpus = text_clean(corpus)
    
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



# Using a basic BERT model for the STS task
# Basic BERT model
word_embedding_model = models.Transformer('bert-base-uncased')

# Defining mean pooling
pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension(),
                               pooling_mode_mean_tokens=True,
                               pooling_mode_cls_token=False,
                               pooling_mode_max_tokens=False)

# Defining the Sentence transformer model from the word embedding model
model = SentenceTransformer(modules=[word_embedding_model, pooling_model])

# Getting the embeddings from the corpus (pair of documents)
embeddings_1 = model.encode(list_1, convert_to_tensor=True)
embeddings_2 = model.encode(list_2, convert_to_tensor=True)

# Computing the Cosine score between the embeddings
cos_score = util.pytorch_cos_sim(embeddings_1, embeddings_2)

# Adding the scores to a list
list = []
for i in range(len(list_1)):
    for j in range(len(list_2)):
        list.append(cos_score[i][j].item())

data_sub = data
data_sub.drop(columns=['text1' ,'text2'], inplace=True)
data_sub['Similarity_score'] = pd.Series(list)
data_sub.to_csv('roberta.csv', index=False)
#!cp roberta.csv "drive/My Drive/Data/"



# Using a BERT base model with mean tokens
# Defining the Sentence Transformer model 
list_1_preprocess = preprocess(list_series_1, cleaning=True, remove_stopwords=True)
list_2_preprocess = preprocess(list_series_2, cleaning=True, remove_stopwords=True)

model = SentenceTransformer('bert-base-nli-mean-tokens')

embeddings_1 = model.encode(list_1_preprocess, convert_to_tensor=True)
embeddings_2 = model.encode(list_2_preprocess, convert_to_tensor=True)

cos_score = util.pytorch_cos_sim(embeddings_1, embeddings_2)

list = []
for i in range(len(list_1)):
    for j in range(len(list_2)):
        list.append(cos_score[i][j].item())
        
data_sub = data
data_sub.drop(columns=['text1' ,'text2'], inplace=True)
data_sub['Similarity_score'] = pd.Series(list)
data_sub.to_csv('bert_mean_tokens.csv', index=False)
#!cp bert_mean_tokens.csv "drive/My Drive/Data/"



# Using RoBERTa model for the STS task
list_1_preprocess = preprocess(list_series_1, cleaning=True, remove_stopwords=True)
list_2_preprocess = preprocess(list_series_2, cleaning=True, remove_stopwords=True)

model = SentenceTransformer('stsb-roberta-large')

# Printing the current maximum sequence length of the model
print("Max Sequence Length:", model.max_seq_length)

# Changing the maximum sequence length to 512 (maximum supported for this model)
model.max_seq_length = 512
print(model.max_seq_length)

embeddings_1 = model.encode(list_1_preprocess, convert_to_tensor=True)
embeddings_2 = model.encode(list_2_preprocess, convert_to_tensor=True)

cos_score = util.pytorch_cos_sim(embeddings_1, embeddings_2)
cos_score[0][0].item()

list = []
for i in range(len(list_1)):
    for j in range(len(list_2)):
        list.append(cos_score[i][j].item())
        
data_sub = data
data_sub.drop(columns=['text1' ,'text2'], inplace=True)
data_sub['Similarity_score'] = pd.Series(list)
data_sub.to_csv('Roberta.csv', index=False)
#!cp Roberta.csv "drive/My Drive/Data/"



# Using a MPNET model for the STS task (MPNet with stopwords)
list_1_preprocess = preprocess(list_series_1, cleaning=True, remove_stopwords=False)
list_2_preprocess = preprocess(list_series_2, cleaning=True, remove_stopwords=False)

model = SentenceTransformer('stsb-mpnet-base-v2')

print("Max Sequence Length:", model.max_seq_length)

model.max_seq_length = 512
print(model.max_seq_length)

embeddings_1 = model.encode(list_1_preprocess, convert_to_tensor=True)
embeddings_2 = model.encode(list_2_preprocess, convert_to_tensor=True)

cos_score = util.pytorch_cos_sim(embeddings_1, embeddings_2)
cos_score[0][0].item()

list = []
for i in range(len(list_1)):
    for j in range(len(list_2)):
        list.append(cos_score[i][j].item())
        
data_sub = data
data_sub.drop(columns=['text1' ,'text2'], inplace=True)
data_sub['Similarity_score'] = pd.Series(list)
data_sub.to_csv('MPNet.csv', index=False)
#!cp MPNet.csv "drive/My Drive/Data/"



# Using a MPNET model for the STS task (MPNet without stopwords)
list_1_preprocess = preprocess(list_series_1, cleaning=True, remove_stopwords=True)
list_2_preprocess = preprocess(list_series_2, cleaning=True, remove_stopwords=True)

model = SentenceTransformer('stsb-mpnet-base-v2')

print("Max Sequence Length:", model.max_seq_length)

model.max_seq_length = 512
print(model.max_seq_length)

embeddings_1 = model.encode(list_1_preprocess, convert_to_tensor=True)
embeddings_2 = model.encode(list_2_preprocess, convert_to_tensor=True)

cos_score = util.pytorch_cos_sim(embeddings_1, embeddings_2)
cos_score[0][0].item()

list = []
for i in range(len(list_1)):
    for j in range(len(list_2)):
        list.append(cos_score[i][j].item())
        
data_sub = data
data_sub.drop(columns=['text1' ,'text2'], inplace=True)
data_sub['Similarity_score'] = pd.Series(list)
data_sub.to_csv('MPNet_without_stopwords.csv', index=False)
#!cp MPNet_without_stopwords.csv "drive/My Drive/Data/"
        



    