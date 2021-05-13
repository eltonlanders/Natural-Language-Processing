# -*- coding: utf-8 -*-
"""
Created on Thu May 13 12:23:43 2021

@author: elton
"""
# !pip install gensim==4.0.1
from gensim.models.word2vec import Word2Vec 
import gensim.downloader as api



# Downloading and loading the pretrained Word2Vec model
model_w2v = api.load("word2vec-google-news-300")

# Getting the top 10 similar words to the test word
model_w2v.most_similar("cookies",topn=10)

# Test 2
model_w2v.most_similar("ronaldo",topn=10)

# Test 3
model_w2v.most_similar("paris",topn=10)

# Finding the odd man out
model_w2v.doesnt_match(["USA","Canada","India","Tokyo"])

# Test 2
model_w2v.doesnt_match(["Uranium","Carbon","button","Graphite"])
