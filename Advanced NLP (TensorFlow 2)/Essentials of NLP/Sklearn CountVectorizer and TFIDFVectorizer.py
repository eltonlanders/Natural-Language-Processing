# -*- coding: utf-8 -*-
"""
Created on Thu May 13 13:02:11 2021

@author: elton
"""

from sklearn.feature_extraction.text import TfidfVectorizer 
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd


# Test corpus
corpus = [
    '''To Sherlock Holmes she is always _the_ woman. I have seldom heard him
mention her under any other name. In his eyes she eclipses and
predominates the whole of her sex. It was not that he felt any emotion
akin to love for Irene Adler. All emotions, and that one particularly,
were abhorrent to his cold, precise but admirably balanced mind. He
was, I take it, the most perfect reasoning and observing machine that
the world has seen, but as a lover he would have placed himself in a
false position. He never spoke of the softer passions, save with a gibe
and a sneer. They were admirable things for the observer—excellent for
drawing the veil from men’s motives and actions. But for the trained
reasoner to admit such intrusions into his own delicate and finely
adjusted temperament was to introduce a distracting factor which might
throw a doubt upon all his mental results. Grit in a sensitive
instrument, or a crack in one of his own high-power lenses, would not
be more disturbing than a strong emotion in a nature such as his. And
yet there was but one woman to him, and that woman was the late Irene
Adler, of dubious and questionable memory.''',
    ''' I had called upon my friend, Mr. Sherlock Holmes, one day in the
 autumn of last year and found him in deep conversation with a very
 stout, florid-faced, elderly gentleman with fiery red hair. With an
 apology for my intrusion, I was about to withdraw when Holmes pulled
 me abruptly into the room and closed the door behind me.'''
    ]

# CountVectorizer
count_vect = CountVectorizer(stop_words='english', analyzer='word')

X_c = count_vect.fit_transform(corpus)

count_tokens = count_vect.get_feature_names()

df_countvect = pd.DataFrame(data = X_c.toarray(), 
                            index =['Doc1','Doc2'], columns = count_tokens)

cos_sim = cosine_similarity(X_c.toarray())


# TFIDFVectorizer
tfidf_vect = TfidfVectorizer(stop_words='english', 
                             analyzer='word')

X_t = tfidf_vect.fit_transform(corpus)

tfidf_tokens = tfidf_vect.get_feature_names()

df_tfvect = pd.DataFrame(data = X_t.toarray(), 
                            index =['Doc1','Doc2'], columns = tfidf_tokens)