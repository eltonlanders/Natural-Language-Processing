# -*- coding: utf-8 -*-
"""
Created on Thu May 13 18:30:38 2021

@author: elton
"""

import tensorflow as tf
import os 
import io
tf.__version__
import re
import stanza
#en = stanza.download('en') 
#en = stanza.Pipeline(lang='en')
import stopwordsiso as stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from keras.utils import np_utils
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from keras.utils import np_utils
import pandas as pd 



# Test data reading
lines = io.open('smsspamcollection/SMSSpamCollection').read().strip().split('\n')
lines[0]

# Pre-process data
spam_dataset = []
count = 0
for line in lines:
  label, text = line.split('\t')
  if label.lower().strip() == 'spam':
    spam_dataset.append((1, text.strip()))
    count += 1
  else:
    spam_dataset.append(((0, text.strip())))

# Data normalization
df = pd.DataFrame(spam_dataset, columns=['Spam', 'Message'])

def make_model(input_dims=3, num_units=12):
  model = tf.keras.Sequential()

  # Adds a densely-connected layer with 12 units to the model:
  model.add(tf.keras.layers.Dense(num_units, input_dim=input_dims, 
                                  activation='relu'))

  # Add a sigmoid layer with a binary output unit:
  model.add(tf.keras.layers.Dense(1, activation='sigmoid'))
  model.compile(loss='binary_crossentropy', optimizer='adam', 
                metrics=['accuracy'])
  return model

train=df.sample(frac=0.8, random_state=42)
test=df.drop(train.index)


tfidf = TfidfVectorizer(binary=True)
X = tfidf.fit_transform(train['Message']).astype('float32')
X_test = tfidf.transform(test['Message']).astype('float32')

_, cols = X.shape
model2 = make_model(cols)  # to match tf-idf dimensions

y_train = train['Spam']
y_test = test['Spam']


lb = LabelEncoder()
y = lb.fit_transform(y_train)
dummy_y_train = np_utils.to_categorical(y)
model2.fit(X.toarray(), y_train, epochs=10, batch_size=10)

model2.evaluate(X_test.toarray(), y_test)


