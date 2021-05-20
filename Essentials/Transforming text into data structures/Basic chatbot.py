# -*- coding: utf-8 -*-
"""
Created on Thu May 20 17:38:15 2021

@author: elton
"""
import numpy as np
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer
import ast



# Loading questions and answers in separate lists 
questions = []
answers = [] 
with open('qa_Electronics.json','r') as f:
    for line in f:
        data = ast.literal_eval(line) # converting to dictionary
        questions.append(data['question'].lower())
        answers.append(data['answer'].lower())
        
# Tokenizing the text and converting the data in a matrix format
vectorizer = CountVectorizer(stop_words='english')
X_vec = vectorizer.fit_transform(questions)

# Transforming data by applying TF-IDF 
tfidf = TfidfTransformer() # by default applies "l2" normalization
X_tfidf = tfidf.fit_transform(X_vec)

# Function to calculate the best matching answer to the query from the corpus
def conversation(im):
    global tfidf, answers, X_tfidf
    Y_vec = vectorizer.transform(im)
    Y_tfidf = tfidf.fit_transform(Y_vec)
    cos_sim = np.rad2deg(np.arccos(max(cosine_similarity(Y_tfidf, X_tfidf)[0])))
    if cos_sim > 60 :
        return "sorry, I did not quite understand that"
    else:
        return answers[np.argmax(cosine_similarity(Y_tfidf, X_tfidf)[0])]

def main():
    usr = input("Please enter your username: ")
    print("support: Hi " + usr + ", welcome to Q&A support. How can I help you?")
    while True:
        im = input("{}: ".format(usr)) # get the input query
        if im.lower() == 'bye':
            print("Q&A support: bye!")
            break
        else:
            print("Q&A support: "+conversation([im]))
            
main()