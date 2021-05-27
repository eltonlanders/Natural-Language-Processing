#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 27 20:17:53 2021

@author: elton
"""
!pip install transformers==4.5.0
!pip install torch==1.8.0.
from transformers import pipeline



# Defining the masked word prediction pipeline
predict_mask = pipeline("fill-mask", 
                        model= "dccuchile/bert-base-spanish-wwm-uncased", 
                        tokenizer="dccuchile/bert-base-spanish-wwm-uncased")

# Using the Spanish sentence "todos los caminos llevan a roma" and masking the first word in the sentence with [MASK] token.
# Feeding the sentence to the pipeline with the word "todos" masked.
result = predict_mask('[MASK] los caminos llevan a Roma')

print(result)
# The pipeplie predicts the masked word correctly
"""
[{'sequence': 'todos los caminos llevan a roma', 'score': 0.9719983339309692, 
  'token': 1399, 'token_str': 'todos'}, 
 {'sequence': 'todas los caminos llevan a roma', 'score': 0.007171058561652899, 
  'token': 1825, 'token_str': 'todas'}, 
 {'sequence': '- los caminos llevan a roma', 'score': 0.0053520179353654385, 
  'token': 1139, 'token_str': '-'}, 
 {'sequence': 'todo los caminos llevan a roma', 'score': 0.004154067952185869,
  'token': 1300, 'token_str': 'todo'}, 
 {'sequence': 'y los caminos llevan a roma', 'score': 0.003964297007769346, 
  'token': 1040, 'token_str': 'y'}]

"""


# Lets take another sentence in Spanish and mask a random word.
spanish_sentence = "Mucho tiempo no veo a mi amigo"
result = predict_mask("Mucho [MASK] no veo a mi amigo")

print(result)
"""
[{'sequence': 'mucho tiempo no veo a mi amigo', 'score': 0.7119235992431641, 
  'token': 1526, 'token_str': 'tiempo'}, 
 {'sequence': 'mucho que no veo a mi amigo', 'score': 0.15607716143131256, 
  'token': 1041, 'token_str': 'que'}, 
 {'sequence': 'mucho ya no veo a mi amigo', 'score': 0.04137589782476425, 
  'token': 1319, 'token_str': 'ya'}, 
 {'sequence': 'mucho mas no veo a mi amigo', 'score': 0.03950831666588783, 
  'token': 2062, 'token_str': 'mas'}, {'sequence': 'mucho, no veo a mi amigo', 
   'score': 0.012160849757492542, 'token': 1019, 'token_str': ','}]

"""