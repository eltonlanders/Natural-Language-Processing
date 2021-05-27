#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 27 20:21:53 2021

@author: elton
"""
!pip install transformers==4.5.0.
!pip install torch==1.8.0.
from transformers import BertForNextSentencePrediction, BertTokenizer 
from torch.nn.functional import softmax



# Downloading and loading the pretrained BERTje model and tokenizer
model = BertForNextSentencePrediction.from_pretrained("GroNLP/bert-base-dutch-cased")
tokenizer = BertTokenizer.from_pretrained("GroNLP/bert-base-dutch-cased")

# Input sentence pairs in Dutch language
sentence_A = 'Ik woon in Amsterdam'
sentence_B = 'Een geweldige plek'

# Encoding the sentences
embeddings = tokenizer(sentence_A, sentence_B, return_tensors='pt')

print(embeddings)

logits = model(**embeddings)[0]

# Converting logits to probabilities
probs = softmax(logits, dim=1)

# Index 0 and 1 represent isnext and notnext respectively
print(probs)
"""
tensor([[0.5751, 0.4249]], grad_fn=<SoftmaxBackward>)
"""



sentence_A = 'Het is zonnig'
sentence_B = 'Mooie tuin'

embeddings = tokenizer.encode_plus(sentence_A, text_pair=sentence_B, 
                                   return_tensors='pt')

logits = model(**embeddings)[0]

probs = softmax(logits, dim=1)

print(probs) 
"""
tensor([[0.5440, 0.4560]], grad_fn=<SoftmaxBackward>)

Since index 0 is higher it represents that the second sentence is the next 
sentences after the first.
"""
"""