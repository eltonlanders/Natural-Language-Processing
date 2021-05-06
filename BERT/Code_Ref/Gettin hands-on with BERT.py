# -*- coding: utf-8 -*-
"""
Created on Wed May  5 11:58:06 2021

@author: elton
"""

from transformers import BertModel, BertTokenizer, BertForQuestionAnswering
from transformers import AutoTokenizer, AutoModel # for pytorch
import torch
from transformers import BertForSequenceClassification, BertTokenizerFast 
from transformers import Trainer, TrainingArguments
from nlp import load_dataset
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
# from transformers import AutoTokenizer, TFAutoModel # for tensorflow
# Refer https://huggingface.co/transformers/index.html for the documentation



# GENERATING BERT EMBEDDINGS
# Using AutoTokenizer and AutoModel
model = AutoModel.from_pretrained('bert-base-uncased')
tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')

inputs = tokenizer("Hello world!", return_tensors="pt")
print(encoded_input) 
tokenizer.decode(encoded_input["input_ids"])

outputs = model(**inputs)

# Using BertModel and BertTokenizer
# Manually preprocessing the input
model = BertModel.from_pretrained('bert-base-uncased')
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

sentence = 'I love Paris'
tokens = tokenizer.tokenize(sentence)

# Adding the [CLS] and [SEP] tokens
tokens = ['[CLS]'] + tokens + ['[SEP]']
print(" Tokens are \n {} ".format(tokens))

# Adding 2 [PAD] tokens to make the length 7 for example
tokens = tokens + ['[PAD]'] + ['[PAD]']

# Creating the attention mask. Value set to 1 if not a [PAD] token
attention_mask = [1 if i!= '[PAD]' else 0 for i in tokens]

# seg_ids=[0 for _ in range(len(padded_tokens))]
# print("Segment Tokens are \n {}".format(seg_ids))

# Converting all the tokens to their token IDs 
# Each token is mapped to a unique token ID
token_ids = tokenizer.convert_tokens_to_ids(tokens)

# Converting token_ids and attention_mask to tensors
token_ids = torch.tensor(token_ids).unsqueeze(0)
attention_mask = torch.tensor(attention_mask).unsqueeze(0)

# Getting the embedding
# hidden_rep consists of the embedding or representation of all the tokens obtained from the final encoder
# cls_head consists of the representation of the [CLS] token. It holds the
# aggregate representation of the sentence
hidden_rep, cls_head = model(token_ids, attention_mask = attention_mask, 
                             return_dict=False) 
hidden_rep.shape
"""
By using return dict=False you now get the output as a tuple, but it is not 
recommended. 
It is recommend to always use return_dict=True so that the outputs can be 
retrieved unambiguously from the dictionary returned by the model. Then use the
dictionary keys to get the attribute values.
output = model(token_ids, attention_mask = attention_mask)
output['last_hidden_state']
"""



# EXTRACTING EMBEDDINGS FROM ALL ENCODER LAYERS OF BERT
# Extracting the embeddings
# Setting output_hidden_states = True obtains encoding from all the hidden layers
model = BertModel.from_pretrained('bert-base-uncased', output_hidden_states = True)
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Preprocessing the input
sentence = 'I love Paris'
tokens = tokenizer.tokenize(sentence)
tokens = ['[CLS]'] + tokens + ['[SEP]']
tokens = tokens + ['[PAD]'] + ['[PAD]']
attention_mask = [1 if i!= '[PAD]' else 0 for i in tokens]
token_ids = tokenizer.convert_tokens_to_ids(tokens)
token_ids = torch.tensor(token_ids).unsqueeze(0)
attention_mask = torch.tensor(attention_mask).unsqueeze(0)

# Getting the embeddings
"""
"last_hidden_state" contains the representation of all the tokens obtained only
from the final encoder layer (here encoder 12)
"pooler_output" indicates the representation of the [CLS] token from the final encoder layer, which is further processed by a linear and tanh activation 
function
"hidden_states" contains the representation of all the tokens obtained from all 
the encoder layers
"""
last_hidden_state, pooler_output, hidden_states = model(token_ids, 
                                        attention_mask = attention_mask, 
                                        return_dict=False) 
last_hidden_state.shape # [batch_size, sequence_length, hidden_size]
# last_hidden_state[0][0]
pooler_output.shape
len(hidden_states)
hidden_states[0].shape # representation of tokens of the first embedding layer
hidden_states[1].shape # representation of tokens of the second embedding layer



# FINE-TUNING BERT FOR DOWNSTREAM TASKS
# Fine-tuning BERT for sentiment analysis
dataset = load_dataset('csv', data_files='imdbs.csv', split='train')
dataset = dataset.train_test_split(test_size=0.3)
train_set = dataset['train'] 
test_set = dataset['test']

model = BertForSequenceClassification.from_pretrained('bert-base-uncased')
tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')

tokenizer('I love Paris')
tokenizer(['I love Paris', 'birds fly','snow fall'], 
          padding = True, max_length=5)

# A function tpo preprocess the dataset
def preprocess(data):
    return tokenizer(data['text'], padding=True, truncation=True)

train_set = train_set.map(preprocess, batched=True, batch_size=len(train_set))
test_set = test_set.map(preprocess, batched=True, batch_size=len(test_set))

# Selecting the columns we need in our dataset and formatting them
train_set.set_format('torch', columns=['input_ids', 'attention_mask', 'label'])
test_set.set_format('torch', columns=['input_ids', 'attention_mask', 'label'])

# Training the model
batch_size = 8
epochs = 2

warmup_steps = 500
weight_decay = 0.01

training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=epochs,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    warmup_steps=warmup_steps,
    weight_decay=weight_decay,
    #evaluate_during_training=True,
    logging_dir='./logs',
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_set,
    eval_dataset=test_set
)

trainer.train()
trainer.evaluate()

# Question-answering
# Performing question-answering with fine-tuned BERT
model = BertForQuestionAnswering.from_pretrained('bert-large-uncased-whole-word-masking-finetuned-squad')
tokenizer = BertTokenizer.from_pretrained('bert-large-uncased-whole-word-masking-finetuned-squad')

question = "What is the immune system?"
paragraph = "The immune system is a system of many biological structures and \
 processes within an organism that protects against disease. To function properly,\
 an immune system must detect a wide variety of agents, known as pathogens, from \
 viruses to parasitic worms, and distinguish them from the organism's own healthy tissue."

question = '[CLS] ' + question + '[SEP]'
paragraph = paragraph + '[SEP]'

question_tokens = tokenizer.tokenize(question)
paragraph_tokens = tokenizer.tokenize(paragraph)

tokens = question_tokens + paragraph_tokens 
input_ids = tokenizer.convert_tokens_to_ids(tokens)

segment_ids = [0] * len(question_tokens)
segment_ids += [1] * len(paragraph_tokens)

input_ids = torch.tensor([input_ids])
segment_ids = torch.tensor([segment_ids])

start_scores, end_scores = model(input_ids, token_type_ids = segment_ids)

start_index = torch.argmax(start_scores)
end_index = torch.argmax(end_scores)

print(' '.join(tokens[start_index:end_index+1]))
