# -*- coding: utf-8 -*-
"""
Created on Thu May  6 18:15:38 2021

@author: elton
"""
from transformers import BertModel, BertTokenizer, BertForQuestionAnswering
from transformers import AutoTokenizer, AutoModel
import torch
from transformers import BertForSequenceClassification, BertTokenizerFast 
from transformers import Trainer, TrainingArguments
from nlp import load_dataset
import numpy as np



# Using IMDB dataset for Sentiment Analysis
dataset = load_dataset('csv', data_files='imdbs.csv', split='train')
dataset = dataset.train_test_split(test_size=0.3)
train_set = dataset['train'] 
test_set = dataset['test']

model = BertForSequenceClassification.from_pretrained('bert-base-uncased')
tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')

# Preprocessing
def preprocess(data):
    return tokenizer(data['text'], padding=True, truncation=True)

train_set = train_set.map(preprocess, batched=True, batch_size=len(train_set))
test_set = test_set.map(preprocess, batched=True, batch_size=len(test_set))

# Selecting and formatting columns
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
    evaluate_during_training=True,
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