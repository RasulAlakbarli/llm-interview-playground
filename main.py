import re
import numpy as np
import matplotlib.pyplot as plt
import torch

##### STEP 1: Processing our input #####

text_corpus = ["Hello my transformer friend!",
               "Hi my name is Rasul.",
               "I am running a test to learn transformer model.",
               "Apples are very tasty and good for health!"]

def clean_str(string, tolower=True):
	"""
	Tokenization/string cleaning.
	Original taken from https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py
	"""
	string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
	string = re.sub(r"\'s", " \'s", string)
	string = re.sub(r"\'ve", " \'ve", string)
	string = re.sub(r"n\'t", " n\'t", string)
	string = re.sub(r"\'re", " \'re", string)
	string = re.sub(r"\'d", " \'d", string)
	string = re.sub(r"\'ll", " \'ll", string)
	string = re.sub(r",", " , ", string)
	string = re.sub(r"!", " ! ", string)
	string = re.sub(r"\(", " \( ", string)
	string = re.sub(r"\)", " \) ", string)
	string = re.sub(r"\?", " \? ", string)
	string = re.sub(r"\s{2,}", " ", string)
	if tolower:
		string = string.lower()
	return string.split()

# tokenize each sentence
tokenized = [[token for token in clean_str(sent)] for sent in text_corpus]

# We create a vocabulary and assign a unique id to each word
PAD_TOKEN = "<PAD>"
UNK_TOKEN = "<UNK>"
vocab = {PAD_TOKEN:0, UNK_TOKEN:1}
for sentence in tokenized:
    for word in sentence:
        if word not in vocab:
            vocab[word] = len(vocab)
            
# Now we recreate the sentences with obtained ids
max_len = max(len(sentence) for sentence in tokenized)
tensor_data = []
for sentence in tokenized:
    sentence_indices = []
    for token in sentence:
        sentence_indices.append(vocab[token])
    if len(sentence_indices) < max_len:
        sentence_indices += [vocab[PAD_TOKEN]]*(max_len-len(sentence_indices))
    tensor_data.append(sentence_indices)

# We get our tensor with ids
tensor_data = torch.tensor(tensor_data, dtype=torch.int16)


##### STEP 2: Transformers model #####
class Transformers:
    def __init__(self):
        pass
    def self_attn(self):
        pass
    def multi_head_self_attn(self):
        pass
    def encoder(self):
        pass
    def decoder(self):
        pass
     