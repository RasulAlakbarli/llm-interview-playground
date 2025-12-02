import re
from numpy import float32
import torch
import pandas as pd
from torch.utils.data import Dataset

class IMDB_Dataset(Dataset):
	"""
	IMDB Dataset for sentiment analysis.
	"""
	def __init__(self, dataset: pd.DataFrame, vocab: dict = None, max_len: int = None):
		super().__init__()
		self.dataset = dataset
		self.dataset['tokens'] = self.dataset['review'].apply(self.clean_str)
        
		self.PAD_TOKEN = "<PAD>"
		self.UNK_TOKEN = "<UNK>"
		self.BOS_TOKEN = "<BOS>"
		self.EOS_TOKEN = "<EOS>"
		if vocab is None:
			# We create a vocabulary and assign a unique id to each word
			self.vocab = {self.PAD_TOKEN:0, self.UNK_TOKEN:1, self.BOS_TOKEN:2, self.EOS_TOKEN:3}
			self.max_len = 0
			for index, row in self.dataset.iterrows():
				sentence = row['tokens']
				self.max_len = max(self.max_len, len(sentence)+2)
				for word in sentence:
					if word not in self.vocab:
						self.vocab[word] = len(self.vocab)
		else:
			self.vocab = vocab
			self.max_len = max_len
     
	def clean_str(self, string, tolower=True):
		"""
		Tokenization/string cleaning.
		Original taken from https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py
		"""
		string = re.sub(r"[^A-Za-z0-9(),.!?\'\`]", " ", string)
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
		string = re.sub(r"\.", " . ", string)

		if tolower:
			string = string.lower()
		return string.split()

	def __len__(self):
		return len(self.dataset)
    
	def __getitem__(self, index):
		row = self.dataset.iloc[index]
		text, label, tokens = row["review"], row["label"], row['tokens']
		sentence_indices = [self.vocab[self.BOS_TOKEN]]
		if len(sentence_indices) > self.max_len:
			sentence_indices = sentence_indices[:self.max_len]
		for token in tokens:
			if token in self.vocab:
				sentence_indices += [self.vocab[token]]
			else:
				sentence_indices += [self.vocab[self.UNK_TOKEN]]
		sentence_indices += [self.vocab[self.EOS_TOKEN]]
		sentence_indices += [self.vocab[self.PAD_TOKEN]] * (self.max_len - len(sentence_indices))
		return torch.tensor(sentence_indices), torch.tensor(label, dtype=torch.float32)