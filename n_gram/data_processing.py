import re

def clean_str(string, tolower=True):
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

def preprocess_data(text_corpus, padding=True):
    # tokenize each sentence
    tokenized = [[token for token in clean_str(sent)] for sent in text_corpus]
    
    # We create a vocabulary and assign a unique id to each word
    PAD_TOKEN = "<PAD>"
    UNK_TOKEN = "<UNK>"
    BOS_TOKEN = "<BOS>"
    EOS_TOKEN = "<EOS>"
    vocab = {PAD_TOKEN:0, UNK_TOKEN:1, BOS_TOKEN:2, EOS_TOKEN:3}
    for sentence in tokenized:
        for word in sentence:
            if word not in vocab:
                vocab[word] = len(vocab)
                
    # Now we recreate the sentences with obtained ids
    max_len = max(len(sentence) for sentence in tokenized) + 2 # Consider BOS and EOS
    corpus_indices = []
    for sentence in tokenized:
        sentence_indices = []
        sentence_indices.append(vocab[BOS_TOKEN])
        for token in sentence:
            sentence_indices.append(vocab[token])
        sentence_indices.append(vocab[EOS_TOKEN])
        if len(sentence_indices) < max_len and padding:
            sentence_indices += [vocab[PAD_TOKEN]]*(max_len-len(sentence_indices))
        corpus_indices.append(sentence_indices)
    
    # Return tensor and vocabulary
    return corpus_indices, vocab