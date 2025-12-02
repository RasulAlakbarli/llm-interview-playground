import numpy as np
from collections import defaultdict
from data_processing import preprocess_data

class N_gram:
    def __init__(self, text_dir: str):
        """ 
        N-gram model class
        
        Args:
            text_dir (str): path to text file
            N (int): number of grams
        """
        self.text_corpus = open(text_dir, "r").readlines()
        self.sentence_ids, self.vocab = preprocess_data(self.text_corpus, padding=False)
        self.BOS_TOKEN = self.vocab["<BOS>"]
        self.EOS_TOKEN = self.vocab["<EOS>"]
        
    def fit(self, N):
        """
        Create an ngram table from our data
        """
        self.N = N
        self.ngrams_table = defaultdict(dict)
        
        for sentence in self.sentence_ids:
            sentence = [self.BOS_TOKEN]*max(0, self.N-1) + sentence
            for i in range(len(sentence)):
                ngram = sentence[i:i+self.N]

                if len(ngram) == self.N:
                    if self.ngrams_table[tuple(ngram[:self.N-1])] == {}:
                        self.ngrams_table[tuple(ngram[:self.N-1])][ngram[self.N-1]] = 1
                    else:
                        if ngram[self.N-1] in self.ngrams_table[tuple(ngram[:self.N-1])]:
                            self.ngrams_table[tuple(ngram[:self.N-1])][ngram[self.N-1]] += 1
                        else:
                            self.ngrams_table[tuple(ngram[:self.N-1])][ngram[self.N-1]] = 1

        self.p_table = defaultdict(dict)
        for k, v in self.ngrams_table.items():
            total_count = sum(v.values())
            self.p_table[k] = {word: count/total_count for word, count in v.items()}
    
    def idx_2_words(self, generated_list:list):
        words = []
        id2word = {v: k for k, v in self.vocab.items()}
        for id in generated_list:
            if id == self.BOS_TOKEN:
                continue
            words.append(id2word[id])
        return " ".join(words)
            
    def generate(self, text:str = None, max_len: int = 50):
        """
        Generate text based on the n-gram model

        Args:
            text (str, optional): Text input that you want model to complete. Defaults to None.
            max_len (int, optional): Maximum length of generated text. Defaults to 50.
        """ 
        if not text:
            input_seq = tuple([self.BOS_TOKEN] * max(0, self.N-1))
        if text:
            input_seq = text.split()[-self.N+1:]
            input_seq = tuple([self.vocab[i] for i in input_seq])
        
        generated = list(input_seq)
        while len(generated) <= max_len:  
            sub_ = self.p_table[input_seq]
            if not sub_:
                return ""
            words = list(sub_.keys())
            probabilities = list(sub_.values())
            next_token = np.random.choice(words, p=probabilities)
            input_seq = tuple(list(input_seq[1:])+[next_token])
            if next_token == self.EOS_TOKEN:
                break
            generated.append(next_token)
        
        return self.idx_2_words(generated)
            
    
if __name__ == "__main__":
    ngram = N_gram(text_dir="data/quotes.txt")
    ngram.fit(N=6)
    for i in range(5):
        print(ngram.generate())
        print()
        
# Sample output:
# 1. it does n't matter what happens to you , but you can control the way you think about all the events . you always have a choice . you can choose to face them with a positive mental attitude .
# 2. motivation does n't work unless your determination is up to the mark
# 3. truth never damages a cause that is just .
# 4. understanding who we are , how god created us , how we grow , and how we give those gifts back to others is core work of the spiritual entrepreneur .
# 5. by changing ourselves , we change our lives .