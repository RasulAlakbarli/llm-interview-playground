import numpy as np
import pandas as pd
import torch.nn as nn
from torch.utils.data import DataLoader

class IMDB_Model(nn.Module):
    def __init__(self, vocab_size, embedding_dim=64):
        super().__init__()
        self.embedding = nn.Embedding(num_embeddings=vocab_size, embedding_dim=embedding_dim, padding_idx=0)
        self.dropout = nn.Dropout(0.2)
        self.linear = nn.Sequential(
            nn.Linear(embedding_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(128, 1)
        )      

    def forward(self, x):
        x = self.embedding(x) # Shape: [batch_size, max_seq_len, embedding_dim]
        x = x.sum(1)
        x = self.dropout(x)
        x = self.linear(x)
        return x
        