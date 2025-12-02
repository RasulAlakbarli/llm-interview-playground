import tqdm
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader

from mlp.dataset import IMDB_Dataset
from mlp.model import IMDB_Model

# Define train and test datasets
data = pd.read_csv("data/imdb_data.csv", nrows=10000)
trainning_db = IMDB_Dataset(data.iloc[:8000])
val_db = IMDB_Dataset(data.iloc[8000:], vocab=trainning_db.vocab, max_len=trainning_db.max_len)

train_loader = DataLoader(trainning_db, batch_size=8, shuffle=True)
val_loader = DataLoader(val_db, batch_size=8)

# Define the model
device = (torch.device("cuda") if torch.cuda.is_available() else "cpu")
model = IMDB_Model(vocab_size=len(trainning_db.vocab), embedding_dim=64)
model.to(device)
criterion = nn.BCEWithLogitsLoss()
optim = AdamW(model.parameters(), lr=0.0001)

# Training loop
epochs = 25

train_loss = []
val_loss = []
val_acc = []
for epoch in range(epochs):
    # Train loop
    model.train()
    train_batch_loss = []
    for train_batch in train_loader:
        x, label = train_batch
        x = x.to(device)
        label = label.unsqueeze(1).float().to(device)
        optim.zero_grad()
        pred = model(x)
        loss = criterion(pred, label)
        loss.backward()
        optim.step()
        train_batch_loss.append(loss.item())
    
    # Validation step
    model.eval()
    val_batch_loss = []
    val_batch_acc = []
    with torch.no_grad():
        for val_batch in val_loader:
            x, label = val_batch
            x = x.to(device)
            label = label.unsqueeze(1).float().to(device)
            pred = model(x)
            loss = criterion(pred, label)
            val_batch_loss.append(loss.item())
            # Compute accuracy
            sigmoid = nn.Sigmoid()
            preds = (sigmoid(pred) > 0.5).float()
            correct = (preds == label).float().mean()
            val_batch_acc.append(correct.item())
            
    print(f"Epoch {epoch}/{epochs} - Train Loss: {np.mean(train_batch_loss):.4f} | Dev Loss: {np.mean(val_batch_loss):.4f} | Dev Acc: {np.mean(val_batch_acc):.4f}")
            
    train_loss.append(np.mean(train_batch_loss))
    val_loss.append(np.mean(val_batch_loss))
    val_acc.append(np.mean(val_batch_acc))
    
    
def plot_metrics(train_loss, val_loss, val_acc):
	# Plotting loss and accuracy curves
	plt.figure(figsize=(12, 5))
	plt.subplot(1, 2, 1)
	plt.plot(train_loss, label='Train Loss')
	plt.plot(val_loss, label='Validation Loss')
	plt.xlabel('Epochs')
	plt.ylabel('Loss')
	plt.legend()
	plt.subplot(1, 2, 2)
	plt.plot(val_acc, label='Validation Accuracy')
	plt.xlabel('Epochs')
	plt.ylabel('Accuracy')
	plt.legend()
	plt.show()
 
plot_metrics(train_loss, val_loss, val_acc)