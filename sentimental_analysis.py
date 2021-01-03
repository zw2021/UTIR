from string import punctuation
from collections import Counter
from torch.utils.data import DataLoader
from torch.utils.data.dataset import random_split

import numpy as np
#import torch as nn
import pandas as pd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt

import glob
import os
import torch
import time

#== functions
def generate_batch(batch):  # batch function is for offsets and generating data batches; recommended to have from pytorch tutorial
    label = torch.tensor([entry[0] for entry in batch])
    text = [entry[1] for entry in batch]
    offsets = [0] + [len(entry) for entry in text]
    offsets = torch.tensor(offsets[:-1]).cumsum(dim=0)
    text = torch.cat(text)
    return text, offsets, label

def train_func(sub_train_):

    # Train the model
    train_loss = 0
    train_acc = 0
    data = DataLoader(sub_train_, batch_size=BATCH_SIZE, shuffle=True,
                      collate_fn=generate_batch)
    for i, (text, offsets, cls) in enumerate(data):
        optimizer.zero_grad()
        text, offsets, cls = text.to(device), offsets.to(device), cls.to(device)
        output = model(text, offsets)
        loss = criterion(output, cls)
        train_loss += loss.item()
        loss.backward()
        optimizer.step()
        train_acc += (output.argmax(1) == cls).sum().item()

    # Adjust the learning rate
    scheduler.step()

    return train_loss / len(sub_train_), train_acc / len(sub_train_)

def test(data_):
    loss = 0
    acc = 0
    data = DataLoader(data_, batch_size=BATCH_SIZE, collate_fn=generate_batch)
    for text, offsets, cls in data:
        text, offsets, cls = text.to(device), offsets.to(device), cls.to(device)
        with torch.no_grad():
            output = model(text, offsets)
            loss = criterion(output, cls)
            loss += loss.item()
            acc += (output.argmax(1) == cls).sum().item()

    return loss / len(data_), acc / len(data_)
#== classes
class LSTMTagger(nn.Module):

    def __init__(self, embedding_dim, hidden_dim, vocab_size):
        super(LSTMTagger, self).__init__()
        self.hidden_dim = hidden_dim

        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)

        # The LSTM takes word embeddings as inputs, and outputs hidden states
        # with dimensionality hidden_dim.
        self.lstm = nn.LSTM(embedding_dim, hidden_dim)


    def forward(self, sentence):
        embeds = self.word_embeddings(sentence)
        lstm_out, _ = self.lstm(embeds.view(len(sentence), 1, -1))
        tag_space = self.hidden2tag(lstm_out.view(len(sentence), -1))
        tag_scores = F.log_softmax(tag_space, dim=1)
        return tag_scores

#== extract files from tarfile
# tar = tarfile.open("aclImdb_v1.tar.gz")
# tar.extractall("./text")
# tar.close()

#== Start Building Model
#== tokenize text
# read data from imbd text files
mydir = "C:/Users/huang/Documents/pythonUTIR/text/aclImdb"
myfile = 'imdb_text.txt'
words = os.path.join(mydir, myfile)
imbd_words = open(words, encoding="cp437").read()

reviews = []
path = "C:/Users/huang/Documents/pythonUTIR/text/aclImdb/test/neg/*"

for file in glob.glob(path):
    with open(file, encoding="cp437") as f:
        neg_text = f.read()

    neg_text = neg_text.lower()
    neg_text = ''.join([c for c in neg_text if c not in punctuation])
    reviews.append(neg_text)

all_text2 = ' '.join(reviews)
words = all_text2.split()
count_words = Counter(words)

total_words = len(words)
sorted_words = count_words.most_common(total_words)
vocab_to_int = {w:i+1 for i, (w,c) in enumerate(sorted_words)}  # vocab to int mapping dictionary

# Analyzing Length of Reviews
reviews_int = []
for review in reviews:
    r = [vocab_to_int[w] for w in review.split()]
    reviews_int.append(r)

reviews_len = [len(x) for x in reviews_int]
pd.Series(reviews_len).hist()
plt.title('Histogram for Lengths of Reviews')
#print(pd.Series(reviews_len).describe())
plt.show()

#== call model
EMBEDDING_DIM = 32
HIDDEN_DIM = 32
BATCH_SIZE = 16 # found from pytorch tutorial
model = LSTMTagger(EMBEDDING_DIM, HIDDEN_DIM, len(reviews_int))
loss_function = nn.NLLLoss()
optimizer = optim.SGD(model.parameters(), lr=0.1)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


N_EPOCHS = 1
min_valid_loss = float('inf')

criterion = torch.nn.CrossEntropyLoss().to(device)
optimizer = torch.optim.SGD(model.parameters(), lr=4.0)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1, gamma=0.9)

train_len = int(len(reviews_int) * 0.95)

sub_train_, sub_valid_ = \
    random_split(reviews, [train_len, len(reviews_int) - train_len])
for epoch in range(N_EPOCHS):

    start_time = time.time()
    train_loss, train_acc = train_func(sub_train_)
    valid_loss, valid_acc = test(sub_valid_)

    secs = int(time.time() - start_time)
    mins = secs / 60
    secs = secs % 60

    print('Epoch: %d' %(epoch + 1), " | time in %d minutes, %d seconds" %(mins, secs))
    print(f'\tLoss: {train_loss:.4f}(train)\t|\tAcc: {train_acc * 100:.1f}%(train)')
    print(f'\tLoss: {valid_loss:.4f}(valid)\t|\tAcc: {valid_acc * 100:.1f}%(valid)')
