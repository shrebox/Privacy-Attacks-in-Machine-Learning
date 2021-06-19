# Based on PyTorch Tutorial form lecture
# and https://github.com/bentrevett/pytorch-image-classification/blob/master/1_mlp.ipynb

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as data

import torchvision.transforms as transforms
import torchvision.datasets as datasets

from sklearn import metrics
from sklearn import decomposition
from sklearn import manifold
import matplotlib.pyplot as plt
import numpy as np

import copy
import random
import time

# set seeds
SEED = 12
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.backends.cudnn.deterministic = True

class MLP(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()

        self.input_fc = nn.Linear(input_dim, 3000)
        self.output_fc = nn.Linear(3000, output_dim)

    def forward(self, x):

        batch_size = x.shape[0]
        x = x.view(batch_size, -1)
        h = torch.sigmoid(self.input_fc(x))
        output = self.output_fc(h)

        return output, h

def calculate_accuracy(y_pred, y):
    top_pred = y_pred.argmax(1, keepdim = True)
    correct = top_pred.eq(y.view_as(top_pred)).sum()
    acc = correct.float() / y.shape[0]
    return acc

def train(mlp, iterator, optimizer, criterion, device):
    
    epoch_loss = 0
    epoch_acc = 0
    
    mlp.train()
    
    for (x, y) in iterator:
        
        x = x.to(device)
        y = y.to(device)
        
        optimizer.zero_grad()
                
        y_pred, _ = mlp(x)
        loss = criterion(y_pred, y)
        acc = calculate_accuracy(y_pred, y)
        
        loss.backward()
        
        optimizer.step()
        
        epoch_loss += loss.item()
        epoch_acc += acc.item()
        
    return epoch_loss / len(iterator), epoch_acc / len(iterator)

def evaluate(mlp, iterator, criterion, device):
    
    epoch_loss = 0
    epoch_acc = 0
    
    mlp.eval()
    
    with torch.no_grad():
        
        for (x, y) in iterator:

            x = x.to(device)
            y = y.to(device)

            y_pred, _ = mlp(x)
            loss = criterion(y_pred, y)
            acc = calculate_accuracy(y_pred, y)

            epoch_loss += loss.item()
            epoch_acc += acc.item()
        
    return epoch_loss / len(iterator), epoch_acc / len(iterator)

def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs

if __name__ == '__main__':
    # dataset
    transform = transforms.Compose([
                    transforms.Grayscale(),
                    transforms.ToTensor(),
                    transforms.Normalize((0.5), (0.5))
                                ])

    train_dataset = datasets.ImageFolder("data_pgm/faces", transform=transform)

    n = len(train_dataset)
    n_val = int(0.3 * n)
    BATCH_SIZE = 64

    train_dataset = torch.utils.data.Subset(train_dataset, range(n_val, n))
    train_data_loader = data.DataLoader(train_dataset,
                                 shuffle=True,
                                 batch_size=BATCH_SIZE)

    validation_dataset = torch.utils.data.Subset(train_dataset, range(n_val))
    validation_data_loader = data.DataLoader(validation_dataset,
                                 batch_size=BATCH_SIZE)                

    # model                        
    INPUT_DIM = 112 * 92
    OUTPUT_DIM = 40
    mlp = MLP(INPUT_DIM, OUTPUT_DIM)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    mlp = mlp.to(device)

    criterion = nn.CrossEntropyLoss()
    criterion = criterion.to(device)
    optimizer = optim.Adam(mlp.parameters())

    # main loop
    EPOCHS = 10

    best_valid_loss = float('inf')

    for epoch in range(EPOCHS):

        train_loss, train_acc = train(mlp, train_data_loader, optimizer, criterion, device)
        valid_loss, valid_acc = evaluate(mlp, validation_data_loader, criterion, device)

        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            torch.save(mlp.state_dict(), 'atnt-mlp.pt')

        print(f'Epoch: {epoch+1:02}')
        print(f'\tTrain Loss: {train_loss:.3f} | Train Acc: {train_acc*100:.2f}%')
        print(f'\t Val. Loss: {valid_loss:.3f} |  Val. Acc: {valid_acc*100:.2f}%')
