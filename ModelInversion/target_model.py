# Based on PyTorch Tutorial form lecture 
# and https://github.com/bentrevett/pytorch-image-classification/blob/master/1_mlp.ipynb

import torch.optim as optim
import torch.nn.functional as F
import torch.nn as nn
import torch
import torchvision
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader

import time

transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
     ])


atnt_faces = datasets.ImageFolder("data_pgm/faces", transform=transform)
n = len(atnt_faces)
n_val = int(0.3 * n)
valset = torch.utils.data.Subset(atnt_faces, range(n_val))
trainset = torch.utils.data.Subset(atnt_faces, range(n_val, n))

batch_size = 32  # 16 32 64 is ok, just depend your computer
trainloader = DataLoader(trainset, batch_size=batch_size,
                         shuffle=True, num_workers=2)
valloader = DataLoader(valset, batch_size=batch_size,
                       shuffle=False, num_workers=2)


class MLP(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()

        self.input_fc = nn.Linear(input_dim, 3000)
        self.output_fc = nn.Linear(3000, output_dim)

    def forward(self, x):

        batch_size = x.shape[0]
        x = x.view(batch_size, -1)
        h = F.sigmoid(self.input_fc(x))
        output = self.output_fc(h)

        return output, h


INPUT_DIM = 3 * 112 * 92
OUTPUT_DIM = 40
model = MLP(INPUT_DIM, OUTPUT_DIM)


criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters())

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using {} device".format(device))

model = model.to(device)
criterion = criterion.to(device)


def calculate_accuracy(y_pred, y):
    top_pred = y_pred.argmax(1, keepdim=True)
    correct = top_pred.eq(y.view_as(top_pred)).sum()
    acc = correct.float() / y.shape[0]
    return acc


def train(model, iterator, optimizer, criterion, device):

    epoch_loss = 0
    epoch_acc = 0

    model.train()

    for (x, y) in iterator:

        x = x.to(device)
        y = y.to(device)

        optimizer.zero_grad()

        y_pred, _ = model(x)

        loss = criterion(y_pred, y)

        acc = calculate_accuracy(y_pred, y)

        loss.backward()

        optimizer.step()

        epoch_loss += loss.item()
        epoch_acc += acc.item()

    return epoch_loss / len(iterator), epoch_acc / len(iterator)


def evaluate(model, iterator, criterion, device):

    epoch_loss = 0
    epoch_acc = 0

    model.eval()

    with torch.no_grad():

        for (x, y) in iterator:

            x = x.to(device)
            y = y.to(device)

            y_pred, _ = model(x)

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
    EPOCHS = 10

    best_valid_loss = float('inf')

    for epoch in range(EPOCHS):

        start_time = time.monotonic()

        train_loss, train_acc = train(
            model, trainloader, optimizer, criterion, device)
        valid_loss, valid_acc = evaluate(model, valloader, criterion, device)

        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            torch.save(model.state_dict(), 'atnt_mlp.pt')

        end_time = time.monotonic()

        epoch_mins, epoch_secs = epoch_time(start_time, end_time)

        print(f'Epoch: {epoch+1:02} | Epoch Time: {epoch_mins}m {epoch_secs}s')
        print(
            f'\tTrain Loss: {train_loss:.3f} | Train Acc: {train_acc*100:.2f}%')
        print(
            f'\t Val. Loss: {valid_loss:.3f} |  Val. Acc: {valid_acc*100:.2f}%')

    model.load_state_dict(torch.load('atnt_mlp.pt'))
    test_loss, test_acc = evaluate(model, valloader, criterion, device)

    print(f'Test Loss: {test_loss:.3f} | Test Acc: {test_acc*100:.2f}%')
