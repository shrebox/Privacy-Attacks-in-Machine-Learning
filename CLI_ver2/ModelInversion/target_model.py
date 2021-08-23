# Based on PyTorch Tutorial form lecture
# and https://github.com/bentrevett/pytorch-image-classification/blob/master/1_mlp.ipynb

import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data

import torchvision.transforms as transforms
import torchvision.datasets as datasets
import os

# set seeds
SEED = 12
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
#torch.backends.cudnn.deterministic = True


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


def train_target(dataFolder,modelFolder):
    # transfrom, wee need grayscale to convert the images to 1 channel
    transform = transforms.Compose([
                    transforms.Grayscale(),
                    transforms.ToTensor(),
                    transforms.Normalize((0.5), (0.5))
                                ])

    data_path = os.path.join(dataFolder, 'data_pgm')
    # load dataset
    atnt_faces = datasets.ImageFolder(data_path, transform=transform)

    # split dataset: 3 images of every class as validation set
    i = [i for i in range(len(atnt_faces)) if i % 10 > 3]
    i_val = [i for i in range(len(atnt_faces)) if i % 10 <= 3]

    # load data
    BATCH_SIZE = 64
    train_dataset = torch.utils.data.Subset(atnt_faces, i)
    train_data_loader = data.DataLoader(train_dataset,
                                 shuffle=True,
                                 batch_size=BATCH_SIZE)

    validation_dataset = torch.utils.data.Subset(atnt_faces, i_val)
    validation_data_loader = data.DataLoader(validation_dataset,
                                 batch_size=BATCH_SIZE)   

    # define dimensions                        
    INPUT_DIM = 112 * 92
    OUTPUT_DIM = 40

    # create model
    mlp = MLP(INPUT_DIM, OUTPUT_DIM)

    # set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    mlp = mlp.to(device)

    # set criterion and optimizer
    criterion = nn.CrossEntropyLoss()
    #criterion = criterion.to(device)
    optimizer = optim.Adam(mlp.parameters())

    # main loop
    EPOCHS = 30
    best_valid_loss = float('inf')

    model_path = os.path.join(modelFolder, 'atnt-mlp-model.pt')
    
    print('---Target Model Training Started---')
    for epoch in range(EPOCHS):

        train_loss, train_acc = train(mlp, train_data_loader, optimizer, criterion, device)
        valid_loss, valid_acc = evaluate(mlp, validation_data_loader, criterion, device)

        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            torch.save(mlp, model_path)

        print(f'Epoch: {epoch+1:02}')
        print(f'\tTrain Loss: {train_loss:.3f} | Train Acc: {train_acc*100:.2f}%')
        print(f'\t Val. Loss: {valid_loss:.3f} |  Val. Acc: {valid_acc*100:.2f}%')
        
    print('---Target Model Training Done---')


if __name__ == '__main__':
    train_target()
    