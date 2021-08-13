import pandas as pd
import glob
import torch
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from torchvision import transforms
import numpy as np
import torch.nn as nn
import torch.optim as optim
import copy
import PIL

from train import train, test
from models import TargetModel, AttackModel
from datasets import UTKFace, AttackData

#################
# Split Dataset #
#################

SEED = 42
PATH = 'Attribute-Inference/UTKFace/'
TEST_SPLIT = 0.2
ATTACK_SPLIT = 0.5 # 50% of training data

samples=pd.DataFrame(columns=['filename','gender','race'])
samples['filename'] = pd.Series([file for file in glob.glob(PATH + '*') 
                                     if len(file) == len(PATH) + len('61_1_0_20170104183517277.jpg.chip.jpg')])
samples['gender'] = samples['filename'].apply(lambda x: (x.split("_")[1]))
samples['race'] = samples['filename'].apply(lambda x: (x.split("_")[2]))

np.random.seed(SEED)

dataset_size = len(samples)
indices = list(range(dataset_size))
split = int(np.floor(TEST_SPLIT * dataset_size))
np.random.shuffle(indices)
train_indices, test_indices = indices[split:], indices[:split]

attack_indices = copy.deepcopy(train_indices)
attack_split = int(np.floor(ATTACK_SPLIT * len(train_indices)))
np.random.shuffle(attack_indices)
attack_indices = attack_indices[attack_split:]

train_samples = samples.iloc[train_indices]
test_samples = samples.iloc[test_indices]
attack_samples = samples.iloc[attack_indices]


################
# Target Model #
################

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

LEARNING_RATE = 0.001
BATCH_SIZE = 128
EPOCHS = 30

transform = transforms.Compose([
        transforms.Resize([50, 50]),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

taget_model = TargetModel().to(DEVICE)
target_criterion = nn.CrossEntropyLoss()
target_optimizer = torch.optim.Adam(taget_model.parameters(), lr=LEARNING_RATE)

target_train_loader = torch.utils.data.DataLoader(UTKFace(train_samples, 'gender', transform), 
                                                batch_size=BATCH_SIZE)

target_test_loader = torch.utils.data.DataLoader(UTKFace(test_samples, 'gender', transform), 
                                                batch_size=BATCH_SIZE)

################
# Attack Model #
################

LEARNING_RATE = 0.001
BATCH_SIZE = 128

attack_model = AttackModel().to(DEVICE)
attack_criterion = nn.CrossEntropyLoss()
attack_optimizer = torch.optim.Adam(attack_model.parameters(), lr=LEARNING_RATE)

attack_train_loader = torch.utils.data.DataLoader(AttackData(attack_samples, taget_model, transform), 
                                                batch_size=BATCH_SIZE)
attack_test_loader = torch.utils.data.DataLoader(AttackData(test_samples, taget_model, transform), 
                                            batch_size=BATCH_SIZE)


##################
# Perform Attack #
##################



def perform_attack(train: bool, epochs):

    target_model_path = 'Attribute-Inference/models/target_model_' + str(EPOCHS) + '.pth'
    attack_model_path = './models/attack_model_' + str(epochs) + '.pth' 

    if train:
        print('Training Target Model:')
        train.train(EPOCHS, target_train_loader, target_optimizer, target_criterion, attack_model, attack_model_path)
    
    print('Loading Target Model:')
    target_model = TargetModel()
    target_model.load_state_dict(torch.load(target_model_path))

    print('Testing Target Model:')
    target_test_loader = torch.utils.data.DataLoader(UTKFace(test_samples, 'gender', transform), 
                                            batch_size=BATCH_SIZE)
    test(target_test_loader, target_model, False)


    if train:
        print('Training Attack Model:')
        train(epochs, attack_train_loader, attack_optimizer, attack_criterion, attack_model, attack_model_path)

    print('Loading Attack Model:')
    attack_model = AttackModel()
    net.load_state_dict(torch.load(attack_model_path))

    print('Testing Attack Model')
    train.test(attack_test_loader, attack_model, True)

