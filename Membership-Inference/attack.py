#!/usr/bin/python
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data.sampler import SubsetRandomSampler
from model import ConvNet,VggModel,AttackMLP
from train import  train_model, train_attack_model
import argparse
import numpy as np
import os
import copy

random_seed = 1234

########################
# Model Hyperparameters
########################
#ConvNet Model Hidden Layers
# hidden_layers = [128, 512, 512, 512, 512, 512] 
hidden_layers = [64, 128, 128, 128]
#FC layers for pretrained model
layer_config= [512, 256] 
#For CIFAR-10 and MNIST dataset
num_classes = 10
#No. of training epocs
num_epochs = 30
#how many samples per batch to load
batch_size = 50
#learning rate
learning_rate = 1e-2
#Learning rate decay 
lr_decay = 0.96
#Regularizer
reg=1e-2
#percentage of training data to use for target model
target_shadow_split = 0.6
#percentage of training set to use as validation
valid_ratio = 0.95
#Input Channels(RGB)for CIFAR-10
input_dim = 3
#For MLP attack mode
input_size = 3 

################################
#Attack Model Hyperparameters
################################
NUM_EPOCHS = 50
BATCH_SIZE = 10
#Learning rate
LR_ATTACK = 1e-2 
#L2 Regulariser(Weight decay)
REG = 1e-6 
LR_DECAY = 0.96
#No of hidden units
n_hidden = 64


def get_cmd_arguments():
    parser = argparse.ArgumentParser(prog="MI Attack")
    # group = parser.add_mutually_exclusive_group(required=True)
    parser.add_argument('--dataset', default='CIFAR10', choices=['CIFAR10', 'MNIST'], help='Which dataset to use (CIFAR10 or MNIST)')
    parser.add_argument('--dataPath', default='./data', help='Path to store data')
    parser.add_argument('--modelPath', default='./model', help='Path to save or load model checkpoints')
    parser.add_argument('--trainTargetModel', action='store_true', help='Train a target model, if false then load an already trained model')
    parser.add_argument('--trainShadowModel', action='store_true', help='Train a shadow model, if false then load an already trained model')
    parser.add_argument('--pretrained',action='store_true', help='Use pretrained target model from Pytorch Model Zoo, if false then use the same model without pretrained weights')
    parser.add_argument('--use_same_arch',action='store_true', help='If True, we use ConvNet model for both Target and Shadow Model training')
    parser.add_argument('--need_augm',action='store_true', help='To use data augmentation on target and shadow training set or not')
    parser.add_argument('--use_same_hyperparam',action='store_false', help='To use same hyperparameter values for training Target and Shadow Model')
    parser.add_argument('--verbose',action='store_true', help='Add Verbosity')
    return parser.parse_args()

def get_data_loader(dataset,
                            data_dir,
                            batch,
                            seed,
                            need_augm,
                            pretrained=False,
                            train_split=0.5,
                            shuffle=True,
                            num_workers=2,
                            pin_memory=False):
    """
     Utility function for loading and returning train and valid
     multi-process iterators over the CIFAR-10 dataset.
     If using CUDA, num_workers should be set to 1 and pin_memory to True.
    """ 
    error_msg = "[!] train_split should be in the range [0, 1]."
    assert ((train_split >= 0) and (train_split <= 1)), error_msg

    if dataset == 'CIFAR10':
 
        if pretrained:
            pretrained_size = 224
            pretrained_means = [0.485, 0.456, 0.406]
            pretrained_stds= [0.229, 0.224, 0.225]

            normalize = transforms.Normalize(mean = pretrained_means, 
                                            std = pretrained_stds)
            test_transforms = transforms.Compose([transforms.Resize(pretrained_size),
                                                transforms.ToTensor(),
                                                normalize])
            if need_augm:
                train_transforms = transforms.Compose([transforms.Resize(pretrained_size),
                                                    transforms.RandomRotation(5),
                                                    transforms.RandomHorizontalFlip(p=0.5),
                                                    transforms.RandomCrop(pretrained_size, padding = 10),
                                                    transforms.ToTensor(),
                                                    normalize])
            else:
                train_transforms = transforms.Compose([transforms.Resize(pretrained_size),
                                                     transforms.ToTensor(),
                                                     normalize])
        else:#Not pretrained

            normalize = transforms.Normalize(mean=[0.5, 0.5, 0.5], 
                                            std=[0.5, 0.5, 0.5])
            test_transforms = transforms.Compose([transforms.ToTensor(),
                                                normalize])

            if need_augm:
                train_transforms = transforms.Compose([transforms.RandomRotation(5),
                                                    transforms.RandomHorizontalFlip(p=0.5),
                                                    transforms.RandomAffine(degrees =  0, translate = (0.125, 0.125)),
                                                    transforms.ToTensor(),
                                                    normalize]) 
            else:
                train_transforms = transforms.Compose([transforms.ToTensor(),
                                                normalize])

        #load the train and test dataset
        cifar10_train_data = torchvision.datasets.CIFAR10(root=data_dir,
                                                       train=True,
                                                       transform=train_transforms,
                                                       download=True)  
        #
        cifar10_test_data = torchvision.datasets.CIFAR10(root=data_dir, 
                                                        train = False,  
                                                        transform = test_transforms)
        
        #-------------------------------------------------------------------
        # Prepare the Target and Shadow Model training and Validation Splits 
        #------------------------------------------------------------------
        n_train_examples = int(len(cifar10_train_data) * valid_ratio)
        n_valid_examples = len(cifar10_train_data) - n_train_examples
        train_data, valid_data = torch.utils.data.random_split(cifar10_train_data, 
                                                               [n_train_examples, n_valid_examples])
        
        #Training plit between Target and shadow model
        train_samples = len(train_data)
        indices = list(range(train_samples))
        split = int(np.floor(train_split * train_samples))

        if shuffle:
            np.random.seed(seed)
            np.random.shuffle(indices)

        target_train_idx, shadow_train_idx = indices[:split], indices[split:]
        train_sampler = SubsetRandomSampler(target_train_idx)
        shadow_sampler = SubsetRandomSampler(shadow_train_idx)
        
        #To ensure Validqtion dataset uses test transforms
        valid_data = copy.deepcopy(valid_data)
        valid_data.dataset.transform = test_transforms

        print(f'Total training samples: {len(train_data)}')
        print(f'Total validation samples: {len(valid_data)}')
        print(f'Total testing samples: {len(cifar10_test_data)}')
        print(f'Number of target samples: {len(target_train_idx)}')
        print(f'Number of shadow samples: {len(shadow_train_idx)}')

        #-------------------------------------------------
        # Data loader
        #-------------------------------------------------
        target_train_loader = torch.utils.data.DataLoader(dataset=train_data, 
                                                         batch_size=batch, 
                                                         sampler=train_sampler,
                                                         num_workers=num_workers, 
                                                         pin_memory=pin_memory)

        
        shadow_train_loader = torch.utils.data.DataLoader(dataset=train_data,
                                                          batch_size=batch, 
                                                          sampler=shadow_sampler,
                                                          num_workers=num_workers, 
                                                          pin_memory=pin_memory)
        
        
        val_loader = torch.utils.data.DataLoader(dataset=valid_data,
                                                 batch_size=batch,
                                                 shuffle=False)
        

        test_loader = torch.utils.data.DataLoader(dataset=cifar10_test_data,
                                                 batch_size=batch,
                                                 shuffle=False)


        return target_train_loader, shadow_train_loader, val_loader, test_loader

def create_attack(args):
 
    dataset = args.dataset
    need_augmentation = args.need_augm
    datasetDir = os.path.join(args.dataPath,dataset)
    modelDir = os.path.join(args.modelPath, dataset)  
    
    #Create dataset and model directories
    if not os.path.exists(datasetDir):
        try:
            os.makedirs(datasetDir)
        except OSError:
            pass
    
    if not os.path.exists(modelDir):
        try:
            os.makedirs(modelDir)
        except OSError:
            pass 

    # setting device on GPU if available, else CPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Using device:', device)

    #Setting
    if device.type == 'cuda':
       target_loader, shadow_loader, val_loader, test_loader = get_data_loader(dataset, 
                                                                datasetDir,
                                                                batch_size,
                                                                random_seed,
                                                                need_augmentation,
                                                                args.pretrained,
                                                                target_shadow_split,
                                                                num_workers=1,
                                                                pin_memory=True)
                        
    else:#device is CPU
       target_loader,shadow_loader, val_loader, test_loader =  get_data_loader(dataset, 
                                                                datasetDir,
                                                                batch_size,
                                                                need_augmentation,
                                                                random_seed,
                                                                target_shadow_split)
    

    if (args.trainTargetModel):
        if (args.use_same_arch):
            target_model = ConvNet(input_dim,hidden_layers,num_classes).to(device) 
        else:             
            target_model = VggModel(num_classes,layer_config,args.pretrained).to(device)
     
        # Print the model we just instantiated
        print(target_model)

        # Loss and optimizer for Tager Model
        loss = nn.CrossEntropyLoss()
            
        params_to_update = target_model.parameters()
        for name,param in target_model.named_parameters():
            if param.requires_grad == True:
                print("\t",name)
        
        optimizer = torch.optim.Adam(params_to_update, lr=learning_rate, weight_decay=reg)
        lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer,gamma=lr_decay)

        targetX, targetY = train_model(target_model,
                                    target_loader,
                                    val_loader,
                                    test_loader,
                                    loss,
                                    optimizer,
                                    lr_scheduler,
                                    device,
                                    modelDir,
                                    args.verbose,
                                    num_epochs,
                                    is_target=True)
        
    if (args.trainShadowModel):
        shadow_model = ConvNet(input_dim,hidden_layers,num_classes).to(device)

        # Print the model we just instantiated
        print(shadow_model)
        
        # Loss and optimizer
        loss = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(shadow_model.parameters(), lr=learning_rate, weight_decay=reg)
        lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer,gamma=lr_decay)

        shadowX, shadowY= train_model(shadow_model,
                                    shadow_loader,
                                    val_loader,
                                    test_loader,
                                    loss,
                                    optimizer,
                                    lr_scheduler,
                                    device,
                                    modelDir,
                                    args.verbose,
                                    num_epochs,
                                    is_target=False)
    
    ################################
    # Train Attack model
    ################################
    attack_model = AttackMLP(input_size, hidden_size=64)
    print(attack_model)
    
    #In ML-Leaks paper, Softmax was used at the output layer
    #CE loss function mimics the same function, so therefore there is no softmax 
    #activation defined at the output layer in AttackMLP model.
    loss = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(shadow_model.parameters(), lr=LR_ATTACK, weight_decay=REG)
    lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer,gamma=LR_DECAY)


    attack_acc, attack_loss = train_attack_model(shadowX,
                                                shadowY,
                                                attack_model,
                                                loss,
                                                optimizer,
                                                lr_scheduler,
                                                device,
                                                NUM_EPOCHS,
                                                BATCH_SIZE,
                                                n_hidden)

                                    

if __name__ == '__main__':
    #get command line arguments from the user
    args = get_cmd_arguments()
    print(args)
    #Generate Membership inference attack1
    create_attack(args)