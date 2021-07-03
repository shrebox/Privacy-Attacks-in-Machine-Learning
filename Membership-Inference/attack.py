#!/usr/bin/python

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data.dataset import TensorDataset
from model import TargetNet,ShadowNet,VggModel, AttackMLP
from train import  train_model, train_attack_model, prepare_attack_data
from sklearn.metrics import classification_report
import argparse
import numpy as np
import os
import copy
import random
import matplotlib.pyplot as plt

#set the seed for reproducibility
np.random.seed(1234)

########################
# Model Hyperparameters
########################
#Number of filters for target and shadow models 
target_filters = [128, 512, 512, 512, 512, 512]
shadow_filters = [64, 64, 128, 128]
#New FC layers size for pretrained model
n_fc= [256, 128] 
#For CIFAR-10 and MNIST dataset
num_classes = 10
#No. of training epocs
num_epochs = 30
#how many samples per batch to load
batch_size = 128
#learning rate
learning_rate = 0.001
#Learning rate decay 
lr_decay = 0.96
#Regularizer
reg=1e-4
#percentage of dataset to use for shadow model
shadow_split = 0.6
#Number of validation samples
n_validation = 1000
#Input Channels(RGB)for CIFAR-10
input_dim = 3
#Number of processes
num_workers = 2


################################
#Attack Model Hyperparameters
################################
NUM_EPOCHS = 15
BATCH_SIZE = 50
#Learning rate
LR_ATTACK = 0.0001 
#L2 Regulariser
REG = 1e-6
#weight decay
LR_DECAY = 0.96
#No of hidden units
n_hidden = 128
#Binary Classsifier
out_classes = 2


def get_cmd_arguments():
    parser = argparse.ArgumentParser(prog="MI Attack")
    # group = parser.add_mutually_exclusive_group(required=True)
    parser.add_argument('--dataset', default='CIFAR10', choices=['CIFAR10', 'MNIST'], help='Which dataset to use (CIFAR10 or MNIST)')
    parser.add_argument('--dataPath', default='./data', help='Path to store data')
    parser.add_argument('--modelPath', default='./model', help='Path to save or load model checkpoints')
    parser.add_argument('--trainTargetModel', action='store_true', help='Train a target model, if false then load an already trained model')
    parser.add_argument('--trainShadowModel', action='store_true', help='Train a shadow model, if false then load an already trained model')
    parser.add_argument('--pretrained',action='store_true', help='Use pretrained target model from Pytorch Model Zoo, if false then use the same model without pretrained weights')
    parser.add_argument('--need_augm',action='store_true', help='To use data augmentation on target and shadow training set or not')
    parser.add_argument('--verbose',action='store_true', help='Add Verbosity')
    return parser.parse_args()

#Prepare data loaders for target and shadow models
#Also perform data augementations if required
def get_data_loader(dataset,
                    data_dir,
                    batch,
                    shadow_split=0.5,
                    augm_required=False,
                    pretrained=False,
                    num_workers=1):
    """
     Utility function for loading and returning train and valid
     iterators over the CIFAR-10 and MNIST dataset.
    """ 
    error_msg = "[!] shadow_split should be in the range [0, 1]."
    assert ((shadow_split >= 0) and (shadow_split <= 1)), error_msg
    
    
    if dataset == 'CIFAR10':
        
        #Create train and test transforms
        if pretrained:#for pretrained model
            pretrained_size = 224
            pretrained_means = [0.485, 0.456, 0.406]
            pretrained_stds= [0.229, 0.224, 0.225]

            normalize = transforms.Normalize(mean = pretrained_means, 
                                            std = pretrained_stds)
            test_transforms = transforms.Compose([transforms.Resize(pretrained_size),
                                                transforms.ToTensor(),
                                                normalize])
            if augm_required:
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
        else:#General data transformations for CIFAR10

            normalize = transforms.Normalize(mean=[0.5, 0.5, 0.5], 
                                            std=[0.5, 0.5, 0.5])
            test_transforms = transforms.Compose([transforms.ToTensor(),
                                                normalize])

            if augm_required:
                train_transforms = transforms.Compose([transforms.RandomRotation(5),
                                                    transforms.RandomHorizontalFlip(p=0.5),
                                                    transforms.RandomCrop(32, padding=4),
                                                    transforms.ToTensor(),
                                                    normalize]) 
            else:
                train_transforms = transforms.Compose([transforms.ToTensor(),
                                                normalize])

        #CIFAR10 training dataset
        cifar10_train_data = torchvision.datasets.CIFAR10(root=data_dir,
                                                       train=True,
                                                       transform=train_transforms,
                                                       download=True)  
        #CIFAR10 test dataset
        cifar10_test_data = torchvision.datasets.CIFAR10(root=data_dir, 
                                                        train = False,  
                                                        transform = test_transforms)
                                                        
        #Concatenate original test and train dataset                                             
        shadow_train_dataset = torch.utils.data.ConcatDataset([cifar10_train_data,cifar10_test_data ])
        
        #-------------------------------------------------------------------
        # Prepare Shadow Model Train, Valid, and Test datasets
        #------------------------------------------------------------------
        num_shadow_total = int(len(shadow_train_dataset) * shadow_split)
        mask = list(range(num_shadow_total))
        random.shuffle(mask)
        shadow_dataset = torch.utils.data.Subset(shadow_train_dataset, mask)
        
        num_shadow_train = int(len(shadow_dataset) * 0.5)
        num_shadow_test = num_shadow_total - num_shadow_train
        shadow_train, shadow_test = torch.utils.data.random_split(shadow_dataset, 
                                                               [num_shadow_train, num_shadow_test])
        
        
        #Train/Valid split for target and shadow model
        n_target_samples = len(cifar10_train_data) - n_validation
        target_train, target_val = torch.utils.data.random_split(cifar10_train_data, 
                                                                [n_target_samples, n_validation])
        
        mask_val = list(range(num_shadow_total, num_shadow_total+n_validation))
        shadow_val = torch.utils.data.Subset(shadow_train_dataset, mask_val)
                
        #To ensure Validation dataset uses test transforms
        target_val = copy.deepcopy(target_val)
        shadow_val = copy.deepcopy(shadow_val)
        target_val.dataset.transform = test_transforms
        shadow_val.dataset.transform = test_transforms
    

        #-------------------------------------------------
        # Data loader
        #-------------------------------------------------
        t_train_loader = torch.utils.data.DataLoader(dataset=target_train, 
                                                         batch_size=batch, 
                                                         shuffle=True,
                                                         num_workers=num_workers)

        t_val_loader = torch.utils.data.DataLoader(dataset=target_val,
                                                 batch_size=batch,
                                                 shuffle=False,
                                                 num_workers=num_workers)
                                                 
        t_test_loader = torch.utils.data.DataLoader(dataset=cifar10_test_data,
                                                    batch_size=batch,
                                                    shuffle=False,
                                                    num_workers=num_workers)
        
        s_train_loader = torch.utils.data.DataLoader(dataset=shadow_train,
                                                     batch_size=batch, 
                                                     shuffle=True,
                                                     num_workers=num_workers)
        
        s_val_loader = torch.utils.data.DataLoader(dataset=shadow_val,
                                                 batch_size=batch,
                                                 shuffle=False,
                                                 num_workers=num_workers)
        
        s_test_loader = torch.utils.data.DataLoader(dataset=shadow_test,
                                                    batch_size=batch,
                                                    shuffle=False,
                                                    num_workers=num_workers)
        
        
        print(f'Number of target train samples: {len(t_train_loader.dataset)}')
        print(f'Number of shadow train samples: {len(s_train_loader.dataset)}')
        print(f'Number of target valid samples: {len(t_val_loader.dataset)}')
        print(f'Number of shadow valid samples: {len(s_val_loader.dataset)}')
        print(f'Number of target test samples: {len(t_test_loader.dataset)}')
        print(f'Number of shadow test samples: {len(s_test_loader.dataset)}')

        return t_train_loader, t_val_loader, t_test_loader, s_train_loader, s_val_loader, s_test_loader


def attack_inference(model,
                    test_X,
                    test_Y,
                    device):
    
    print('----Attack Model Testing----')

    targetnames= ['Non-Member', 'Member']
    pred_y = []
    true_y = []
    
    #Tuple of tensors
    X = torch.cat(test_X)
    Y = torch.cat(test_Y)
    

    #Create Inference dataset
    inferdataset = TensorDataset(X,Y) 

    dataloader = torch.utils.data.DataLoader(dataset=inferdataset,
                                            batch_size=50,
                                            shuffle=False,
                                            num_workers=num_workers)

    #Evaluation of Attack Model
    model.eval() 
    with torch.no_grad():
        correct = 0
        total = 0
        for i, (inputs, labels) in enumerate(dataloader):
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            outputs = model(inputs)
            
            #Predictions for accuracy calculations
            _, predictions = torch.max(outputs.data, 1)
            total+=labels.size(0)
            correct+=(predictions == labels).sum().item()
            
            # print('True Labels for Batch [{}] are : {}'.format(i,labels))
            # print('Predictions for Batch [{}] are : {}'.format(i,predictions))
            
            true_y.append(labels.cpu())
            pred_y.append(predictions.cpu())
        
    attack_acc = correct / total
    print('---Attack Test Accuracy is---- : {:.2f}%'.format(100*attack_acc))
    
    true_y =  torch.cat(true_y).numpy()
    pred_y = torch.cat(pred_y).numpy()
    
    print('---More Detailed Results----')
    print(classification_report(true_y,pred_y, target_names=targetnames))


def create_attack(args):
 
    dataset = args.dataset
    need_augm = args.need_augm
    pretrained = args.pretrained
    verbose = args.verbose

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

    #Creating data loaders
    t_train_loader, t_val_loader, t_test_loader,\
    s_train_loader, s_val_loader, s_test_loader = get_data_loader(dataset, 
                                                                datasetDir,
                                                                batch_size,
                                                                shadow_split,
                                                                need_augm,
                                                                pretrained,
                                                                num_workers)
                        
    
    if (args.trainTargetModel):
        if (pretrained):
            target_model = VggModel(num_classes,n_fc,args.pretrained).to(device)
        else:             
            target_model = TargetNet(input_dim,target_filters,num_classes).to(device) 
        
        # Print the model we just instantiated
        if args.verbose:
            print('----Target Model Architecure----')
            print(target_model)
            print('----Model Learnable Params----')
            for name,param in target_model.named_parameters():
                 if param.requires_grad == True:
                    print("\t",name)
        

        # Loss and optimizer for Tager Model
        loss = nn.CrossEntropyLoss()
        
        optimizer = torch.optim.Adam(target_model.parameters(), lr=learning_rate, weight_decay=reg)
        lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer,gamma=lr_decay)

        
        targetX, targetY = train_model(target_model,
                                    t_train_loader,
                                    t_val_loader,
                                    t_test_loader,
                                    loss,
                                    optimizer,
                                    lr_scheduler,
                                    device,
                                    modelDir,
                                    verbose,
                                    num_epochs,
                                    is_target=True)

    if (args.trainShadowModel):
        shadow_model = ShadowNet(input_dim,shadow_filters,num_classes).to(device)

        # Print the model we just instantiated
        if args.verbose:
            print('----Shadow Model Architecure---')
            print(shadow_model)
            print('---Model Learnable Params----')
            for name,param in shadow_model.named_parameters():
                 if param.requires_grad == True:
                    print("\t",name)
        
        # Loss and optimizer
        shadow_loss = nn.CrossEntropyLoss()
        shadow_optimizer = torch.optim.Adam(shadow_model.parameters(), lr=learning_rate, weight_decay=reg)
        shadow_lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(shadow_optimizer,gamma=lr_decay)

        shadowX, shadowY = train_model(shadow_model,
                                    s_train_loader,
                                    s_val_loader,
                                    s_test_loader,
                                    shadow_loss,
                                    shadow_optimizer,
                                    shadow_lr_scheduler,
                                    device,
                                    modelDir,
                                    verbose,
                                    num_epochs,
                                    is_target=False)


    if not(args.trainShadowModel):
        shadow_file = os.path.join(modelDir,'best_shadow_model.ckpt')
        print('Use Shadow model at the path  ====> [{}] '.format(modelDir))
        #Instantiate Shadow Model Class
        shadow_model = ShadowNet(input_dim,shadow_filters,num_classes).to(device)
        #Load the saved model
        shadow_model.load_state_dict(torch.load(shadow_file))
        #Prepare dataset for training attack model
        print('----Preparing Attack training data---')
        trainX, trainY = prepare_attack_data(shadow_model,s_train_loader,device)
        testX, testY = prepare_attack_data(shadow_model,s_test_loader,device,test_dataset=True)
        shadowX = trainX + testX
        shadowY = trainY + testY
    
    if not(args.trainTargetModel):
        target_file = os.path.join(modelDir,'best_target_model.ckpt')
        print('Use Target model at the path for Attack Inference ====> [{}] '.format(modelDir))
        #Instantiate Target Model Class
        target_model = TargetNet(input_dim,target_filters,num_classes).to(device)
        target_model.load_state_dict(torch.load(target_file))
        print('---Peparing Attack inference data---')
        t_trainX, t_trainY = prepare_attack_data(target_model,t_train_loader,device)
        t_testX, t_testY = prepare_attack_data(target_model,t_test_loader,device,test_dataset=True)
        targetX = t_trainX + t_testX
        targetY = t_trainY + t_testY

    ###################################
    # Attack Model Training
    ##################################
    #The input dimension to MLP attack model
    input_size = shadowX[0].size(1)
    print('Input Feature dim for Attack Model : [{}]'.format(input_size))
    
    attack_model = AttackMLP(input_size,n_hidden,out_classes).to(device)
    # Loss and optimizer
    attack_loss = nn.CrossEntropyLoss()
    attack_optimizer = torch.optim.Adam(shadow_model.parameters(), lr=LR_ATTACK, weight_decay=REG)
    attack_lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(attack_optimizer,gamma=LR_DECAY)

    
    #Feature vector and labels for training Attack model
    attackdataset = (shadowX, shadowY)
    
    attack_valacc = train_attack_model(attack_model, attackdataset, attack_loss,
                       attack_optimizer, attack_lr_scheduler, device, modelDir,
                        NUM_EPOCHS, BATCH_SIZE, verbose)
   
  
    print('Validation Accuracy for the Best Attack Model is: {:.2f} %'.format(100* attack_valacc))
   
    #Load the trained attack model
    attack_path = os.path.join(modelDir,'best_attack_model.ckpt')
    attack_model.load_state_dict(torch.load(attack_path))
    
    #Inference on trained attack model
    attack_inference(attack_model, targetX, targetY, device)

if __name__ == '__main__':
    #get command line arguments from the user
    args = get_cmd_arguments()
    print(args)
    #Generate Membership inference attack1
    create_attack(args)