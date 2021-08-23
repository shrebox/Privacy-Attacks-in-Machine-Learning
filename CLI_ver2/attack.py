#!/usr/bin/python

from scipy.sparse import data
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data.dataset import TensorDataset
from torch.utils.data.sampler import SubsetRandomSampler
import model
from model import init_params as w_init
from train import  prepare_attack_data, train_model, train_attack_model, get_feature_representation
from datasets import UTKFace
from sklearn.metrics import classification_report
import argparse
import numpy as np
import os
import copy
import random
import matplotlib.pyplot as plt
import pandas as pd
import sys

sys.path.insert(1, 'ModelInversion')
import ModelInversion.model_inversion as mn
import ModelInversion.target_model as target


#set the seed for reproducibility
np.random.seed(1234)
#Flag to enable early stopping
need_earlystop = False

####################################################
#Hyperparameters for Memembership Inference
#-------------------------------------------------
#Target(Shadow) Model Hyperparameters
####################################################
#Number of filters for target and shadow models 
target_filters = [128, 256, 256]
shadow_filters = [64, 128, 128]
#New FC layers size for pretrained model
n_fc= [256, 128] 
#For CIFAR-10 and MNIST dataset
num_classes = 10
#No. of training epocs
num_epochs = 50
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
#Number of processes
num_workers = 2
#Hidden units for MNIST model
n_hidden_mnist = 32

################################
#Attack Model Hyperparameters
################################
NUM_EPOCHS = 50
BATCH_SIZE = 10
#Learning rate
LR_ATTACK = 0.001 
#L2 Regulariser
REG = 1e-7
#weight decay
LR_DECAY = 0.96
#No of hidden units
n_hidden = 128
#Binary Classsifier
out_classes = 2

#####################################################
# Attribute Inference: Target Model Hyperparameters
#####################################################
ATTR_TEST_SPLIT = 0.1 #Though the paper use 20% as test split but with less training samples, reduced the test split
ATTR_ATTACK_SPLIT = 0.5 # 50% of training data
ATTR_BATCH_SIZE = 128
ATTR_LEARNING_RATE = 0.001
ATTR_L2_RATIO = 1e-4
ATTR_LR_DECAY = 0.96
FC_DIM = 128
ATTR_NUM_EPOCHS = 50
ATTACK_L2_RATIO = 1e-7
ATTACK_BATCH = 10


def get_cmd_arguments():
    parser = argparse.ArgumentParser(description ="Privacy attacks against Machine Learning Models")
    parser.add_argument('--attack', default='MemInv', type=str, choices=['MemInf', 'ModelInv', 'AttrInf'], required=True, help='Attack Type')
    parser.add_argument('--dataset', default='CIFAR10', type=str, choices=['CIFAR10', 'MNIST','UTKFace','ATTFace'], required=True, help='Which dataset to use for attack')
    parser.add_argument('--dataPath', default='./data', type=str, help='Path to store or load data')
    parser.add_argument('--modelPath', default='./model', type=str, help='Path for model checkpoints')
    parser.add_argument('--inference', action='store_true', help='For testing attack performance')
    parser.add_argument('--modelCkpt', default='', type=str, help='Trained Target model checkpoint file for test attack')
    parser.add_argument('--label_idx', default=-1, type=int, help='Label/Tag to be reconstructed in Model Inversion attack')
    parser.add_argument('--num_iter', default=1, type=int, help='Number of Iterations for Model Inversion attack')
    parser.add_argument('--need_augm',action='store_true', help='To use data augmentation on target and shadow training set or not')
    parser.add_argument('--need_topk',action='store_true', help='Flag to enable using Top 3 posteriors for attack data')
    parser.add_argument('--param_init', action='store_true', help='Flag to enable custom model params initialization')
    parser.add_argument('--verbose',action='store_true', help='Add Verbosity')

    args = parser.parse_args()

    if args.inference:
        if args.modelCkpt == None:
            parser.error('--modelCkpt parameter cannot be empty string')
        elif args.attack == 'ModelInv' and args.label_idx != -1:
                if args.label_idx < 1 or args.label_idx > 40:
                     parser.error('Label Index should either be -1 or between 1 and 40(both inclusive')
    
    if args.attack == 'MemInf':
        if (args.dataset == 'UTKFace' or args.dataset == 'ATTFace'):
            parser.error('Wrong dataset for Membership Inference Attack')
    elif args.attack == 'AttrInf':
        if (args.dataset not in ['UTKFace']):
            parser.error('Wrong dataset for Attribute Inference Attack')
    elif (args.dataset not in ['ATTFace']):
        parser.error('Wrong dataset for Model Inversion Attack')
    
    if args.label_idx != -1:
        if args.label_idx < 1 or args.label_idx > 40:
            parser.error('Label Index should either be -1 or between 1 and 40(both inclusive')
      

    return args

#####################################################################
# Data Transformations for different datasets used in the attack
####################################################################
def get_data_transforms(dataset, augm=False):

    if dataset == 'CIFAR10':
        normalize = transforms.Normalize(mean=[0.5, 0.5, 0.5], 
                                            std=[0.5, 0.5, 0.5])
        test_transforms = transforms.Compose([transforms.ToTensor(),
                                            normalize])

        if augm:
            train_transforms = transforms.Compose([transforms.RandomRotation(5),
                                                transforms.RandomHorizontalFlip(p=0.5),
                                                transforms.RandomCrop(32, padding=4),
                                                transforms.ToTensor(),
                                                normalize]) 
        else:
            train_transforms = transforms.Compose([transforms.ToTensor(),
                                                normalize])

    elif dataset == 'MNIST':
        #The values 0.1307 and 0.3081 used for the Normalize() transformation below are the global mean and standard deviation 
        #of the MNIST dataset
        test_transforms = torchvision.transforms.Compose([torchvision.transforms.ToTensor(),
                                                        torchvision.transforms.Normalize((0.1307,), (0.3081,))])
        if augm:
            train_transforms = torchvision.transforms.Compose([transforms.RandomRotation(5),
                                                    transforms.RandomHorizontalFlip(p=0.5),
                                                    torchvision.transforms.ToTensor(),
                                                    torchvision.transforms.Normalize((0.1307,), (0.3081,))])
        else:
      
            train_transforms = torchvision.transforms.Compose([torchvision.transforms.ToTensor(),
                                                    torchvision.transforms.Normalize((0.1307,), (0.3081,))])
    elif dataset == 'UTKFace':
        #Mean and Std values for UTKFace dataset
        normalize = transforms.Normalize(mean=[152.13768243, 116.5061518,   99.7395918], 
                                            std=[65.71289385, 58.56545956, 57.4306078])
        
        test_transforms = transforms.Compose([transforms.ToTensor(),
                                            normalize])

        train_transforms = transforms.Compose([transforms.ToTensor(),
                                            normalize])
        
    return train_transforms, test_transforms

################################################################################
# Dataset split beween Train and Shadow model For Membership Inference Ata=tack
################################################################################
def split_dataset(train_dataset):
    
    #For simplicity we are only using orignal training set and splitting into 4 equal parts
    #and assign it to Target train/test and Shadow train/test.
    total_size = len(train_dataset)
    split1 = total_size // 4
    split2 = split1*2
    split3 = split1*3
    
    indices = list(range(total_size))
    
    np.random.shuffle(indices)
    
    #Shadow model train and test set
    s_train_idx = indices[:split1]
    s_test_idx = indices[split1:split2]

    #Target model train and test set
    t_train_idx = indices[split2:split3]
    t_test_idx = indices[split3:]
    
    return s_train_idx, s_test_idx,t_train_idx,t_test_idx
    

#--------------------------------------------------------------------------------------------
# Prepare dataloaders for Shadow and Target models for Membership Inference Attack
# Data Strategy:
# - Split the entire training dataset into 4 disjoint parts(T_tain, T_test, S_train, S_test)
#  Target -  Train on T_train and T_test
#  Shadow -  Train on S_train and S_test
#  Attack - Use T_train and T_test for evaluation
#           Use S_train and S_test for training
#----------------------------------------------------------------------------------------------
def get_data_loader(dataset,
                    data_dir,
                    batch,
                    shadow_split=0.5,
                    augm_required=False,
                    test_flag =False,
                    num_workers=1):
    """
     Utility function for loading and returning train and valid
     iterators over the CIFAR-10 and MNIST dataset.
    """ 
    error_msg = "[!] shadow_split should be in the range [0, 1]."
    assert ((shadow_split >= 0) and (shadow_split <= 1)), error_msg
    
    
    train_transforms, test_transforms = get_data_transforms(dataset,augm_required)
        
    #Download test and train dataset
    if dataset == 'CIFAR10':
        #CIFAR10 training set
        train_set = torchvision.datasets.CIFAR10(root=data_dir,
                                                    train=True,
                                                    transform=train_transforms,
                                                    download=True)  
        #CIFAR10 test set
        test_set = torchvision.datasets.CIFAR10(root=data_dir, 
                                                train = False,  
                                                transform = test_transforms)
        
        s_train_idx, s_out_idx, t_train_idx, t_out_idx = split_dataset(train_set)    
    else:
        #MNIST train set
        train_set = torchvision.datasets.MNIST(root=data_dir,
                                        train=True,
                                        transform=train_transforms,
                                        download=True)
        #MNIST test set
        test_set = torchvision.datasets.MNIST(root=data_dir, 
                                        train = False,  
                                        transform = test_transforms)
        
        s_train_idx, s_out_idx, t_train_idx, t_out_idx = split_dataset(train_set)
   
    
    # Data samplers
    s_train_sampler = SubsetRandomSampler(s_train_idx)
    s_out_sampler = SubsetRandomSampler(s_out_idx)
    t_train_sampler = SubsetRandomSampler(t_train_idx)
    t_out_sampler = SubsetRandomSampler(t_out_idx)
       

    #In our implementation we are keeping validation set to measure training performance
    #From the held out set for target and shadow, we take n_validation samples. 
    #As train set is already small we decided to take valid samples from held out set
    #as these are samples not used in training. 
    if dataset == 'CIFAR10':
        target_val_idx = t_out_idx[:n_validation]
        shadow_val_idx = s_out_idx[:n_validation]
    
        t_val_sampler = SubsetRandomSampler(target_val_idx)
        s_val_sampler = SubsetRandomSampler(shadow_val_idx)
    else:
        target_val_idx = t_out_idx[:n_validation]
        shadow_val_idx = s_out_idx[:n_validation]

        t_val_sampler = SubsetRandomSampler(target_val_idx)
        s_val_sampler = SubsetRandomSampler(shadow_val_idx)
    

    #-------------------------------------------------
    # Data loader
    #-------------------------------------------------
    if dataset == 'CIFAR10':

        t_train_loader = torch.utils.data.DataLoader(dataset=train_set, 
                                            batch_size=batch, 
                                            sampler = t_train_sampler,
                                            num_workers=num_workers)
                                            
        t_out_loader = torch.utils.data.DataLoader(dataset=train_set,
                                            batch_size=batch,
                                            sampler = t_out_sampler,
                                            num_workers=num_workers)
                                            
        t_val_loader = torch.utils.data.DataLoader(dataset=train_set,
                                            batch_size=batch,
                                            sampler=t_val_sampler,
                                            num_workers=num_workers)
        
        s_train_loader = torch.utils.data.DataLoader(dataset=train_set,
                                            batch_size=batch,
                                            sampler=s_train_sampler,
                                            num_workers=num_workers)
                                            
        s_out_loader = torch.utils.data.DataLoader(dataset=train_set,
                                            batch_size=batch,
                                            sampler=s_out_sampler,
                                            num_workers=num_workers)
        
        s_val_loader = torch.utils.data.DataLoader(dataset=train_set,
                                            batch_size=batch,
                                            sampler=s_val_sampler,
                                            num_workers=num_workers)

    else:
        t_train_loader = torch.utils.data.DataLoader(dataset=train_set, 
                                            batch_size=batch, 
                                            sampler=t_train_sampler,
                                            num_workers=num_workers)
    
        t_out_loader = torch.utils.data.DataLoader(dataset=train_set,
                                            batch_size=batch,
                                            sampler=t_out_sampler,
                                            num_workers=num_workers)
        
        t_val_loader = torch.utils.data.DataLoader(dataset=train_set,
                                            batch_size=batch,
                                            sampler=t_val_sampler,
                                            num_workers=num_workers)
        
        s_train_loader = torch.utils.data.DataLoader(dataset=train_set,
                                            batch_size=batch,
                                            sampler=s_train_sampler,
                                            num_workers=num_workers)
                                            
        s_out_loader = torch.utils.data.DataLoader(dataset=train_set,
                                            batch_size=batch,
                                            sampler=s_out_sampler,
                                            num_workers=num_workers)
        
        s_val_loader = torch.utils.data.DataLoader(dataset=train_set,
                                            batch_size=batch,
                                            sampler=s_val_sampler,
                                            num_workers=num_workers)
    

    if not test_flag:
        print('Total Test samples in {} dataset : {}'.format(dataset, len(test_set))) 
        print('Total Train samples in {} dataset : {}'.format(dataset, len(train_set)))  
        print('Number of Target train samples: {}'.format(len(t_train_sampler)))
        print('Number of Target valid samples: {}'.format(len(t_val_sampler)))
        print('Number of Target test samples: {}'.format(len(t_out_sampler)))
        print('Number of Shadow train samples: {}'.format(len(s_train_sampler)))
        print('Number of Shadow valid samples: {}'.format(len(s_val_sampler)))
        print('Number of Shadow test samples: {}'.format(len(s_out_sampler)))
   

    return t_train_loader, t_val_loader, t_out_loader, s_train_loader, s_val_loader, s_out_loader


def test_attack(attack_type,
                model,
                test_X,
                test_Y,
                device):
    
    print('----Attack Model Testing----')

    if attack_type == 'MemInf':
        targetnames= ['Non-Member', 'Member']
    else: #AttrInf
        targetnames = ['White', 'Black', 'Asian', 'Indian', 'Others']

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
        for _, (inputs, labels) in enumerate(dataloader,0):
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
    print('Attack Test Accuracy is  : {:.2f}%'.format(100*attack_acc))
    
    true_y = torch.cat(true_y).numpy()
    pred_y = torch.cat(pred_y).numpy()

    print('---Detailed Results----')
    print(classification_report(true_y,pred_y, target_names=targetnames))

##################################################################
# Membership Inference Attack Baseline
##################################################################
def train_mem_baseline(args,dataloaders,device,modelDir):

    if args.inference:
        #Test attack with pretrained target model
        test_attack_pretrained_target(args.attack,
                                    args.dataset,
                                    args.modelCkpt,
                                    args.modelPath,
                                    dataloaders[0], #Target train loader
                                    dataloaders[2], # Target Test Loader
                                    device)

    else:
        #For using top 3 posterior probabilities 
        top_k = args.need_topk

        if args.dataset == 'CIFAR10':    
            img_size = 32
            #Input Channels for the Image
            input_dim = 3        
            target_model = model.TargetNet(input_dim,target_filters,img_size,num_classes).to(device)
            shadow_model = model.ShadowNet(input_dim,shadow_filters,img_size,num_classes).to(device)
        else:
            img_size = 28
            input_dim = 1
            target_model = model.MNISTNet(input_dim, n_hidden_mnist, num_classes).to(device)
            #Using less hidden units than target model to mimic the architecture
            n_shadow_hidden = 16 
            shadow_model = model.MNISTNet(input_dim,n_shadow_hidden,num_classes).to(device)

        if (args.param_init):
            #Initialize params
            target_model.apply(w_init)
            shadow_model.apply(w_init)
        
        
        # Print the model we just instantiated
        if args.verbose:
            print('----Target Model Architecure----')
            print(target_model)
            print('----Model Learnable Params----')
            for name,param in target_model.named_parameters():
                if param.requires_grad == True:
                    print("\t",name)
        

        # Loss and optimizer for Taget Model
        target_loss = nn.CrossEntropyLoss()
        
        target_optimizer = torch.optim.Adam(target_model.parameters(), lr=learning_rate, weight_decay=reg)
        lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(target_optimizer,gamma=lr_decay)

        
        targetX, targetY = train_model(args.attack,
                                target_model,
                                dataloaders[0],
                                dataloaders[1],
                                dataloaders[2],
                                target_loss,
                                target_optimizer,
                                lr_scheduler,
                                device,
                                modelDir,
                                args.verbose,
                                num_epochs,
                                top_k,
                                need_earlystop,
                                is_target=True)

    
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

        shadowX, shadowY = train_model(args.attack,
                                shadow_model,
                                dataloaders[3],
                                dataloaders[4],
                                dataloaders[5],
                                shadow_loss,
                                shadow_optimizer,
                                shadow_lr_scheduler,
                                device,
                                modelDir,
                                args.verbose,
                                num_epochs,
                                top_k,
                                need_earlystop,
                                is_target=False)
      

        ###################################
        # Attack Model Training
        ##################################
        #The input dimension to MLP attack model
        input_size = shadowX[0].size(1)
        print('Input Feature dim for Attack Model : [{}]'.format(input_size))
    
        attack_model = model.AttackMLP(input_size,n_hidden,out_classes).to(device)
    
        if (args.param_init):
            #Initialize params
            attack_model.apply(w_init)

        # Loss and optimizer
        attack_loss = nn.CrossEntropyLoss()
        attack_optimizer = torch.optim.Adam(attack_model.parameters(), lr=LR_ATTACK, weight_decay=REG)
        attack_lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(attack_optimizer,gamma=LR_DECAY)

        #Feature vector and labels for training Attack model
        attackdataset = (shadowX, shadowY)
    
        attack_valacc = train_attack_model(args.attack,
                                    attack_model, 
                                    attackdataset, 
                                    attack_loss,
                                    attack_optimizer, 
                                    attack_lr_scheduler, 
                                    device,
                                    modelDir,
                                    NUM_EPOCHS, 
                                    BATCH_SIZE, 
                                    num_workers, 
                                    args.verbose)
   
  
        print('Validation Accuracy for the Best Attack Model is: {:.2f} %'.format(100* attack_valacc))
   
        #Load the trained attack model
        attack_path = os.path.join(modelDir,'best_attack_model.ckpt')
        attack_model.load_state_dict(torch.load(attack_path))
    
        #Inference on trained attack model
        test_attack(args.attack, attack_model, targetX, targetY, device)



#############################################################
# Attribute Inference Attack baseline
#############################################################
def train_attr_baseline(args,
                        dataset,
                        dataFolder,
                        modelFolder,
                        device,
                        top_k,
                        verbose):
 
    utk_img_size = 50
    #Input Channels for the Image
    utk_input_dim = 3 
    #Gender classfication
    utk_num_classes = 2
    #Filters
    utk_cnn_filters = [16, 32, 64] 
    #Latent Vector
    n_z = 64

    attack_type = 'AttrInf'
    utk_filename = 'utk_resize.npz'
    
    #Data transforms
    data_transform, _ = get_data_transforms(dataset)


    #UTKFace dataset
    utk_dataset = UTKFace(dataFolder,utk_filename, transform=data_transform)

    #Train/Test split - 80:20
    total_samples = len(utk_dataset)
    indices = list(range(total_samples))
    np.random.shuffle(indices)

    split = int(np.floor(ATTR_TEST_SPLIT * total_samples))
    train_indices, test_indices = indices[split:], indices[:split]

    attack_split = int(np.floor(ATTR_ATTACK_SPLIT * len(train_indices)))
    np.random.shuffle(train_indices)
    attack_indices, target_indices = train_indices[attack_split:], train_indices[:attack_split]

    target_train_sampler = SubsetRandomSampler(target_indices)
    attack_train_sampler = SubsetRandomSampler(attack_indices)
    test_sampler = SubsetRandomSampler(test_indices)
    val_sampler = SubsetRandomSampler(test_indices[:n_validation])
   
    target_train_loader = torch.utils.data.DataLoader(dataset=utk_dataset,
                                            batch_size=ATTR_BATCH_SIZE,
                                            sampler = target_train_sampler,
                                            num_workers=num_workers)

    attack_train_loader = torch.utils.data.DataLoader(dataset=utk_dataset,
                                            batch_size=ATTR_BATCH_SIZE,
                                            sampler = attack_train_sampler,
                                            num_workers=num_workers)

   
    test_loader = torch.utils.data.DataLoader(dataset=utk_dataset,
                                            batch_size=ATTR_BATCH_SIZE,
                                            sampler = test_sampler,
                                            num_workers=num_workers)

    val_loader = torch.utils.data.DataLoader(dataset=utk_dataset,
                                            batch_size=ATTR_BATCH_SIZE,
                                            sampler=val_sampler,
                                            num_workers=num_workers)
                                            
    if not args.inference:
        print('Total Test samples from {} dataset : {}'.format(dataset, len(test_indices))) 
        print('Total Train samples from {} dataset : {}'.format(dataset, len(train_indices)))  
        print('#Target Model train samples: {}'.format(len(target_train_sampler)))
        print('#Attack Model train samples: {}'.format(len(attack_train_sampler)))
        print('#Validation samples: {}'.format(len(val_sampler)))
        print('#Test samples: {}'.format(len(test_sampler)))

                                           

        #Target Model class object
        attr_target_model = model.AttrTargetNet(utk_input_dim,utk_cnn_filters,utk_img_size,utk_num_classes,n_z).to(device)

        #Loss, optimizer and scheduler
        attr_target_loss = nn.CrossEntropyLoss()
    
        attr_target_optimizer = torch.optim.Adam(attr_target_model.parameters(), lr=ATTR_LEARNING_RATE, weight_decay=ATTR_L2_RATIO)
        lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(attr_target_optimizer,gamma=ATTR_LR_DECAY)
    

        test_dataset = train_model(attack_type, #Attack Type 
                        attr_target_model,
                        target_train_loader,
                        val_loader,
                        test_loader,
                        attr_target_loss,
                        attr_target_optimizer,
                        lr_scheduler,
                        device,
                        modelFolder,
                        verbose,
                        ATTR_NUM_EPOCHS,
                        top_k,
                        need_earlystop,
                        is_target=True)

    

        #Attack Training Data - Feauture Representation from Last FC layer of Target Model
        attackX, attackY = get_feature_representation(attr_target_model,modelFolder,attack_train_loader,device)

        attack_dataset = (attackX, attackY)
    
        #Attack Model training for Race identification using features from Last FC layer of Target Model
        attr_input_size = attackX[0].size(1)
        print('Input Dimensions to the Attack Model = [{}]'.format(attr_input_size))

        #Classes for Race Classification
        attr_num_classes = 5

        #Attack Model class object
        attackmodel = model.AttackMLPAttr(attr_input_size, FC_DIM, attr_num_classes).to(device)


        #Loss, optimizer and Scheduler
        attack_loss = nn.CrossEntropyLoss()
        attack_optimizer = torch.optim.Adam(attackmodel.parameters(), lr=ATTR_LEARNING_RATE, weight_decay=ATTACK_L2_RATIO)
        attack_lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(attack_optimizer,gamma=ATTR_LR_DECAY)

        attack_valacc = train_attack_model(attack_type,
                                    attackmodel, 
                                    attack_dataset, 
                                    attack_loss,
                                    attack_optimizer, 
                                    attack_lr_scheduler, 
                                    device, 
                                    modelFolder,
                                    ATTR_NUM_EPOCHS, 
                                    ATTACK_BATCH, 
                                    num_workers, 
                                    verbose)

        print('Validation Accuracy for the Best Attack Model is: {:.2f} %'.format(100* attack_valacc))
   
        #Load the trained attack model
        attack_model_path = os.path.join(modelFolder,'best_attack_model.ckpt')
        attackmodel.load_state_dict(torch.load(attack_model_path))


        test_X , test_Y = test_dataset
    
        #Inference on trained attack model
        test_attack(attack_type,attackmodel, test_X, test_Y, device)
    else:
        #Test attack with pretrained target model
        test_attack_pretrained_target(attack_type,
                                    dataset,
                                    args.modelCkpt,
                                    args.modelPath,
                                    target_train_loader,
                                    test_loader,
                                    device)


def load_trained_target_model(attack_type, dataset, model_path,device):

    #Below code is to match our own target model architecture for Membership and Attr Inference. 
    # To use other model architecture, define model class in model.py and instantiate here with the parameters needed.
    if attack_type == 'MemInf':
        
        if dataset == 'CIFAR10': 
            img_size = 32
            #Input Channels for the Image
            input_dim = 3    

            target_model = model.TargetNet(input_dim,target_filters,img_size,num_classes).to(device)
        else:
            img_size = 28
            input_dim = 1
            target_model = model.MNISTNet(input_dim, n_hidden_mnist, num_classes).to(device)
        
         #load state dict, this is common code for any target model
        target_model.load_state_dict(torch.load(model_path))

    else: #Attribute Inference        
        utk_img_size = 50
        #Input Channels for the Image
        utk_input_dim = 3 
        #Gender classfication
        utk_num_classes = 2
        #Filters
        utk_cnn_filters = [16, 32, 64] 
        #Latent Vector
        n_z = 64

        target_model = model.AttrTargetNet(utk_input_dim,utk_cnn_filters,utk_img_size,utk_num_classes,n_z).to(device)
       
    return target_model

def test_attack_pretrained_target(attack_type,
                                dataset,
                                model_file,
                                model_path,
                                target_loader,
                                test_loader,
                                device):
    
    print('-----Testing Attack performace on pretrained target model-----')

    attack_model_path = os.path.join(model_path,'best_attack_model.ckpt')

    
    if model_file == '':
        print('Give a valid pretraind target mode file name')
        exit()
    
    if os.path.exists(model_path) == False:
        print('Path to fetch pretrained model from does not exist.')
        exit()

    target_model_path = os.path.join(model_path,model_file)
    print('Target Model Path Used is [{}]'.format(target_model_path))

    targetmodel = load_trained_target_model(attack_type,dataset, target_model_path, device)

   
    if attack_type == 'AttrInf':
        attackX, attackY = get_feature_representation(targetmodel,model_path,target_loader,device)
        
        #Attack Model training for Race identification using features from Last FC layer of Target Model
        attr_input_size = attackX[0].size(1)
        print('Input Dimensions to the Attack Model = [{}]'.format(attr_input_size))

        #Classes for Race Classification
        attr_num_classes = 5

        #Attack Model class object
        attr_attack_model = model.AttackMLPAttr(attr_input_size, FC_DIM, attr_num_classes).to(device)

        #Load state dict
        attr_attack_model.load_state_dict(torch.load(attack_model_path))

        #Test Attack
        test_attack(attack_type,attr_attack_model, attackX, attackY, device)

         
    else:
        targetmodel.load_state_dict(torch.load(target_model_path))
        t_trainX, t_trainY = prepare_attack_data(targetmodel,target_loader,device)
        t_testX, t_testY = prepare_attack_data(targetmodel,test_loader,device,test_dataset=True)
        attackX = t_trainX + t_testX
        attackY = t_trainY + t_testY

        input_size = attackX[0].size(1)
        print('Input Feature dim for Attack Model : [{}]'.format(input_size))

        #Instantiate model class
        attack_model = model.AttackMLP(input_size,n_hidden,out_classes).to(device)
        #Load state dict
        attack_model.load_state_dict(torch.load(attack_model_path))

        #Test Attack
        test_attack(attack_type,attack_model, attackX, attackY, device)

    
#######################################################
# Main Method to invoke model training and attack
#######################################################
def main(args):
 

    dataset = args.dataset
    need_augm = args.need_augm
    attack_type = args.attack
    dataPath = args.dataPath
    modelPath = args.modelPath

    if (dataPath != './data') and os.path.exists(dataPath) == False:
        print('The data path provided does not exist. Kindly run the cli command again with correct path')
        exit()
    
    if (modelPath != './model') and os.path.exists(modelPath) == False:
        print('The model path provided does not exist. Kindly run the cli command again with correct path')
        exit()
    
    
    if dataset == 'CIFAR10' or dataset == 'MNIST':
        # Directory where Pytorch datasets will be be downloaded
        datasetDir = os.path.join(dataPath,dataset)

        if not os.path.exists(datasetDir):
            try:        
                os.makedirs(datasetDir)
            except OSError:
                    pass
    elif dataset == 'UTKFace':
        if not os.path.exists(dataPath):
            print('UTKFace data directory not found. Kindly give the correct path')
            exit()

    if not args.inference:
        modelDir = os.path.join(modelPath, dataset)      
        if not os.path.exists(modelDir):
            try:
                os.makedirs(modelDir)
            except OSError:
                pass
    else:
        #Just assignment, not used during test time
        modelDir = args.modelPath 
   
    # setting device on GPU if available, else CPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Using device:', device)

 
    if attack_type == 'MemInf':
        if not args.inference:
            print('-----Executing Membership Inference Attack------')
            test_flag = False
        else:
            test_flag = True

        dataloaders = get_data_loader(dataset, 
                                    datasetDir,
                                    batch_size,
                                    shadow_split,
                                    need_augm,
                                    test_flag,
                                    num_workers)
    
        train_mem_baseline(args,dataloaders,device,modelDir)

    elif attack_type == 'AttrInf':
        if not args.inference:
            print('-----Executing Attribute Inference Attack------')
        top_k = args.need_topk
        train_attr_baseline(args,dataset,dataPath,modelDir,device,top_k,args.verbose)
        
    else: #attack_type == 'ModelInv'
        restuls_dir = './results'
        if not os.path.exists(restuls_dir):
            try:        
                os.makedirs('./results')
            except OSError:
                pass
        
        if not args.inference:
            print('-----Executing Model Inversion Attack------')
            modelDir = os.path.join(modelPath, dataset)      
            if not os.path.exists(modelDir):
                try:
                    os.makedirs(modelDir)
                except OSError:
                    pass

            model_file = 'atnt-mlp-model.pt'
            target.train_target(args.dataPath, modelDir)
            mn.label_reconstruction(args.dataPath,modelDir, model_file, args.label_idx, args.num_iter)
        else:
            mn.label_reconstruction(args.dataPath,args.modelPath, args.modelCkpt, args.label_idx, args.num_iter)


        

if __name__ == '__main__':
    #command line arguments from the user
    user_args = get_cmd_arguments()
    if user_args.verbose:
        print(user_args)
    main(user_args)