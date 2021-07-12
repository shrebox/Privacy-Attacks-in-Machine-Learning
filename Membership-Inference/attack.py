#!/usr/bin/python

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data.dataset import TensorDataset
from torch.utils.data.sampler import SubsetRandomSampler
import model
from model import init_params as w_init
from train import  train_model, train_attack_model, prepare_attack_data
from sklearn.metrics import classification_report
from sklearn.metrics import precision_score, recall_score
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
target_filters = [128, 256, 256]
shadow_filters = [64, 128, 128]
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
#Number of processes
num_workers = 2
#Hidden units for MNIST model
n_hidden_mnist = 32


################################
#Attack Model Hyperparameters
################################
NUM_EPOCHS = 15
BATCH_SIZE = 20
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



def get_cmd_arguments():
    parser = argparse.ArgumentParser(prog="Membership Inference Attack")
    parser.add_argument('--dataset', default='CIFAR10', type=str, choices=['CIFAR10', 'MNIST'], help='Which dataset to use (CIFAR10 or MNIST)')
    parser.add_argument('--dataPath', default='./data', type=str, help='Path to store data')
    parser.add_argument('--modelPath', default='./model',type=str, help='Path to save or load model checkpoints')
    parser.add_argument('--trainTargetModel', action='store_true', help='Train a target model, if false then load an already trained model')
    parser.add_argument('--trainShadowModel', action='store_true', help='Train a shadow model, if false then load an already trained model')
    parser.add_argument('--need_augm',action='store_true', help='To use data augmentation on target and shadow training set or not')
    parser.add_argument('--need_topk',action='store_true', help='Flag to enable using Top 3 posteriors for attack data')
    parser.add_argument('--param_init', action='store_true', help='Flag to enable custom model params initialization')
    parser.add_argument('--verbose',action='store_true', help='Add Verbosity')
    return parser.parse_args()

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

    else:
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
        
    return train_transforms, test_transforms

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
    

#--------------------------------------------------------------------------------
# Get dataloaders for Shadow and Target models 
# Data Strategy:
# - Split the entire training dataset into 4 parts(T_tain, T_test, S_train, S_test)
#  Target -  Train on T_train and T_test
#  Shadow -  Train on S_train and S_test
#  Attack - Use T_train and T_test for evaluation
#           Use S_train and S_test for training
#--------------------------------------------------------------------------------
def get_data_loader(dataset,
                    data_dir,
                    batch,
                    shadow_split=0.5,
                    augm_required=False,
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
    

      
    print('Total Test samples in {} dataset : {}'.format(dataset, len(test_set))) 
    print('Total Train samples in {} dataset : {}'.format(dataset, len(train_set)))  
    print('Number of Target train samples: {}'.format(len(t_train_sampler)))
    print('Number of Target valid samples: {}'.format(len(t_val_sampler)))
    print('Number of Target out samples: {}'.format(len(t_out_sampler)))
    print('Number of Shadow train samples: {}'.format(len(s_train_sampler)))
    print('Number of Shadow valid samples: {}'.format(len(s_val_sampler)))
    print('Number of Shadow out samples: {}'.format(len(s_out_sampler)))
   

    return t_train_loader, t_val_loader, t_out_loader, s_train_loader, s_val_loader, s_out_loader


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
    print('Attack Test Accuracy is  : {:.2f}%'.format(100*attack_acc))
    
    true_y =  torch.cat(true_y).numpy()
    pred_y = torch.cat(pred_y).numpy()

    print('---Detailed Results----')
    print(classification_report(true_y,pred_y, target_names=targetnames))


#Main Method to initate model training and attack
def create_attack(args):
 
    dataset = args.dataset
    need_augm = args.need_augm
    verbose = args.verbose
    #For using top 3 posterior probabilities 
    top_k = args.need_topk

    if dataset == 'CIFAR10':
        img_size = 32
        #Input Channels for the Image
        input_dim = 3
    else:#MNIST
        img_size = 28
        input_dim = 1

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
                                                                num_workers)
    
     
    if (args.trainTargetModel):

        if dataset == 'CIFAR10':            
            target_model = model.TargetNet(input_dim,target_filters,img_size,num_classes).to(device)
        else:
            target_model = model.MNISTNet(input_dim, n_hidden_mnist, num_classes).to(device)

        if (args.param_init):
            #Initialize params
            target_model.apply(w_init) 
        
        
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
                                    top_k,
                                    is_target=True)
    else: #Target model training not required, load the saved checkpoint
        target_file = os.path.join(modelDir,'best_target_model.ckpt')
        print('Use Target model at the path ====> [{}] '.format(modelDir))
        #Instantiate Target Model Class
        if dataset == 'CIFAR10':
            target_model = model.TargetNet(input_dim,target_filters,img_size,num_classes).to(device)
        else:
            target_model = model.MNISTNet(input_dim,n_hidden_mnist,num_classes).to(device)

        target_model.load_state_dict(torch.load(target_file))
        print('---Peparing Attack Training data---')
        t_trainX, t_trainY = prepare_attack_data(target_model,t_train_loader,device,top_k)
        t_testX, t_testY = prepare_attack_data(target_model,t_test_loader,device,top_k,test_dataset=True)
        targetX = t_trainX + t_testX
        targetY = t_trainY + t_testY

    if (args.trainShadowModel):

        if dataset == 'CIFAR10':
            shadow_model = model.ShadowNet(input_dim,shadow_filters,img_size,num_classes).to(device)
        else:
            #Using less hidden units than target model to mimic the architecture
            n_shadow_hidden = 16 
            shadow_model = model.MNISTNet(input_dim,n_shadow_hidden,num_classes).to(device)

        if (args.param_init):
            #Initialize params
            shadow_model.apply(w_init)

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
                                    top_k,
                                    is_target=False)
    else: #Shadow model training not required, load the saved checkpoint
        print('Using Shadow model at the path  ====> [{}] '.format(modelDir))
        shadow_file = os.path.join(modelDir,'best_shadow_model.ckpt')
        assert os.path.isfile(shadow_file), 'Shadow Mode Checkpoint not found, aborting load'
        #Instantiate Shadow Model Class
        if dataset == 'CIFAR10':
            shadow_model = model.ShadowNet(input_dim,shadow_filters,img_size,num_classes).to(device)
        else:
            #Using less hidden units than target model to mimic the architecture
            n_shadow_hidden = 16
            shadow_model = model.MNISTNet(input_dim,n_shadow_hidden,num_classes).to(device)

        #Load the saved model
        shadow_model.load_state_dict(torch.load(shadow_file))
        #Prepare dataset for training attack model
        print('----Preparing Attack training data---')
        trainX, trainY = prepare_attack_data(shadow_model,s_train_loader,device,top_k)
        testX, testY = prepare_attack_data(shadow_model,s_test_loader,device,top_k,test_dataset=True)
        shadowX = trainX + testX
        shadowY = trainY + testY    
    

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
    
    attack_valacc = train_attack_model(attack_model, attackdataset, attack_loss,
                       attack_optimizer, attack_lr_scheduler, device, modelDir,
                        NUM_EPOCHS, BATCH_SIZE, num_workers, verbose)
   
  
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