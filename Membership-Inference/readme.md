## Overview

This is the implementation of Membership Inference Attack (Adversay 1) from ML Leaks [1] paper. In this setting, the attacker/ adversary aims to determine whether a data sample is used to train a ML model(aka target model) or not. In the implementation we have assumed the adversary knows the training data distribution of the target model as well as its architecture and hyperparameters used to train the model. 


## Dataset
CIFAR10, MNIST from Pytorch dataset library

## Requirements
We tested the code on Google Colab with 'GPU' backend. The below versions of Python and Pytorch were selected by default. 

Python 3.7.10
Pytroch 1.9.0

We have not tested our code with Python version 2.7. The code should work with Python version 3.7.10 and above. We also recommend to use latest stable release (1.9.0) of Pytorch to avoid any errors during execution.


## File structure:  

    Membership-Inferfernece
     |- attack.py (main file that initiates the memebership inference attack)
     |- model.py  (target, shadow and attack model architectures)
     |- train.py  (model training and evaluation)
     |- README.md

## Data and Model Folders
  
   model (for model checkpoints)
     |- CIFAR10
     |- MNIST

   data (dataset)
     |- CIFAR10
     |- MNIST

* The two main folders and sub folders are created automatically, if not present, in the working directory from the where the code is run. User can specify explicit data and model path using the command line arguments '--dataPath' and '--modelPath'. In this case the sub-folders for respective datasets will then be created during the execution.

* The model checkpoints during training are saved in the respective folder the dataset. 
e.g : /model/CIFAR10 (default path)

* Each dataset is downloaded automatically by Pytorch(if not present) in the respective dataset folder.
e.g: /data/CIFAR10 (default path)

* The best model checkpoints are named as 'best_shadow_model.ckpt', 'best_target_model.ckpt', and 'best_attack_model.ckpt' respectively.

NOTE: We have provided the best models from our test run. They are in the 'best_models' folder in the main code repository.


## Run Experiements:

NOTE : 1) Keep all the code (.py) files in the same folder before starting the execution.
       2) For running without the 'cli.py', uncomment the '__main__' function in the attack.py and run the attack using the below commands.

* Membership inference attack with target and shadow model training enabled:
```
  python attack.py --dataset CIFAR10 --trainTargetModel --trainShadowModel 
```

Specify the dataset to be used as part of '--dataset' argument.


* Membership inference attack with target,shadow model training enabled and using top 3 posteriors for training the attack model:
```
  python attack.py --dataset CIFAR10 --trainTargetModel --trainShadowModel  --need_topk
```

* Membership inference attack using pre-trained target and shadow models stored in the 'model' folder:
```
  python attack.py --dataset CIFAR10
```
NOTE: To use the pre-trained target and shadow models provided by us, copy them from 'best_models' folder into 'model/{CIFAR10, MNIST}'.

* Details about all the command line arguments can be found using the following command:
```
  python attack.py --help 
```
    usage: Membership Inference Attack [-h] [--dataset {CIFAR10,MNIST}]
                                       [--dataPath DATAPATH]
                                       [--modelPath MODELPATH]
                                       [--trainTargetModel] [--trainShadowModel]
                                       [--need_augm] [--need_topk] [--param_init]
                                       [--verbose]

    optional arguments:
      -h, --help            show this help message and exit
      --dataset {CIFAR10,MNIST}
                            Which dataset to use (CIFAR10 or MNIST)
      --dataPath DATAPATH   Path to store data
      --modelPath MODELPATH
                            Path to save or load model checkpoints
      --trainTargetModel    Train a target model, if false then load an already
                            trained model
      --trainShadowModel    Train a shadow model, if false then load an already
                            trained model
      --need_augm           To use data augmentation on target and shadow training
                            set or not
      --need_topk           Flag to enable using Top 3 posteriors for attack data
      --param_init          Flag to enable custom model params initialization
      --verbose             Add Verbosity



## References:
1. Salem, A., Zhang, Y., Humbert, M., Berrang, P., Fritz, M. and Backes, M., 2018. Ml-leaks: Model and data independent membership inference attacks and defenses on machine learning models. arXiv preprint arXiv:1806.01246.
2. Shokri, R., Stronati, M., Song, C. and Shmatikov, V., 2017, May. Membership inference attacks against machine learning models. In 2017 IEEE Symposium on Security and Privacy (SP) (pp. 3-18). IEEE.
3. https://github.com/AhmedSalem2/ML-Leaks
4. https://github.com/Lab41/cyphercat

