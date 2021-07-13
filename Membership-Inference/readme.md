# Membership Inference

* We have implemented the Membership Inference (Attack 1) from ML Leaks [1] where the task is to classify whether a sample belongs to traning set of the model (target) we are attacking (using the attack model). 

* Attack model is a binary classifer to detect whether an input is used in the training set of the target model. Shadow model is used to mimic the target model and used to train the attack model using the posteriros. Check ML-Leaks - Section III (Adversary 1) [1] for more details.

* Target model and shadow models are CNN based architectures with two convolution layers followed by batch normalization and max-pooling and one fully connected layer. We use ReLU activation for non-linearity. Attack model is an MLP with one hidden layer and ReLU activation for non-linearity.

* We assume that we know the data distribution on which the target model is trained is same on which our shadow model is trained. Also, we know the architecture and hyperparameters of the target model which we can use to train out shadow model.

* We are using the datasets from PyTorch library - MNIST and CIFAR10. 

## Implemention

### Folder structure:  

    1. '/model/{MNIST, CIFAR10}' 
    2. '/data/{MNIST, CIFAR10}' 

* Both 'data' and 'model' folder and sub-folders are created automatically if not present; will be created in the same directory from where the code is run.

* 'model' folder saves model checkpoints and 'data' folder saves the dataset (in the respective sub-folders).

* Best model checkpoints 'best_shadow_model.ckpt', 'best_target_model.ckpt', and 'best_attack_model.ckpt' are stored in the 'model' folder (as specified above) during the run. 

* NOTE: To find the best models saved from our run, check 'best_models' folder.

### Files: 

* The following files are present in 'Membership-Inference' folder:

    1. 'attack.py'
    2. 'model.py' 
    3. 'train.py'

* 'attack.py' is the main file that initiates the memebership inference attack.

* 'model.py' consists target, shadow and attack model architectures.

* 'train.py' contains methods for model training and evaluation.

* NOTE: Please keep all the above three files in the same folder at the time of execution.

### How to run:

* NOTE: Code tested on Google Colab with runtime type as 'GPU'. 

* Requirements: Python 3.7.10, Pytroch 1.9.0+cu102

* Please refrain from using the arguments not specified in the commands below.
---------------------------------------------------------------------------------

* With target and shadow model training enabled (for the first timers):
```
$ python attack.py --trainTargetModel --trainShadowModel --dataset CIFAR10
```

* With target and shadow model training enabled and using top 3 posteriors for training the attack model:
```
$ python attack.py --trainTargetModel --trainShadowModel --dataset MNIST --need_topk
```

* With pre-trained target and shadow models stored in the 'model' folder:
```
$ python attack.py --dataset CIFAR10
```

* Details about all the arguments can be found using the following command:
```
$ python attack.py --help 
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

