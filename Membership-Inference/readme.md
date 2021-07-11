# Membership Inference - ML Leaks

## Folders:  

1. './model/{MNIST, CIFAR10}' --> Will be generated when code is run with --trainTarget and --trainShadow arguments else will load from these folders. 
2/ './data/{MNIST, CIFAR10}' --> Will download when code is run else will load from these folders. 

## Files: 

1. './attack.py' --> Main file to be run with arguments
2. './model.py' --> Containing model definitions for target, shadow and attack models
3. './train.py' --> Train 
4. './model/{MNIST, CIFAR10}/{best_shadow_model.ckpt, best_target_model.ckpt, best_attack_model.ckpt}/'

In which file datasets are generated??

## How to run:

$ attack.py --help --> To see all the possible argument options

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

[1] https://github.com/AhmedSalem2/ML-Leaks
[2] https://github.com/Lab41/cyphercat

