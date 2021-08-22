# PETS-Project

Course homepage: https://cms.cispa.saarland/pets2021/

Project guidelines: https://docs.google.com/document/d/141wj_FKeYANgSlLN2ECfQgMCdQgV_bTsiT82GYtNp6Y/edit?usp=sharing

Project references: https://docs.google.com/document/d/1pcAXpFTQDd4hGnLlhSa4PsQSv3HDJ8avuR-5pppO9NA/edit?usp=sharing

Timeline: https://docs.google.com/spreadsheets/d/1dKo0ajoctzqFjUMUhbgnqzykvuLyO6nnrc5uTpatqck/edit?usp=sharing

Results and Inferences: https://docs.google.com/document/d/18JmdYclua216xaiem06lATwA-hMAecLLGZL-DNDFRd0/edit?usp=sharing


###########################################################################################################
###########################################################################################################


# Membership Inference

## How to run
`$ python cli.py membership-inference [OPTIONS] COMMAND [ARGS]...`

* Options:
  * `--help`  Show this message and exit.

* Commands:
  * `pretrained-dummy`  Load trained target and shadow model and train attack model
  * `train-dummy`     Train target, shadow and attack model
  * `train-plus-dummy`       Train target, shadow and attack model + augmentation, topk posteriors, parameter initialization and verbose enabled

* Options:
  * `--dataset TEXT`     Which dataset to use (CIFAR10 or MNIST) [Default = CIFAR10]
  * `--data_path TEXT`   Path to store data [Default = 'Membership-Inference/data']
  * `--model_path TEXT`  Path to save or load model checkpoints [Default = 'Membership-Inference/model']
  * `--help`   Show this message and exit.

* Example commands:
  * `$ python cli.py membership-inference pretrained-dummy --dataset MNIST --data_path PATH1 --model_path PATH2`
  * `$ python cli.py membership-inference train-dummy`
  * `$ python cli.py membership-inference train-plus-dummy`


###########################################################################################################
###########################################################################################################


# Attribute Inference

## How to run
`$ python cli.py attribute-inference [OPTIONS] COMMAND [ARGS]...`

* Options:
  * `--help`  Show this message and exit.

* Commands:
  * `pretrained-dummy`  Load trained target and attack model
  * `supply-target`     Supply own target model and train attack model
  * `train-dummy`       Train target and attack model

#### `pretrained-dummy`
* `$ python cli.py attribute-inference pretrained-dummy [OPTIONS]`

  * Load trained target and attack model

* Options:
  * `--help`  Show this message and exit.

* Example command:
  * `$ python cli.py attribute-inference pretrained-dummy`


#### `train-dummy`
* `$ python cli.py attribute-inference train-dummy [OPTIONS]`

  * Train target and attack model

* Options:
  * `-t`, `--target_epochs` `INTEGER`  Number of training epochs for the target model [default=30]
  * `-a`, `--attack_epochs` `INTEGER`  Number of training epochs for the attack model [default=50]
  * `--help`                      Show this message and exit.

* Example command:
  * `$ python cli.py attribute-inference train-dummy -t 30 -a 50`

#### `supply-target`

* `$ python cli.py attribute-inference supply-target [OPTIONS]`

  * Supply own target model and train attack model.
  * Specifications for the target Model:
    * The target model should predict gender of a human [0: male, 1:female] and trained in UTKFace dataset.
    * The target model should give the following output: `y, x` where y are the two posteriors and y is the last fully connected layer. E.g. `[ 5.0912e-01, -5.4544e-01], [-0.0656,  0.0087, -0.0543,  ...,  0.0134,  0.0608, -0.0347]`
    * The class file needs to be in the attribute inference folder.
    * Name of the class needs to be 'TargetModel'.

* Options:
  * `-c`, `--class_file` `TEXT`        File that holds the target models nn.Module class  [required]
  * `-s`, `--state_path` `TEXT`        Path of the state dictionary  [required]
  * `-d`, `--dimension` `INTEGER`      Flattend dimension of the layer used as attack modelinput   [required]
  * `-a`, `--attack_epochs` `INTEGER`  Number of training epochs for the attack model [default=30]
  * `--help`                       Show this message and exit.

* Example command:
  * `$ python cli.py attribute-inference supply-target -c af_models -s Attribute-Inference/models/target_model_30.pth -d 64 -a 50`


###########################################################################################################
###########################################################################################################


# Model Inversion

## How to run
`$ python cli.py model-inversion [OPTIONS] COMMAND [ARGS]...`

* Options:
  * `--help`  Show this message and exit.

* Commands:
  * `pretrained-dummy`  Load trained target model and perform inversion
  * `train-dummy`       Train target model and perform model inversion
  * `supply-target`     Use trained external target model and perform model inversion

#### `pretrained-dummy`
* `$ python cli.py model-inversion pretrained-dummy [OPTIONS]`

  * Load trained target model and perform inversion
  * Pretrained model 'atnt-mlp-model.pth' is in the ModelInversion folder.

* Options:
  * `--iterations INTEGER`   Number of Iterations in attack [default = 30]
  * `--loss_function [crossEntropy|softmax]`  which loss function to used crossEntropy or softmax [default = crossEntropy]
  * `--generate_specific_class INTEGER`     choose class, number between 1 and 40, which you want recovered or nothing to get all recovered [default = -1]
  * `--help`  Show this message and exit.

* Example command:
  * `$ python cli.py model-inversion pretrained-dummy`


#### `train-dummy`
* `$ python cli.py model-inversion train-dummy [OPTIONS]`

  * Train target model and perform model inversion

* Options:
  * `--iterations INTEGER`   Number of Iterations in attack [default = 30]
  * `--epochs INTEGER`      Number of epochs for the target model [default = 30]
  * `--loss_function [crossEntropy|softmax]`  which loss function to used crossEntropy or softmax [default = crossEntropy]
  * `--generate_specific_class INTEGER`     choose class, number between 1 and 40, which you want recovered or nothing to get all recovered [default = -1]
  * `--help`                      Show this message and exit.

* Example command:
  * `$ python cli.py model-inversion train-dummy --epochs 30`

#### `supply-target`

* `$ python cli.py model-inversion supply-target [OPTIONS]`

  * Use trained external target model and perform model inversion
  * Specifications for the target Model:
    * The target model should based on ATnT faces dataset.
    * The target model needs to return 'output, h' where output are the posteriors (h can be neglected).
    * The class file needs to be in the model inversion folder.
    * Name of the class needs to be 'TargetModel'.

* Options:
  * `--class_file TEXT`   File that holds the target models nn.Module class [required]
  * `--target_model_path TEXT`   target model file  [required]
  * `--iterations INTEGER`   Number of Iterations in attack [default = 30]
  * `--loss_function [crossEntropy|softmax]`  which loss function to used crossEntropy or softmax [default = crossEntropy]
  * `--generate_specific_class INTEGER`     choose class, number between 1 and 40, which you want recovered or nothing to get all recovered [default = -1]
  * `--help`   Show this message and exit.

* Example command:
  * `$ python cli.py model-inversion supply-target --class_file target_model --target_model_path ModelInversion/atnt-mlp-model.pth --generate_specific_class 25`


  ###########################################################################################################
  ###########################################################################################################





