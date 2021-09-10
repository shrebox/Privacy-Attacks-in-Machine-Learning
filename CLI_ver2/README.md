# Readme

* Prerequisities:
	* Python 3.7.10 or above
	* Pytorch release 1.9.0 or above
	* GPU backend

* Folder structure:
	* Dataset: ./data/{DATASET_NAME}
	* Model Checkpoint: ./model/{DATASET_NAME}
	* Results for model inversion: ./results

* `$ python attack.py --help`
	
	```
	usage: attack.py [-h] --attack {MemInf,ModelInv,AttrInf} --dataset
	                 {CIFAR10,MNIST,UTKFace,ATTFace} [--dataPath DATAPATH]
	                 [--modelPath MODELPATH] [--inference] [--modelCkpt MODELCKPT]
	                 [--label_idx LABEL_IDX] [--num_iter NUM_ITER] [--need_augm]
	                 [--need_topk] [--param_init] [--verbose]

	Privacy attacks against Machine Learning Models

	optional arguments:
	  -h, --help            show this help message and exit
	  --attack {MemInf,ModelInv,AttrInf}
	                        Attack Type
	  --dataset {CIFAR10,MNIST,UTKFace,ATTFace}
	                        Which dataset to use for attack
	  --dataPath DATAPATH   Path to store or load data
	  --modelPath MODELPATH
	                        Path for model checkpoints
	  --inference           For testing attack performance
	  --modelCkpt MODELCKPT
	                        Trained Target model checkpoint file for test attack
	  --label_idx LABEL_IDX
	                        Label/Tag to be reconstructed in Model Inversion
	                        attack
	  --num_iter NUM_ITER   Number of Iterations for Model Inversion attack
	  --need_augm           To use data augmentation on target and shadow training
	                        set or not
	  --need_topk           Flag to enable using Top 3 posteriors for attack data
	  --param_init          Flag to enable custom model params initialization
	  --verbose             Add Verbosity

	```


## End to end training

`$ python attack.py --attack <ATTACK TYPE> --dataset <DATASET NAME> [OPTIONS]...`

### How to run

#### Membership inference
	
`$ python attack.py --attack MemInf --dataset CIFAR10`

#### Attribute inference
	
`$ python attack.py --attack AttrInf --dataset UTKFace --dataPath <PATH_TO_UTKFace_npz>`
	
* Example: `$ python attack.py --attack AttrInf --dataset UTKFace --dataPath ./data/UTKFace/`
* Note: 
	* Run load_data.py on the UTKFace dataset and copy the generated file 'utk_resize.npz' to the path specified by the dataPath parameter.
	* 'utk_resize.npz' is provided with the code in the './data/UTKFace'.

#### Model inversion
	
`$ python attack.py --attack ModelInv --dataset ATTFace --dataPath ./ModelInversion/ --num_iter <NUM_INTERATIONS> --label_idx <CLASS_LABEL>`
	
* Example: `$ python attack.py --attack ModelInv --dataset ATTFace --dataPath ./ModelInversion/ --num_iter 30 --label_idx 32`
* Note: ATnT Face dataset is inside ModelInversion directory under the folder 'data_pgm'.



## For inference (Testing attack performace with pre-trained models)

`$ python attack.py --attack <ATTACK TYPE> --dataset <DATASET NAME> --inference [OPTIONS]...`

### How to run

#### Membership inference
	
`$ python attack.py --attack MemInf --dataset CIFAR10 --inference --modelPath <PATH_TO_PRETRAINED_TARGET_MODEL> --modelCkpt <PRETRAINED_TARGET_MODEL_FILE>`

* Example: `$ python attack.py --attack MemInf --dataset CIFAR10 --inference --modelPath ./model/CIFAR10 --modelCkpt best_target_model.ckpt`

#### Attribute inference

`$ python attack.py --attack AttrInf --dataset UTKFace --inference --dataPath ./data/UTKFace/ --modelPath <PATH_TO_PRETRAINED_TARGET_MODEL> --modelCkpt <PRETRAINED_TARGET_MODEL_FILE>`
	
* Example: `$ python attack.py --attack AttrInf --dataset UTKFace --inference --dataPath ./data/UTKFace/ --modelPath ./model/UTKFace/ --modelCkpt best_target_model.ckpt`

#### Model inversion

`$ python attack.py --attack ModelInv --dataset ATTFace --inference --dataPath ./ModelInversion/ --modelPath <PATH_TO_PRETRAINED_TARGET_MODEL> --modelCkpt atnt-mlp-model.pt --label_idx <CLASS_LABEL> --num_iter <ITERATION_VALUE>`

* Example: `$ python attack.py --attack ModelInv --dataset ATTFace --inference --dataPath ./ModelInversion/ --modelPath ./model/ATTFace/ --modelCkpt atnt-mlp-model.pt --label_idx 40 --num_iter 30`

We would also like to provide supplemental information that we felt would help in smoother testing and provide more clarity in understanding the CLI structure better:

Common for all attacks:

1) '--inference' flag enables evaluation of the attack model provided by us with the user-specified pre-trained target model. No model training is done in this step.

2) The attack models for the 3 attacks are saved under the 'model/<dataset>' folder. They are saved with name 'best_attack_model.ckpt'. When running the end-to-end attack, they will be overwritten if the user doesn't specify a different model folder path. The default path is './model'.

3)  Both trained target and attack model files should be in the same folder for running attack evaluation. Also, do not rename the attack model files provided.

4) For attack evaluation, a user has to define the model class in 'model.py' and instantiate the class so that the model structure matches with the pre-trained target model.

Membership Inference:

5) The attack model provided for membership inference is trained on top 10 posteriors. For attack evaluation, we will fetch the top 10 posteriors from the target model. (Default case)

6) We do not ask for the dataset from the user for membership inference. Both CIFAR10 and MNIST datasets would be downloaded from Pytorch, if not present under <dataPath>/<dataset> folder. The data loaders are prepared according to the predefined splits in the code.
This behaviour is the same for both end-to-end training or attack evaluation. The reason for this design decision is to avoid any errors with the parsing of user-specified dataset files.

7) For membership inference, user can provide their own data and model path. We will download the datasets from Pytorch on the specified data folder, and the model checkpoints will be saved in the given model path. Note that for all attacks, the model files will be saved within the <dataset> folder either inside the default or user-specified model path.

Attribute Inference:

8) We expect the UTKFace dataset to be saved as a '.npz' file. Please use 'load_data.py' to generate the file from the UTKFace dataset or you can use the file we provided under the 'data/UTKFace' folder.

Model Inversion:

9) We use the AT&T dataset provided in the 'data_pgm' folder under 'ModelInversion'. As the dataset is not provided by Pytorch or any other open-source library and is not easily available, there are chances that it can be saved in different ways. To avoid any parsing errors, we have saved it in a structure that we expect at the time of parsing during the test run. Kindly do not move the data to any different folder or change the structure.

10) Reconstruction files will be saved under './results' folder during the test run.

11) For model inversion, a user should give label_idx as -1 to reconstruct all the 40 classes in the dataset, else specified the class label to be reconstructed.
	
## References:
1. Salem, A., Zhang, Y., Humbert, M., Berrang, P., Fritz, M. and Backes, M., 2018. Ml-leaks: Model and data independent membership inference attacks and defenses on machine learning models. arXiv preprint arXiv:1806.01246.
2. Shokri, R., Stronati, M., Song, C. and Shmatikov, V., 2017, May. Membership inference attacks against machine learning models. In 2017 IEEE Symposium on Security and Privacy (SP) (pp. 3-18). IEEE.
3. Fredrikson, M., Jha, S. and Ristenpart, T., 2015, October. Model inversion attacks that exploit confidence information and basic countermeasures. In Proceedings of the 22nd ACM SIGSAC conference on computer and communications security (pp. 1322-1333)
4. LeCun, Yann, et al. "Backpropagation applied to handwritten zip code recognition." Neural computation 1.4 (1989): 541-551.
5. Song, Congzheng, and Vitaly Shmatikov. "Overlearning reveals sensitive attributes." arXiv preprint arXiv:1905.11742 (2019).
6. https://github.com/AhmedSalem2/ML-Leaks
7. https://github.com/Lab41/cyphercat
8. https://github.com/bentrevett/pytorch-image-classification/blob/master/1_mlp.ipynb
9. https://github.com/Lab41/cyphercat
10. https://github.com/harveyslash/Facial-Similarity-with-Siamese-Networks-in-Pytorch
11. https://susanqq.github.io/UTKFace/
