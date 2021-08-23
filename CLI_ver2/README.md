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
