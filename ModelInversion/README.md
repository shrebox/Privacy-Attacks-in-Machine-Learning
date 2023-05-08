## Model Inversion

Contents:
 * 'data_pgm', folder with the ATnT face dataset containing 40 classes, each with 10 different images of the same person.
 * 'atnt-mlp-model.pt', trained target model.
 * 'results', folder with results for 30 (specific class) and 100 (all classes) iterations of mi_face algorithm.
 * 'target_model.py', the target model we are using.
 * 'model_inversion.py' our implementation of mi_face algorithm and some functions to perform the attack with the target model we are using.

### target_model.py
* Our target model is a simple MLP based on the PyTorch Tutorial form lecture and https://github.com/bentrevett/pytorch-image-classification/blob/master/1_mlp.ipynb and the ATNT-Dataset from https://github.com/harveyslash/Facial-Similarity-with-Siamese-Networks-in-Pytorch. 
* Pre-trained model 'atnt-mlp-model.pth' can be downloaded from https://drive.google.com/file/d/10Swb7sddVHNJWqBe3Bx3zaUifSy1it1r/view?usp=sharing and put in the 'ModelInversion/' folder.

### model_inversion.py
We perform a model inversion attack similar to Fredrikson et al. More concrete we perform an reconstruction attack, where given a label we reconstruct an image of the respective class.
This file contains our implementation of mi_face algorithm from Fredrikson et al. We perform a simple gradient descent, where we query the model with a random image, and the try to minimize 1-probability(label).

### Note:

* Please use the shipped dataset
* Code tested on Google Colab with runtime type as GPU. 

## References
1. Fredrikson, M., Jha, S. and Ristenpart, T., 2015, October. Model inversion attacks that exploit confidence information and basic countermeasures. In Proceedings of the 22nd ACM SIGSAC conference on computer and communications security (pp. 1322-1333)
2. https://github.com/bentrevett/pytorch-image-classification/blob/master/1_mlp.ipynb
3. https://github.com/Lab41/cyphercat
4. https://github.com/harveyslash/Facial-Similarity-with-Siamese-Networks-in-Pytorch
