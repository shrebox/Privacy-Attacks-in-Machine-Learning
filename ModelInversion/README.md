## Model Inversion

Contents:
 * 'data_pgm', folder with the ATnT face dataset containing 40 classes, each with 10 different images of the same person.
 * 'atnt-mlp-model.pt', trained target model.
 * 'results', folder with results for 5, 10, and 100 iterations of mi_face algorithm.
 * 'target_model.py', the target model we are using.
 * 'model_inversion.py' our implementation of mi_face algorithm and some functions to perform the attack with the target model we are using.

### target_model.py
Our target model is a simple MLP based on the PyTorch Tutorial form lecture and https://github.com/bentrevett/pytorch-image-classification/blob/master/1_mlp.ipynb

### model_inversion.py
We perform a model inversion attack similar to Fredrikson et al. More concrete we perform an reconstruction attack, where given a label we reconstruct an image of the respective class.
This file contains our implementation of mi_face algorithm from Fredrikson et al. We perform a simple gradient descent, where we query the model with a random image, and the try to minimize 1-probability(label).

### How to run:
##### To train the target model and get the resulting atnt-mlp-model.pt:
`$ python target_model.py`

##### To perform the model inversion attack:
`$ python model_inversion.py`

optional arguments:

  `-h`, `--help`           show this help message and exit
  
  `--modelPath MODEL default=atnt-mlp-model.pt`         target model file
  
  `--iterations NUM_ITERATIONS default=10`              number of iterations mi_face should perform

  `--lossFunction LOSS_FUNCTION default=crossEntropy`  choose between crossEntropy and softmax as loss function 

  `--numberOfResults NUM_RESULTS default=one` choose between reconstructing one or all pictures




