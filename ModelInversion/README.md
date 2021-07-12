## Model Inverison

### How to run:

$ python model_inversions.py --help --> To see all the possible argument options

usage: Membership Inference Attack [-h]
                                   [--modelPath MODELPATH]
                                   [--iterations ITERATIONS]

optional arguments:
  -h, --help            show this help message and exit
  --modelPath           path of the targert model, default: Our Target Model
  --iterations          number of iterations mi_face should perform


### Data
We are using a dataset called 'Atnt Dataset of Faces'. It has 1 channel images of 40 different persons, 10 different images for each person. The images have a dimension of 112 * 92 pixels.

### model_inversion.py
We perform a model inversion attack. More concrete we perform an reconstruction attack, where given a label we reconstruct an image of the respective class.
To do so we perform a simple gradient descent. We query the model with a random image, and the try to minimze 1-probability(label). This process is implemented in the function ```def mi_face```.
#### Functions
```def mi_face```
- Input: 
  - ```label_index```: the (index of) the label of the clas we want to reconstruct
  - ```num_iterations```: number of interations we try to minimize the 1-probability(label)
  - ```gradient_step```: the gradient step, metric of the gradient decent algrothim (x - gradient_step * x.gradient)
- Output: reconstructed image in the form of a 112 * 92 tensor, just like the models input dim.

```main```
- see code comments for details

### target_model.py

#### Model
Our target model is a simple MLP based on the PyTorch Tutorial form lecture and https://github.com/bentrevett/pytorch-image-classification/blob/master/1_mlp.ipynb

```class MLP``` (Simple MLP)
- Input layer dim: 112 * 92
- Hidden layer dim: 3000 (sigmoid)
- Output layer dim: 40

#### Functions
```def calculate_accuracy```
- function to calculate the accuracy

```def train```
- function to train the model

```def evaluate```
- function to evaluate the model

```main```
- see code comments for details



