import io
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image
import matplotlib.pyplot as plt

class MLP(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()

        self.input_fc = nn.Linear(input_dim, 3000)
        self.output_fc = nn.Linear(3000, output_dim)

    def forward(self, x):

        batch_size = x.shape[0]
        x = x.view(batch_size, -1)
        h = torch.sigmoid(self.input_fc(x))
        output = self.output_fc(h)

        return output, h

def mi_face(label_index, num_iterations, gradient_step):

    # set the loss function
    criterion = torch.nn.CrossEntropyLoss()

    # initialize two 112 * 92 tensors with zeros
    tensor = torch.zeros(112, 92).unsqueeze(0)
    image = tensor

    # initialize with infinity
    min_loss = float("inf")

    for i in range(num_iterations):
        tensor.requires_grad = True
        # get the prediction probs
        pred, _ = model(tensor)

        # calculate the loss for the class we want to reconstruct
        loss = criterion(pred, torch.tensor([label_index]))
        loss.backward()

        with torch.no_grad():
            # apply gradient decent formula
            # tensor = torch.clamp(tensor - gradient_step * tensor.grad, 0, 255)
            tensor = (tensor - gradient_step * tensor.grad)
            
            # set image = tensor only if the new loss is the min from all iterations
            if loss < min_loss:
                min_loss = loss
                image = tensor
        print(min_loss)

    return image


if __name__ == '__main__':

    #load the mdoel and set it to eval mode
    model = torch.load('atnt-mlp-model.pt')
    model.eval()

    # random generated dummy names
    class_index = json.load(open('class_index.json'))

    # call gradient decent algorithm
    image = mi_face(4, 30, 0.1)

    #plot reconstructed image
    plt.imshow(image.permute(1, 2, 0).detach().numpy())
    plt.show()

