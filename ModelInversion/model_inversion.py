import random
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

import argparse


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

    # initialize two 112 * 92 tensors with zeros
    tensor = torch.zeros(112, 92).unsqueeze(0)
    image = torch.zeros(112, 92).unsqueeze(0)


    # initialize with infinity
    min_loss = float("inf")

    for i in range(num_iterations):
        tensor.requires_grad = True
        # get the prediction probs
        pred, _ = model(tensor)

        # calculate the loss and gardient for the class we want to reconstruct
        if args.lossFunction == "crossEntropy":
            # use this
            crit = nn.CrossEntropyLoss()
            loss = crit(pred, torch.tensor([label_index]))
        else:
            # or this
            soft_pred = nn.functional.softmax(pred, 1)
            loss = soft_pred.squeeze()[label_index]

        print('Loss: ' + str(loss.item()))
        loss.backward()

        with torch.no_grad():
            # apply gradient descent
            tensor = (tensor - gradient_step * tensor.grad)

        # set image = tensor only if the new loss is the min from all iterations
        if loss < min_loss:
            min_loss = loss
            image = tensor.detach().clone()

        return image


def get_cmd_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--modelPath', default='atnt-mlp-model.pt', type=str, help='')
    parser.add_argument('--iterations', default='10', type=int, help='Number of Iterations')
    parser.add_argument('--lossFunction', default="crossEntropy", type=str, choices=['crossEntropy', 'softmax'], help='which loss function to use crossEntropy or softmax')
    parser.add_argument('--numberOfResults', default='one', type=str, choices=['one', 'all'], help='chose how many results between one and all')
    return parser.parse_args()

if __name__ == '__main__':
    # get command line arguments from the user
    args = get_cmd_arguments()
    print(args)

    # load the model and set it to eval mode
    model = torch.load(args.modelPath)
    model.eval()

    # set params
    gradient_step_size = 0.1

    # Print only one picture
    if args.numberOfResults == 'one':
        # create figure
        fig, (ax1, ax2) = plt.subplots(1, 2, constrained_layout=True, sharey=True)
        reconstruction = mi_face(0, args.iterations, gradient_step_size)
        ran = random.randint(1, 2)
        path = 'data_pgm/faces/s0' + str(1) + '/' + str(
                     ran) + '.pgm' if 0 < 10 else 'data_pgm/faces/s' + str(0) + '/' + str(ran) + '.pgm'

        with open(path, 'rb') as f:
            original = plt.imread(f)
        # add both images to the plot
        ax1.imshow(original, cmap='gray')
        ax1.set_title('original')
        ax1.axis('off')
        ax2.imshow(reconstruction.squeeze().detach().numpy(), cmap='gray')
        ax2.set_title('reconstructed')
        ax2.axis('off')

        # plot reconstructed image
        fig.suptitle('Images reconstructed with\n ' + str(
            args.iterations) + ' iterations of mi_face. ', fontsize=15)
        fig.savefig('results/results_' + str(args.iterations) + '.png', dpi=100)
        plt.show()
    else:
        # print all pictures
        # create figure
        fig, axs = plt.subplots(8, 10)
        fig.set_size_inches(20, 24)
        random.seed(7)
        count = 0
        for i in range(0, 8, 2):
            for j in range(10):
                # get random validation set image from respective class
                count += 1
                print('Reconstructing Class ' + str(count))

                ran = random.randint(1, 2)
                path = 'data_pgm/faces/s0' + str(count) + '/' + str(
                    ran) + '.pgm' if count < 10 else 'data_pgm/faces/s' + str(count) + '/' + str(ran) + '.pgm'

                with open(path, 'rb') as f:
                    original = plt.imread(f)

                # reconstruct respective class
                reconstruction = mi_face(count - 1, args.iterations, gradient_step_size)


                # add both images to the plot
                axs[i, j].imshow(original, cmap='gray')
                axs[i + 1, j].imshow(reconstruction.squeeze().detach().numpy(), cmap='gray')
                axs[i, j].axis('off')
                axs[i + 1, j].axis('off')

        # plot reconstructed image
        fig.suptitle('Images reconstructed with ' + str(
            args.iterations) + ' iterations of mi_face. Find the reconstruction below the respective original.', fontsize=20)
        fig.savefig('results/results_' + str(args.iterations) + '.png', dpi=100)
        plt.show()

