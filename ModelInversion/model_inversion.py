import random
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from target_model import MLP, train_target_model

import argparse

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Using device: %s' % device)


# class MLP(nn.Module):

#     def __init__(self, input_dim, output_dim):
#         super().__init__()

#         self.input_fc = nn.Linear(input_dim, 3000)
#         self.output_fc = nn.Linear(3000, output_dim)

#     def forward(self, x):
#         batch_size = x.shape[0]
#         x = x.view(batch_size, -1)
#         h = torch.sigmoid(self.input_fc(x))
#         output = self.output_fc(h)

#         return output, h


def mi_face(label_index, model, num_iterations, gradient_step, loss_function):
    model.to(device)
    model.eval()

    # initialize two 112 * 92 tensors with zeros
    tensor = torch.zeros(112, 92).unsqueeze(0).to(device)
    image = torch.zeros(112, 92).unsqueeze(0).to(device)

    # initialize with infinity
    min_loss = float("inf")

    for i in range(num_iterations):
        tensor.requires_grad = True

        # get the prediction probs
        pred, _ = model(tensor)

        # calculate the loss and gardient for the class we want to reconstruct
        if loss_function == "crossEntropy":
            # use this
            crit = nn.CrossEntropyLoss()
            loss = crit(pred, torch.tensor([label_index]).to(device))
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
            image = tensor.detach().clone().to('cpu')

    return image


#
# def get_cmd_arguments():
#     parser = argparse.ArgumentParser()
#     parser.add_argument('--modelPath', default='atnt-mlp-model.pt', type=str, help='')
#     parser.add_argument('--iterations', default='10', type=int, help='Number of Iterations')
#     parser.add_argument('--lossFunction', default="crossEntropy", type=str, choices=['crossEntropy', 'softmax'], help='which loss function to use crossEntropy or softmax')
#     parser.add_argument('--numberOfResults', default='one', type=str, choices=['one', 'all'], help='chose how many results between one and all')
#     return parser.parse_args()
#
# if __name__ == '__main__':
#     # get command line arguments from the user
#     args = get_cmd_arguments()
#     print(args)
#
#     # load the model and set it to eval mode
#     #model = torch.load(args.modelPath, map_location='cpu')
#     model = torch.load(args.modelPath)
#
#     # set params
#     gradient_step_size = 0.1
#
#     # Print only one picture
#     if args.numberOfResults == 'one':
#         # create figure
#         fig, (ax1, ax2) = plt.subplots(1, 2, constrained_layout=True, sharey=True)
#         #reconstruction for class 0
#         reconstruction = mi_face(0, model, args.iterations, gradient_step_size)
#         ran = random.randint(1, 2)
#         path = 'data_pgm/s0' + str(1) + '/' + str(
#                      ran) + '.pgm' if 0 < 10 else 'data_pgm/s' + str(0) + '/' + str(ran) + '.pgm'
#
#         with open(path, 'rb') as f:
#             original = plt.imread(f)
#         # add both images to the plot
#         ax1.imshow(original, cmap='gray')
#         ax1.set_title('Sample train set image')
#         ax1.axis('off')
#         ax2.imshow(reconstruction.squeeze().detach().numpy(), cmap='gray')
#         ax2.set_title('Reconstructed image')
#         ax2.axis('off')
#
#         # plot reconstructed image
#         fig.suptitle('Images reconstructed with\n ' + str(
#             args.iterations) + ' iterations of mi_face. ', fontsize=15)
#         fig.savefig('results/results_' + str(args.iterations) + '.png', dpi=100)
#         plt.show()
#         print('Reconstruction Results can be found in results folder')
#
#
#     else:
#         # print all pictures
#         # create figure
#         fig, axs = plt.subplots(8, 10)
#         fig.set_size_inches(20, 24)
#         random.seed(7)
#         count = 0
#         for i in range(0, 8, 2):
#             for j in range(10):
#                 # get random validation set image from respective class
#                 count += 1
#                 print('Reconstructing Class ' + str(count))
#
#                 ran = random.randint(1, 2)
#                 path = 'data_pgm/s0' + str(count) + '/' + str(
#                     ran) + '.pgm' if count < 10 else 'data_pgm/s' + str(count) + '/' + str(ran) + '.pgm'
#
#                 with open(path, 'rb') as f:
#                     original = plt.imread(f)
#
#                 # reconstruct respective class
#                 reconstruction = mi_face(count - 1, model, args.iterations, gradient_step_size)
#
#
#                 # add both images to the plot
#                 axs[i, j].imshow(original, cmap='gray')
#                 axs[i + 1, j].imshow(reconstruction.squeeze().detach().numpy(), cmap='gray')
#                 axs[i, j].axis('off')
#                 axs[i + 1, j].axis('off')
#
#         # plot reconstructed image
#         fig.suptitle('Images reconstructed with ' + str(
#             args.iterations) + ' iterations of mi_face. Find the reconstruction below each row with train set samples.', fontsize=20)
#         fig.savefig('results/results_' + str(args.iterations) + '.png', dpi=100)
#         plt.show()
#         print('Reconstruction Results can be found in results folder')


def perform_pretrained_dummy():
    data_path = 'ModelInversion/atnt-mlp-model.pt'
    epochs = 30
    loss_function = 'crossEntropy'
    perform_attack_and_print_all_results(data_path, epochs, loss_function)


# Todo: input parameter for result
def perform_train_dummy(iterations, epochs, loss_function, number_of_results):
    data_path = 'ModelInversion/atnt-mlp-model.pt'

    if number_of_results > 40 | number_of_results < -1 | number_of_results == 0:
        print('please provide a tag number between 1 and 40 or nothing for recover all')
        return

    print('Training Target Model for ' + str(epochs) + ' epochs...')
    train_target_model(epochs)

    if number_of_results == -1:
        print('start model inversion for all tags')
        perform_attack_and_print_all_results(data_path, iterations, loss_function)
    else:
        print('start model inversion for ' + str(number_of_results) + 'tag')
        perform_attack_and_print_one_result(data_path, iterations, loss_function, number_of_results)


def perform_supply_target(class_file, iterations, loss_function, number_of_results):
    data_path = class_file

    if number_of_results > 40 | number_of_results < -1 | number_of_results == 0:
        print('please provide a tag number between 1 and 40 or nothing for recover all')
        return

    if number_of_results == -1:
        print('start model inversion for all tags')
        perform_attack_and_print_all_results(data_path, iterations, loss_function)
    else:
        print('start model inversion for ' + str(number_of_results) + 'tag')
        perform_attack_and_print_one_result(data_path, iterations, loss_function, number_of_results)


def perform_attack_and_print_all_results(data_path, iterations, loss_function):
    model = torch.load(data_path)
    gradient_step_size = 0.1

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
            path = 'data_pgm/s0' + str(count) + '/' + str(
                ran) + '.pgm' if count < 10 else 'data_pgm/s' + str(count) + '/' + str(ran) + '.pgm'

            with open(path, 'rb') as f:
                original = plt.imread(f)

            # reconstruct respective class
            reconstruction = mi_face(count - 1, model, iterations, gradient_step_size, loss_function)

            # add both images to the plot
            axs[i, j].imshow(original, cmap='gray')
            axs[i + 1, j].imshow(reconstruction.squeeze().detach().numpy(), cmap='gray')
            axs[i, j].axis('off')
            axs[i + 1, j].axis('off')

    # plot reconstructed image
    fig.suptitle('Images reconstructed with ' + str(
        iterations) + ' iterations of mi_face. Find the reconstruction below each row with train set samples.',
                 fontsize=20)
    fig.savefig('results/results_' + str(iterations) + '.png', dpi=100)
    plt.show()
    print('Reconstruction Results can be found in results folder')


def perform_attack_and_print_one_result(target_model, iterations, loss_function, number_of_results):
    model = torch.load(target_model)
    # set params
    gradient_step_size = 0.1

    # create figure
    fig, (ax1, ax2) = plt.subplots(1, 2, constrained_layout=True, sharey=True)
    # reconstruction for class 0
    reconstruction = mi_face(number_of_results-1, model, iterations, gradient_step_size, loss_function)
    ran = random.randint(1, 2)
    path = 'data_pgm/s0' + str(1) + '/' + str(
        ran) + '.pgm' if 0 < 10 else 'data_pgm/s' + str(0) + '/' + str(ran) + '.pgm'

    with open(path, 'rb') as f:
        original = plt.imread(f)
    # add both images to the plot
    ax1.imshow(original, cmap='gray')
    ax1.set_title('Sample train set image')
    ax1.axis('off')
    ax2.imshow(reconstruction.squeeze().detach().numpy(), cmap='gray')
    ax2.set_title('Reconstructed image')
    ax2.axis('off')

    # plot reconstructed image
    fig.suptitle('Images reconstructed with\n ' + str(
        iterations) + ' iterations of mi_face. ', fontsize=15)
    fig.savefig('results/results_' + str(iterations) + '.png', dpi=100)
    plt.show()
    print('Reconstruction Results can be found in results folder')
