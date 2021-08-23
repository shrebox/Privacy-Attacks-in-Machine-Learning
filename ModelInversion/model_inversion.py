import random
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from target_model import TargetModel, train_target_model

import argparse

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Using device: %s' % device)
# define dimensions
INPUT_DIM = 112 * 92
OUTPUT_DIM = 40


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


def perform_pretrained_dummy(iterations, loss_function, generate_specific_class):
    data_path = 'ModelInversion/atnt-mlp-model.pth'

    model = TargetModel(INPUT_DIM, OUTPUT_DIM)
    model.load_state_dict(torch.load(data_path))

    if generate_specific_class == -1:
        print('\nStart model inversion for all classes\n')
        perform_attack_and_print_all_results(model, iterations, loss_function)
    else:
        print('\nstart model inversion for class ' + str(generate_specific_class) + '\n')
        perform_attack_and_print_one_result(model, iterations, loss_function, generate_specific_class)


def perform_train_dummy(iterations, epochs, loss_function, generate_specific_class):
    data_path = 'ModelInversion/atnt-mlp-model.pth'

    if generate_specific_class > 40 | generate_specific_class < -1 | generate_specific_class == 0:
        print('please provide a class number between 1 and 40 or nothing for recover all')
        return

    print('\nTraining Target Model for ' + str(epochs) + ' epochs...')
    train_target_model(epochs)

    model = TargetModel(INPUT_DIM, OUTPUT_DIM)
    model.load_state_dict(torch.load(data_path))

    if generate_specific_class == -1:
        print('\nStart model inversion for all classes\n')
        perform_attack_and_print_all_results(model, iterations, loss_function)
    else:
        print('\nStart model inversion for class ' + str(generate_specific_class) + '\n')
        perform_attack_and_print_one_result(model, iterations, loss_function, generate_specific_class)


def perform_supply_target(class_file, target_model_path, iterations, loss_function, generate_specific_class):

    try:
        module = __import__(class_file, globals(), locals(), ['TargetModel'])
    except ImportError:
        print('Target model class could not be imported... Please check if file is inside "PETS-PROJECT/ModelInversion" and class name is "TargetModel"')
        return

    TargetModel = vars(module)['TargetModel']

    target_model = TargetModel(INPUT_DIM, OUTPUT_DIM).to('cpu')
    print('Loading Target Model...')
    target_model.load_state_dict(torch.load(target_model_path))

    if generate_specific_class > 40 | generate_specific_class < -1 | generate_specific_class == 0:
        print('please provide a class number between 1 and 40 or nothing for recover all')
        return

    if generate_specific_class == -1:
        print('\nStart model inversion for all classes\n')
        perform_attack_and_print_all_results(target_model, iterations, loss_function)
    else:
        print('\nstart model inversion for class ' + str(generate_specific_class) + '\n' )
        perform_attack_and_print_one_result(target_model, iterations, loss_function, generate_specific_class)


def perform_attack_and_print_all_results(model, iterations, loss_function):

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
            print('\nReconstructing Class ' + str(count))

            ran = random.randint(1, 2)
            path = 'ModelInversion/data_pgm/s0' + str(count) + '/' + str(
                ran) + '.pgm' if count < 10 else 'ModelInversion/data_pgm/s' + str(count) + '/' + str(ran) + '.pgm'

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
    fig.savefig('ModelInversion/results/results_' + str(iterations) + '.png', dpi=100)
    plt.show()
    print('\nReconstruction Results can be found in results folder')


def perform_attack_and_print_one_result (model,  iterations, loss_function, generate_specific_class):
    # set params
    gradient_step_size = 0.1

    # create figure
    fig, (ax1, ax2) = plt.subplots(1, 2, constrained_layout=True, sharey=True)
    # reconstruction for class 0
    reconstruction = mi_face(generate_specific_class-1, model, iterations, gradient_step_size, loss_function)
    ran = random.randint(1, 2)
    path = 'ModelInversion/data_pgm/s0' + str(generate_specific_class) + '/' + str(
        ran) + '.pgm' if generate_specific_class < 10 else 'ModelInversion/data_pgm/s' + str(generate_specific_class) + '/' + str(ran) + '.pgm'

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
    fig.savefig('ModelInversion/results/results_' + str(iterations) + '.png', dpi=100)
    plt.show()
    print('Reconstruction Results can be found in results folder')