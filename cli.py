import click
import sys

sys.path.insert(1, 'Attribute-Inference')
import af_attack

sys.path.insert(1, 'Membership-Inference')
import attack

sys.path.insert(1, 'ModelInversion')
import ModelInversion.model_inversion as mn

@click.group()
def cli():
    pass

@cli.group()
def attribute_inference(): 
    pass

@attribute_inference.command(help='Load trained target and attack model')
def pretrained_dummy():
    click.echo('Performing Attribute Inference with trained target and attack model')
    af_attack.perform_pretrained_dummy()

@attribute_inference.command(help='Train target and attack model')
@click.option('-t', '--target_epochs', default=30, help='Number of training epochs for the target model')
@click.option('-a', '--attack_epochs', default=50, help='Number of training epochs for the attack model')
def train_dummy(target_epochs, attack_epochs):
    click.echo('Performing Attribute Inference with training of target and attack model')
    af_attack.perform_train_dummy(target_epochs, attack_epochs)

@attribute_inference.command(help='Supply own target model and train attack model')
@click.option('-c', '--class_file', required=True, type=str, help='File that holds the target models nn.Module class')
@click.option('-s', '--state_path', required=True, type=str, help='Path of the state dictionary')
@click.option('-d', '--dimension', required=True, type=int, help='Flattend dimension of the layer used as attack modelinput ')
@click.option('-a', '--attack_epochs', default=50, type=int, help='Number of training epochs for the attack model')
def supply_target(class_file, state_path, dimension, attack_epochs):
    click.echo('Performing Attribute Inference')
    af_attack.perform_supply_target(class_file, state_path, dimension, attack_epochs)

@cli.group()
def membership_inference():
    pass

@membership_inference.command(help='Membership Inference Attack with pre-trained target and shadow models')
@click.option('--dataset', default='CIFAR10', type=str, help='Which dataset to use (CIFAR10 or MNIST)')
@click.option('--data_path', default='Membership-Inference/data', type=str, help='Path to store data')
@click.option('--model_path', default='Membership-Inference/model',type=str, help='Path to save or load model checkpoints')
def pretrained_dummy(dataset, data_path, model_path):
    click.echo('Performing Membership Inference')
    attack.create_attack(dataset, data_path, model_path, False, False, False, False, False, False)

@membership_inference.command(help='Membership Inference Attack with training enabled')
@click.option('--dataset', default='CIFAR10', type=str, help='Which dataset to use (CIFAR10 or MNIST)')
@click.option('--data_path', default='Membership-Inference/data', type=str, help='Path to store data')
@click.option('--model_path', default='Membership-Inference/model',type=str, help='Path to save or load model checkpoints')
def train_dummy(dataset, data_path, model_path):
    click.echo('Performing Membership Inference')
    attack.create_attack(dataset, data_path, model_path, True, True, False, False, False, False)

@membership_inference.command(help='Membership Inference Attack with training enabled + augmentation, topk posteriors, parameter initialization and verbose enabled')
@click.option('--dataset', default='CIFAR10', type=str, help='Which dataset to use (CIFAR10 or MNIST)')
@click.option('--data_path', default='Membership-Inference/data', type=str, help='Path to store data')
@click.option('--model_path', default='Membership-Inference/model',type=str, help='Path to save or load model checkpoints')
@click.option('--need_augm',is_flag=True, help='To use data augmentation on target and shadow training set or not')
@click.option('--need_topk',is_flag=True, help='Flag to enable using Top 3 posteriors for attack data')
@click.option('--param_init', is_flag=True, help='Flag to enable custom model params initialization')
@click.option('--verbose',is_flag=True, help='Add Verbosity')
def train_plus_dummy(dataset, data_path, model_path, need_augm, need_topk, param_init, verbose):
    click.echo('Performing Membership Inference')
    attack.create_attack(dataset, data_path, model_path, True, True, need_augm, need_topk, param_init, verbose)

@cli.group()
def model_inversion():
    pass

@model_inversion.command(help='Load trained target model and perform inversion')
@click.option('--iterations', default=30, help='Number of Iterations in attack')
@click.option('--loss_function', default="crossEntropy", type=click.Choice(['crossEntropy', 'softmax']),
              help='which loss function to use crossEntropy or softmax')
@click.option('--generate_specific_class', default=-1, type=int,
              help='choose class, number between 1 and 40, which you want recovered or nothing to get all recovered')
def pretrained_dummy(iterations, loss_function, generate_specific_class):
    click.echo('Performing model inversion with trained target model')
    mn.perform_pretrained_dummy(iterations, loss_function, generate_specific_class)


@model_inversion.command(help='Train target model and perform model inversion')
@click.option('--iterations', default=30, help='Number of Iterations in attack')
@click.option('--epochs', default=30, help='Number of epochs for the target model')
@click.option('--loss_function', default="crossEntropy", type=click.Choice(['crossEntropy', 'softmax']),
              help='which loss function to use crossEntropy or softmax')
@click.option('--generate_specific_class', default=-1, type=int,
              help='choose class, number between 1 and 40, which you want recovered or nothing to get all recovered')
def train_dummy(iterations, epochs, loss_function, generate_specific_class):
    click.echo('Performing model inversion with training of target model')
    mn.perform_train_dummy(iterations, epochs, loss_function, generate_specific_class)


# Todo: class? and all?
@model_inversion.command(help='Use trained external target model and perform model inversion')
@click.option('--class_file', required=True, type=str, help='File that holds the target models nn.Module class')
@click.option('--target_model_path', required=True, type=str, help='target model file')
@click.option('--iterations', default=30, type=int, help='Number of Iterations in the attack')
@click.option('--loss_function', default="crossEntropy", type=click.Choice(['crossEntropy', 'softmax']),
              help='which loss function to use crossEntropy or softmax')
@click.option('--generate_specific_class', default=-1, type=int,
              help='choose class, number between 1 and 40, which you want recovered or nothing to get all recovered')
def supply_target(class_file, target_model_path, iterations, loss_function, generate_specific_class):
    click.echo('performing Model Inversion')
    mn.perform_supply_target(class_file, target_model_path, iterations, loss_function, generate_specific_class)

if __name__ == "__main__":
    cli()
    