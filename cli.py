import click
import sys

sys.path.insert(1, 'Attribute-Inference')
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
    attack.perform_pretrained_dummy()


@attribute_inference.command(help='Train target and attack model')
@click.option('-t', '--target_epochs', default=30, help='Number of training epochs for the target model')
@click.option('-a', '--attack_epochs', default=50, help='Number of training epochs for the attack model')
def train_dummy(target_epochs, attack_epochs):
    click.echo('Performing Attribute Inference with training of target and attack model')
    attack.perform_train_dummy(target_epochs, attack_epochs)


@attribute_inference.command(help='Supply own target model and train attack model')
@click.option('-c', '--class_file', required=True, type=str, help='File that holds the target models nn.Module class')
@click.option('-s', '--state_path', required=True, type=str, help='Path of the state dictio nary')
@click.option('-d', '--dimension', required=True, type=int,
              help='Flattend dimension of the layer used as attack modelinput ')
@click.option('-a', '--attack_epochs', default=50, type=int, help='Number of training epochs for the attack model')
def supply_target(class_file, state_path, dimension, attack_epochs):
    click.echo('Performing Attribute Inference')
    attack.perform_supply_target(class_file, state_path, dimension, attack_epochs)


@cli.group()
def membership_inference():
    pass


@cli.group()
def model_inversion():
    pass


@model_inversion.command(help='load trained target model and perform inversion')
@click.option('--iterations', default=30, help='Number of Iterations in attack')
@click.option('--loss_function', default="crossEntropy", type=click.Choice(['crossEntropy', 'softmax']),
              help='which loss function to use crossEntropy or softmax')
@click.option('--generate_specific_class', default=-1, type=int,
              help='choose class, number between 1 and 40, which you want recovered or nothing to get all recovered')
def pretrained_dummy(iterations, loss_function, generate_specific_class):
    click.echo('Performing model inversion with trained target model')
    mn.perform_pretrained_dummy(iterations, loss_function, generate_specific_class)


@model_inversion.command(help='train target model')
@click.option('--iterations', default=30, help='Number of Iterations in attack')
@click.option('--epochs', default=30, help='Number of epochs for the target model')
@click.option('--loss_function', default="crossEntropy", type=click.Choice(['crossEntropy', 'softmax']),
              help='which loss function to use crossEntropy or softmax')
@click.option('--generate_specific_class', default=-1, type=int,
              help='choose class, number between 1 and 40, which you want recovered or nothing to get all recovered')
def train_dummy(iterations, epochs, loss_function, generate_specific_class):
    click.echo('Performing model inversion with training of target model')
    mn.perform_train_dummy(iterations, epochs, loss_function, generate_specific_class)


# Todo: tag? and all?
@model_inversion.command(help='use trained external target model and perform model inversion')
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
