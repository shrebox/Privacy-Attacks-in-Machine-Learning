import click
import sys

sys.path.insert(1, 'Attribute-Inference')
import attack
sys.path.insert(1, 'ModelInversion')

import model_inversion 


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
@click.option('-c', '--class_file', required=True, type=str, help='Path of the models nn.Module class')
@click.option('-s', '--state_path', required=True, type=str, help='Path of the state dictionary')
@click.option('-d', '--dimension', required=True, type=int, help='Flattend dimension of the layer used as attack modelinput ')
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

if __name__ == "__main__":
    cli()
    