import click
import sys

sys.path.insert(1, 'Attribute-Inference')
import attack

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
@click.option('--target_epochs', default=30, help='Number of training epochs for the target model')
@click.option('--attack_epochs', default=50, help='Number of training epochs for the attack model')
def train_dummy(target_epochs, attack_epochs):
    click.echo('Performing Attribute Inference with training of target and attack model')
    attack.perform_train_dummy(target_epochs, attack_epochs)

@attribute_inference.command(help='Supply own target model and train attack model')
@click.option('--class-path', required=True, type=str, help='Path of the models nn.Module class')
@click.option('--state-path', required=True, type=str, help='Path of the state dictionary')
@click.option('--attack_epochs', required=True, type=str, help='Number of training epochs for the attack model')
def supply_target(epochs):
    click.echo('Performing Attribute Inference')

@cli.group()
def membership_inference():
    pass

@cli.group()
def model_inversion():
    pass

if __name__ == "__main__":
    cli()
    