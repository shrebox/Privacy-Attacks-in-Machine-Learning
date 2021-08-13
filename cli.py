import click
import sys

sys.path.insert(1, 'Attribute-Inference')
import attack

@click.group()
def cli():
    pass

@cli.command(help='Perform Attribute Inference Attack')
@click.option('--epochs', default=50, help='Number of Epochs')
@click.option('--fewsfew', default=50, help='rdfgrtfhg')
def attribute_inference(epochs):
    click.echo('Performing Attribute Inference')
    attack.perform_attack(False, 30)

@cli.command()
def model_inversion():
    click.echo('Performing Model Inversion')

@cli.command()
def membership_inference():
    click.echo('Performing Membership inference')

if __name__ == "__main__":
    cli()
    