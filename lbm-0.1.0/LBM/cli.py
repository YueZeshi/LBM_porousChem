import click
import LBM3D

@click.command()
@click.option('--name', prompt='你的名字', help='要问候的人')
@click.option('--count', default=1, help='重复次数')
def main(name, count):
    for i in range(count):
        click.echo(f"Hello, {name}!")
