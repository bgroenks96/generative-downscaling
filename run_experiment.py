# Add submodule paths
import sys
sys.path += ['.', './normalizing_flows', './baselines', './climdex']
import click
import logging
from experiments import glow_jflvm_cv, bcsd, bmd

@click.group()
def main():
    pass

# setup logging
logging.basicConfig(filename='run.log',level=logging.INFO)

# add subcommands
main.add_command(glow_jflvm_cv)
main.add_command(bcsd)
main.add_command(bmd)

main()