# Add submodule paths
import sys
sys.path += ['.', './normalizing_flows', './baselines', './climdex']
import click
import mlflow
import os
import logging
from experiments import glow_jflvm, bcsd, bmd

mlflow.set_tracking_uri("http://localhost:5000")
os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = "./gcs.secret.json"

@click.group()
def main():
    pass

# setup logging
logging.basicConfig(filename='run.log',level=logging.INFO)

# add subcommands
main.add_command(glow_jflvm)
main.add_command(bcsd)
main.add_command(bmd)

main()