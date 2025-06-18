import wandb
import argparse
import regression

def parse_args():
    parser = argparse.ArgumentParser(prog='regression.py')
    parser.add_argument('-t', '--datatype', type=str,
                        choices=['SHO', 'SineGaussian', 'LIGO'],
                        help='Data type or dataset.')
    parser.add_argument('-l', '--loss', type=str,
                        choice=['NLLGaussian', 'Quantile'],
                        help='Type of loss function to use.')
    parser.add_argument('-d', '--device', type=str,
                        default='cpu',
                        help='Device to run on.')
    parser.add_argument('-e', '--epochs', type=int,
                        default=220,
                        help='Number of epochs.')