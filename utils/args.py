import argparse


def train_parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--logs", default="./runs/0")
    parser.add_argument("--batch_size", default=64, type=int)
    parser.add_argument("--n_epochs", default=10, type=int)
    return parser.parse_args()
