"""CLI entry point – also allows `python -m nncm`."""

import argparse
import sys


def main():
    parser = argparse.ArgumentParser(
        prog="nncm",
        description="Neural Network Consequence Modelling (NNCM)",
    )
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("generate", help="Generate LHS training data (pressure vessels & leaks)")
    sub.add_parser("train", help="Train the consequence neural network")
    sub.add_parser("gui", help="Launch the nncm prediction GUI")

    args = parser.parse_args()

    if args.command == "generate":
        from nncm.data_generator import main as _main
        _main()
    elif args.command == "train":
        from nncm.neural_network import main as _main
        _main()
    elif args.command == "gui":
        from nncm.gui_predictor import main as _main
        _main()


if __name__ == "__main__":
    main()
