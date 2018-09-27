import torch
from utils import get_args


def main():
    args = get_args()
    checkpoint = torch.load(args.model_path)
    print("acc: {} after (pretrain + transfer) epochs: {}".format(
        checkpoint['acc'], checkpoint['epoch']))


if __name__ == "__main__":
    main()
