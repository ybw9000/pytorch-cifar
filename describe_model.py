import torch
from utils import get_args


def main():
    args = get_args()
    checkpoint = torch.load(args.checkpoint)
    print("Total number of parameters: ", checkpoint['num_paras'])
    print("acc: {} after (pretrain + transfer) epochs: {}".format(
        checkpoint['acc'], checkpoint['epoch']))


if __name__ == "__main__":
    main()
