import os

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torchvision
import models
import argparse
import sys
import torch.nn.functional as F
from ARC import ARC_calculate

def add_learner_params(parser):
    parser.add_argument('--problem', default='sim-clr',
        help='The problem to train',
        choices=models.REGISTERED_MODELS,
    )
    parser.add_argument('--name', default='',
        help='Name for the experiment',
    )
    parser.add_argument('--ckpt', default='',
        help='checkpoint to calculate the init ACR'
    )
    parser.add_argument('--ckpt2', default='',
        help='checkpoint to calculate the final ACR'
    )
    parser.add_argument('--gpu', default=0, type=int)
    parser.add_argument('--verbose', default=False, type=bool)
    parser.add_argument('--seed', default=-1, type=int, help='Random seed')
    parser.add_argument('-j', '--workers', default=4, type=int, help='The number of data loader workers')
    parser.add_argument('--views', default=2, type=int, help='The number of augmented views')
    parser.add_argument('--data', help='Dataset to use', default='cifar')
    parser.add_argument('--arch', default='ResNet50', help='Encoder architecture')
    parser.add_argument('--batch_size', default=256, type=int, help='The number of unique images in the batch')
    parser.add_argument('--aug', default=True, type=bool, help='Applies random augmentations if True')
    # data params
    parser.add_argument('--multiplier', default=2, type=int)
    parser.add_argument('--color_dist_s', default=1., type=float, help='Color distortion strength')
    parser.add_argument('--scale_lower', default=0.08, type=float, help='The minimum scale factor for RandomResizedCrop')
    # ddp
    parser.add_argument('--sync_bn', default=True, type=bool,
        help='Syncronises BatchNorm layers between all processes if True'
    )


def main():
    parser = argparse.ArgumentParser()
    add_learner_params(parser)
    args = parser.parse_args()
    if '--help' in sys.argv or '-h' in sys.argv:
        sys.argv.pop(sys.argv.index('--help' if '--help' in sys.argv else '-h'))
        is_help = True

    torch.cuda.set_device(args.gpu)
    device = torch.device('cuda')
    
    # create model
    init_model = models.REGISTERED_MODELS[args.problem](args, device=device)
    
    if args.ckpt != '':
        ckpt = torch.load(args.ckpt, map_location=device)
        init_model.load_state_dict(ckpt['state_dict'])

    cudnn.benchmark = True
    final_model = models.REGISTERED_MODELS[args.problem](args, device=device)
    if args.ckpt2 != '':
        ckpt = torch.load(args.ckpt2, map_location=device)
        final_model.load_state_dict(ckpt['state_dict'])
    ARC_calculate(args, init_model, final_model, device)


if __name__ == '__main__':
    main()
