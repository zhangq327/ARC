from argparse import Namespace, ArgumentParser

import os
import torch
from torch import nn
from torch.nn import functional as F
from torchvision import datasets
import torchvision.transforms as transforms
from utils import datautils
import models
from utils import utils
import numpy as np
import PIL
import torch.distributed as dist

class BaseSSL(nn.Module):
    """
    Inspired by the PYTORCH LIGHTNING https://pytorch-lightning.readthedocs.io/en/latest/
    Similar but lighter and customized version.
    """
    DATA_ROOT = os.environ.get('DATA_ROOT', os.path.dirname(os.path.abspath(__file__)) + '/data')
    DATA_ROOT = '.'

    def __init__(self, hparams):
        super().__init__()
        self.hparams = hparams

    def get_ckpt(self):
        return {
            'state_dict': self.state_dict(),
            'hparams': self.hparams,
        }

    @classmethod
    def load(cls, ckpt, device=None):
        res = cls(hparams, device=device)
        res.load_state_dict(ckpt['state_dict'])
        return res


    def forward(self, x):
        pass

    def transforms(self):
        pass

    def samplers(self):
        return None, None

    def prepare_data(self):
        train_transform, test_transform = self.transforms()
        if self.hparams.data == 'cifar':
            self.trainset = datasets.CIFAR10(root=self.DATA_ROOT, train=True, download=True, transform=train_transform)
            self.testset = datasets.CIFAR10(root=self.DATA_ROOT, train=False, download=True, transform=test_transform)
        else:
            raise NotImplementedError

    def dataloaders(self, iters=None):
        train_batch_sampler, test_batch_sampler = self.samplers()
        iters=None
        if iters is not None:
            train_batch_sampler = datautils.ContinousSampler(
                train_batch_sampler,
                iters
            )
        train_loader = torch.utils.data.DataLoader(
            self.trainset,
            num_workers=self.hparams.workers,
            pin_memory=True,
            batch_sampler=train_batch_sampler,
        )
        test_loader = torch.utils.data.DataLoader(
            self.testset,
            num_workers=self.hparams.workers,
            pin_memory=True,
            batch_sampler=test_batch_sampler,
        )
        
        
        return train_loader, test_loader




class SimCLR(BaseSSL):

    def __init__(self, hparams, device=None):
        super().__init__(hparams)
        model = models.encoder.EncodeProject(hparams)
        self.reset_parameters()
        if device is not None:
            model = model.to(device)
        device_ids=[1]
        self.model = nn.DataParallel(model,device_ids)


    def reset_parameters(self):
        def conv2d_weight_truncated_normal_init(p):
            fan_in = p.shape[1]
            stddev = np.sqrt(1. / fan_in) / .87962566103423978
            r = scipy.stats.truncnorm.rvs(-2, 2, loc=0, scale=1., size=p.shape)
            r = stddev * r
            with torch.no_grad():
                p.copy_(torch.FloatTensor(r))

        def linear_normal_init(p):
            with torch.no_grad():
                p.normal_(std=0.01)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                conv2d_weight_truncated_normal_init(m.weight)
            elif isinstance(m, nn.Linear):
                linear_normal_init(m.weight)


    def samplers(self):
        trainsampler = torch.utils.data.sampler.RandomSampler(self.trainset)
        testsampler = torch.utils.data.sampler.RandomSampler(self.testset)

        batch_sampler = datautils.MultiplyBatchSampler
        batch_sampler.MULTILPLIER = self.hparams.multiplier
        self.trainsampler = trainsampler
        self.batch_trainsampler = batch_sampler(trainsampler, self.hparams.batch_size, drop_last=True)

        return (
            self.batch_trainsampler,
            batch_sampler(testsampler, self.hparams.batch_size, drop_last=True)
        )

    def transforms(self):
        if self.hparams.data == 'cifar':
            train_transform = transforms.Compose([
                transforms.RandomResizedCrop(
                    32,
                    #scale=(self.hparams.scale_lower, 1),
	            scale=(0.08,1),
                    interpolation=PIL.Image.BICUBIC,
                ),
                #transforms.RandomHorizontalFlip(),
                #datautils.get_color_distortion(s=self.hparams.color_dist_s),
                transforms.ToTensor(),
                datautils.Clip(),
            ])
            test_transform = train_transform
        return train_transform, test_transform

    def load_state_dict(self, state):
        k = next(iter(state.keys()))
        if k.startswith('model.module'):
            super().load_state_dict(state)
        else:
            self.model.module.load_state_dict(state)


