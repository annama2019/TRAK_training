from argparse import ArgumentParser
from typing import List
import time
import datetime
import random
import numpy as np
from tqdm import tqdm
import math
import timm
import json

import torch as ch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import torchvision
import torchattacks
from torch.cuda.amp import GradScaler, autocast
from torch.nn import CrossEntropyLoss, Conv2d, BatchNorm2d
from torch.optim import SGD, lr_scheduler
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms

from fastargs import get_current_config, Param, Section
from fastargs.decorators import param
from fastargs.validation import And, OneOf

from pathlib import Path

Section('training', 'Hyperparameters').params(
    lr=Param(float, 'The learning rate to use', required=True),
    epochs=Param(int, 'Number of epochs to run for', required=True),
    lr_peak_epoch=Param(int, 'Peak epoch for cyclic lr',default=5),
    batch_size=Param(int, 'Batch size', default=512),
    momentum=Param(float, 'Momentum for SGD', default=0.9),
    weight_decay=Param(float, 'l2 weight decay', default=5e-4),
    label_smoothing=Param(float, 'Value of label smoothing', default=0.1),
    num_workers=Param(int, 'The number of workers', default=8),
    lr_tta=Param(bool, 'Test time augmentation by averaging with horizontally flipped version', default=True),
    weighting=Param(bool, 'use importance weighting or not', default=True),
    arch = Param(str, 'model architecture', required=True),
)

@param('training.batch_size')
@param('training.num_workers')
def make_dataloaders(batch_size=None, num_workers=None, device = 'cuda:0'):
    transform = transforms.Compose([transforms.Resize(256), transforms.CenterCrop(224), transforms.ToTensor(),
                                    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])
    ds_train = datasets.ImageFolder(root='/mnt/cfs/datasets/pytorch_imagenet/train', transform=transform)
    ds_val = datasets.ImageFolder(root='/mnt/cfs/datasets/pytorch_imagenet/val', transform=transform)

    data_loader_params = {
        'batch_size': batch_size,  # Batch size for data loading
        'num_workers': num_workers,  # Number of subprocesses to use for data loading
        'persistent_workers': True,  # If True, the data loader will not shutdown the worker processes after a dataset has been consumed once. This allows to maintain the worker dataset instances alive.
        'pin_memory': True,  # If True, the data loader will copy Tensors into CUDA pinned memory before returning them. Useful when using GPU.
        'pin_memory_device': device,  # Specifies the device where the data should be loaded. Commonly set to use the GPU.
    }


    start_time = time.time()
    loader_train = DataLoader(ds_train, **data_loader_params, shuffle=True)
    loader_val = DataLoader(ds_val, **data_loader_params)
    loaders = {'train':loader_train, 'test':loader_val}
    return loaders, start_time

@param('training.arch')
def construct_model(arch=None):
    model = timm.create_model(arch,pretrained = True).cuda()
    return model

@param('training.lr')
@param('training.epochs')
@param('training.momentum')
@param('training.weight_decay')
@param('training.label_smoothing')
@param('training.lr_peak_epoch')
@param('training.arch')
def train(model, loaders, lr=None, epochs=None, label_smoothing=None,
          momentum=None, weight_decay=None, lr_peak_epoch=None, arch=None, 
          device='cuda:0'):
    opt = SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
    iters_per_epoch = len(loaders['train'])
    best_loss = float('inf')
    lr_schedule = np.interp(np.arange((epochs+1) * iters_per_epoch),
                            [0, lr_peak_epoch * iters_per_epoch, epochs * iters_per_epoch],
                            [0, 1, 0])
    scheduler = lr_scheduler.LambdaLR(opt, lr_schedule.__getitem__)
    scaler = GradScaler()
    loss_fn = CrossEntropyLoss(reduction="none",label_smoothing=label_smoothing)
    for e in range(epochs):
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        checkpoint_dir = Path(f"models/{timestamp}")
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        checkpoint_path = checkpoint_dir/f"{arch}.pth"

        model.train()
        for ims, labs in tqdm(loaders['train']):
            train_loss = 0.
            opt.zero_grad(set_to_none=True)
            
            with autocast():
                out = model(ims.to(device))
                loss = ch.mean(loss_fn(out, labs.to(device)))
                train_loss += loss.item()

            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            scheduler.step()
        print(f'loss at epoch {e}: {train_loss}')
        model.eval()
        test_loss = 0.
        with ch.no_grad():
            for ims, labs in tqdm(loaders['test']):
                with autocast():
                    out = model(ims.to(device))
                    loss = ch.mean(loss_fn(out, labs.to(device)))
                    test_loss += loss.item()
                    
        if test_loss < best_loss:
            best_loss = test_loss
            ch.save(model.state_dict(), checkpoint_path)
            accuracy = evaluate(model,loaders)
            
            training_metadata = {
                'epoch': e,
                'train_loss': train_loss,
                'test_loss': test_loss, 
                'train_accuracy': accuracy['train'],
                'test_accuracy': accuracy['test'],
                'learning_rate': lr,
                'momentum': momentum,
                'weight_decay': weight_decay,
                'lr_peak_epoch': lr_peak_epoch,
                'label_smoothing': label_smoothing,
                'model_architecture': arch,
            }
            with open(Path(checkpoint_path.parent/'training_metadata.json'), 'w') as f:
                json.dump(training_metadata, f)
            
@param('training.lr_tta')
def evaluate(model, loaders, lr_tta=False,device='cuda:0'):
    metric = {}
    model.eval()
    with ch.no_grad():
        for name in ['train', 'test']:
            total_correct, total_num = 0., 0.
            for ims, labs in tqdm(loaders[name]):                
                with autocast():
                    out = model(ims.to(device))
                    if lr_tta:
                        out += model(ims.flip(-1).to(device))
                    total_correct += out.argmax(1).eq(labs.to(device)).sum().cpu().item()
                    total_num += ims.shape[0]
                   
            print(f'{name} total accuracy: {total_correct / total_num * 100:.1f}%')
            metric[name] = total_correct / total_num
    return metric
            
if __name__ == "__main__":
    config = get_current_config()
    parser = ArgumentParser(description='Imagenet fast finetune')
    config.augment_argparse(parser)
    # Also loads from args.config_path if provided
    config.collect_argparse_args(parser)
    config.validate(mode='stderr')
    config.summary()

    loaders, start_time = make_dataloaders()
    model = construct_model()
    train(model, loaders)
    print(f'Total time: {time.time() - start_time:.5f}')
    evaluate(model, loaders)