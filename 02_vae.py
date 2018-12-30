# coding: utf-8

# DISTRIBUTION STATEMENT A. Approved for public release: distribution unlimited.
#
# This material is based upon work supported by the Assistant Secretary of Defense for Research and
# Engineering under Air Force Contract No. FA8721-05-C-0002 and/or FA8702-15-D-0001. Any opinions,
# findings, conclusions or recommendations expressed in this material are those of the author(s) and
# do not necessarily reflect the views of the Assistant Secretary of Defense for Research and
# Engineering.
#
# © 2018 Massachusetts Institute of Technology.
#
# MIT Proprietary, Subject to FAR52.227-11 Patent Rights - Ownership by the contractor (May 2014)
#
# The software/firmware is provided to you on an As-Is basis
#
# Delivered to the U.S. Government with Unlimited Rights, as defined in DFARS Part 252.227-7013 or
# 7014 (Feb 2014). Notwithstanding any copyright notice, U.S. Government rights in this work are
# defined by DFARS 252.227-7013 or DFARS 252.227-7014 as detailed above. Use of this work other than
# as specifically authorized by the U.S. Government may violate any copyrights that exist in this
# work.

''' Training code for the VAE '''

import os
import datetime
import argparse
import pandas as pd

import torch
from torchvision import transforms
from models.models import VAE, vae_loss
from dataloader.data_class import ObservationDataset
from dataloader.transforms import ToFloat, RandomFlip, ObservationToTensor
from utils.utils import crop_and_resize, save_checkpoint, check_mode

parser = argparse.ArgumentParser()
parser.add_argument('mode', help="The mapping you want the VAE to learn. Options\
                                  are standard, texture, segment.", type=check_mode)
parser.add_argument('processes', help="The number of processes to use in VAE training")
args = parser.parse_args()


def train(epoch, model, optimizer, train_loader, save_path, device=None):
    ''' Train the VAE model.
    Parameters
    ----------
    epoch : int
        The number of epochs for which to train the model.
    model : torch.nn.Module
        The model to train.
    optimizer : torch.optim.Optimizer
        The optimizer to use in training the model.
    train_loader : torch.utils.data.DataLoader
        The data loader to use for training.
    save_path : Union[str, pathlib.Path]
        The path to which to save the model.
    device : Optional[str]
        The device to  use for training, or `None` to use CUDA where available.
    '''
    device = device if device else 'cuda' if torch.cuda.is_available() else 'cpu'
    model = model.to(device)
    model.train()
    train_loss = 0
    batch_count = len(train_loader)
    save_interval = 1000
    print_interval = 100

    for e in range(epochs):
        for batch_idx, data in enumerate(train_loader):
            data = data.to(device)
            optimizer.zero_grad()
            recon_batch, mu, logvar = model(data)
            loss_recon, loss_kld = vae_loss(recon_batch, data, mu, logvar)
            loss = loss_recon + loss_kld
            loss.backward()
            train_loss += loss.item() / batch_size
            optimizer.step()

            percent_comp = ((batch_idx + 1) + e * batch_count) / (batch_count * epochs)
            time = datetime.datetime.now().time()
            date = datetime.datetime.now().date()
            df = pd.DataFrame([[batch_idx,percent_comp,loss.item()
                                ,loss_recon.item(), loss_kld.item(),time,date]],
                              columns = ['batch_idx', '%_comp', 'batch_loss',
                                         'mse_loss', 'kld_loss', 'time', 'date'])

            if batch_idx % print_interval == 0:
                print('===>percent complete: ' + str(percent_comp) +
                      ' |loss_recon = ' + str(loss_recon) +
                      ' |loss_kld = ' + str(loss_kld))

            if batch_idx % save_interval == 0:
                # Save data log
                with open(train_save_path + '.txt', 'a') as f:
                    record = df.to_json(orient='records')
                    f.write(record)
                    f.write(os.linesep)

                # Save model and optimizer state dicts
                save_checkpoint({'epoch': e,
                                 'state_dict': model.state_dict(),
                                 'optimizer' : optimizer.state_dict()},
                                train_save_path, is_best=False)


def train_seg(epoch, model, optimizer, train_loader, save_path, device=None):
    ''' Train the VAE model.
    Parameters
    ----------
    epoch : int
        The number of epochs for which to train the model.
    model : torch.nn.Module
        The model to train.
    optimizer : torch.optim.Optimizer
        The optimizer to use in training the model.
    train_loader : torch.utils.data.DataLoader
        The data loader to use for training.
    save_path : Union[str, pathlib.Path]
        The path to which to save the model.
    device : Optional[str]
        The device to  use for training, or `None` to use CUDA where available.
    '''
    device = device if device else 'cuda' if torch.cuda.is_available() else 'cpu'
    model = model.to(device)
    model.train()
    train_loss = 0
    batch_count = len(train_loader)

    for e in range(epochs):
        for batch_idx, data in enumerate(train_loader):
            data = data.to(device)
            input = data[0].float().to(device)
            target = data[1].float().to(device)
            optimizer.zero_grad()
            recon_batch, mu, logvar = model(input)
            loss_ce, loss_kld = vae_seg_loss(recon_batch,
                                                target.view(batch_size, 64, 64).long(),
                                                mu, logvar)

            # TODO: apply loss weighting for semantic segmentation training
            loss = loss_ce + loss_kld
            loss.backward()
            train_loss += loss.item() / len(data)
            optimizer.step()

            percent_comp = ((batch_idx + 1) + e * batch_count) / (batch_count * epochs)
            time = datetime.datetime.now().time()
            date = datetime.datetime.now().date()
            df = pd.DataFrame([[batch_idx,percent_comp,loss.item(),
                                loss_ce.item(), loss_kld.item(),time,date]],
                              columns = ['batch_idx', '%_comp', 'batch_loss',
                                         'ce_loss', 'kld_loss', 'time', 'date'])

            if batch_idx % print_interval == 0:
                print('===>percent complete: ' + str(percent_comp) +
                      ' |loss_recon = ' + str(loss_recon) +
                      ' |loss_ce = ' + str(loss_ce))

            if batch_idx % save_interval == 0:
                # Save data log
                with open(train_save_path + '.txt', 'a') as f:
                    record = df.to_json(orient='records')
                    f.write(record)
                    f.write(os.linesep)

                # Save model and optimizer state dicts
                save_checkpoint({'epoch': e,
                                 'state_dict': model.state_dict(),
                                 'optimizer' : optimizer.state_dict()},
                                train_save_path, is_best=False)


os.environ["CUDA_VISIBLE_DEVICES"] = "0"
batch_size = 128
num_workers = int(args.processes)
epochs = 20

save_interval = 1000
print_interval = 100
obs_dir = 'data/' + str(args.mode)
train_save_path = 'trained/vae_train_' + str(args.mode)


if not os.path.exists('trained'):
    os.makedirs('trained')

if args.mode == 'standard' or args.mode == 'texture':
    data_class = ObservationDataset(data_dir=obs_dir,
                                    transform=transforms.Compose([ToFloat(),
                                                                  RandomFlip(),
                                                                  ObservationToTensor()]))
    train_loader = torch.utils.data.DataLoader(data_class, batch_size=batch_size,
                                           pin_memory=True, num_workers=num_workers)
    model = VAE(True)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    train(epochs, model, optimizer, train_loader, train_save_path)
elif args.mode == 'segment':
    obs_dir = 'data/texture'
    seg_dir = 'data/segment'
    obs_class = ObservationDataset(data_dir=obs_dir,
                                   transform=transforms.Compose([ToFloat(),
                                                                 ObservationToTensor()]))
    seg_class = SegmentationDataset(data_dir=seg_dir,
                                    transform=transforms.Compose([ToSeg(),
                                                                  SegmentationToTensor()]))
    data_class = ConcatDataset(obs_class, seg_class)
    train_loader = torch.utils.data.DataLoader(data_class, batch_size=batch_size,
                                           pin_memory=True, num_workers=num_workers)
    model = VAE(True)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    train_seg(epochs, model, optimizer, train_loader, train_save_path)
