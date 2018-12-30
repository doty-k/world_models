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

''' Code for training the MDNRNN '''

import os
import datetime
from pathlib import Path
import argparse
import pandas as pd

import torch
from models.models import MDNRNN, mdnrnn_loss
from dataloader.data_class import RolloutDataset
from utils.utils import save_checkpoint, check_mode

parser = argparse.ArgumentParser()
parser.add_argument('mode', help="The compressed observation sequence you want to train the MDNRNN on. Options are standard, texture, segment.", type=check_mode)
parser.add_argument('processes', help="The number of processes to use for MDNRNN training")
args = parser.parse_args()

def train(epochs, model, optimizer, train_loader, save_path, device=None):
    ''' Train MDNRNN.

    Parameters
    ----------
    epochs : int
        The number of epochs for which to train the model.

    model : torch.nn.Module
        The model to train.

    optimizer : torch.optim.Optimizer
        The optimizer to use in training the model.

    train_loader : torch.utils.data.DataLoader
        The data loader to use in training.

    save_path : Union[str, pathlib.Path]
       The path to which to save the model.

    device : Optional[str]
        The device to use for training, or `None` to auto-detect whether CUDA can be used.
    '''
    device = device if device else 'cuda' if torch.cuda.is_available() else 'cpu'
    model = model.to(device).train()
    batch_count = len(train_loader)
    train_iters = batch_count * epochs

    for epoch in range(epochs):
        hidden = model.init_hidden(batch_size, device)

        for batch_idx, (inputs, targets) in enumerate(train_loader):
            inputs = inputs.to(device)   # shape (999, 1, 1, 35)
            targets = targets.to(device) # shape (999, 1, 1, 32)

            optimizer.zero_grad()

            hidden = model.init_hidden(inputs.size(1), device)

            y, (h, c) = model.lstm(inputs.view(-1, 1, 35), hidden)
            h = h.detach()
            c = c.detach()
            log_pi, mu, sigma = model.get_mixture(y)
            l = mdnrnn_loss(log_pi, mu, sigma, targets.view(-1, 1, 32))
            l.backward()
            optimizer.step()

            # data logging parameters
            percent_comp = ((batch_idx + 1) + (epoch * batch_count)) / train_iters
            time = datetime.datetime.now().time()
            date = datetime.datetime.now().date()
            df = pd.DataFrame([[batch_idx, percent_comp, l.item(), time, date]],
                              columns=['batch_idx', '%_comp', 'batch_loss', 'time', 'date'])

            # printing progress
            if batch_idx % print_interval == 0:
                print('percent complete: ' + str(percent_comp) + ', batch: ' + str(batch_idx)
                      + ', loss: ' + str(l))

            # logging progress
            if batch_idx % save_interval == 0:
                # Save training log
                with open(save_path + '.txt', 'a') as f:
                    record = df.to_json(orient='records')
                    f.write(record)
                    f.write(os.linesep)

                # Save model and optimizer state dicts
                save_checkpoint({'epoch': epoch,
                                 'state_dict': model.state_dict(),
                                 'optimizer': optimizer.state_dict()},
                                save_path, True)


os.environ["CUDA_VISIBLE_DEVICES"] = "0"
batch_size = 1
epochs = 40
num_workers = int(args.processes)

action_dir = 'data/actions'
mean_dir = 'data/' + str(args.mode) + '_mean'
var_dir = 'data/' + str(args.mode) + '_var'
save_path = 'trained/mdnrnn_train_' + str(args.mode)

save_interval = 1000
print_interval = 100
model = MDNRNN()
optimizer = torch.optim.Adam(model.parameters())
data_class = RolloutDataset(action_dir, mean_dir, var_dir, transform=None)
train_loader = torch.utils.data.DataLoader(data_class, batch_size, pin_memory=True,
                                           num_workers=num_workers)

train(epochs, model, optimizer, train_loader, save_path)
