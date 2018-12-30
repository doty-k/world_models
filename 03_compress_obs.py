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

import numpy as np
import os
import argparse
from multiprocessing import Pool

import torch
from torch.autograd import Variable
from torch import nn, optim
from models.models import VAE
from utils.utils import crop_and_resize, check_mode

parser = argparse.ArgumentParser()
parser.add_argument('mode', help="The compressed observation sequence you want to train the MDNRNN on. Options are standard, texture, segment.", type=check_mode)
parser.add_argument('processes', help="The number of processes to use in rollout compression")
args = parser.parse_args()

def process_rollout_list(r_list):
    ''' Train the VAE model.
    Parameters
    ----------
    r_list : list(string)
        List of filenames for saved observations from rollout
    Saves
    -----
    mu : np.array
        Mean vectors of VAE observation compressions
    var : np.array
        Variance vectors of VAE observation compressions
    '''
    with torch.no_grad():
        model = VAE().to(device)
        checkpoint = torch.load(vae_path)
        model.load_state_dict(checkpoint['state_dict'])
        print(r_list[0][:4])
        data = torch.Tensor()

        for i in range(len(r_list)):
            obs = np.load(obs_dir + '/' + str(r_list[i]))
            obs = obs.astype('float32') / 255
            obs = obs.transpose((2, 0, 1))
            obs = torch.from_numpy(obs).view(-1, 3, 64, 64)
            data = torch.cat((data, obs), 0)

        data = data.to(device)
        _, mu, log_var = model(data)
        var = torch.exp(log_var)

        np.save(mean_dir + '/' + r_list[0][:4] + '.npy', mu.detach().cpu().numpy()
                .reshape(-1, 32))
        np.save(var_dir + '/' + r_list[0][:4] + '.npy', var.detach().cpu().numpy()
                .reshape(-1, 32))
        del data
        del mu
        del var
        del log_var
        torch.cuda.empty_cache()


rollouts = len(os.listdir('data/actions'))
num_workers = int(args.processes)

vae_path = 'trained/vae_train_' + str(args.mode) + '.pth.tar'
mean_dir = 'data/' + str(args.mode) + '_mean'
var_dir = 'data/' + str(args.mode) + '_var'
obs_dir = 'data/' + str(args.mode)

dir_list = (sorted(os.listdir(obs_dir)))
rollout_list = [[] for _ in range(rollouts)]
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
device = torch.device("cuda")


for directory in [mean_dir, var_dir]:
    if not os.path.exists(directory):
        os.makedirs(directory)

r = 0
# creating sorted lists of observation files organized by rollout
for i in range(len(dir_list)):
    if dir_list[i][3] != str(r)[-1]:
        r += 1
    rollout_list[r] += [dir_list[i]]

pool = Pool(num_workers)
pool.map(process_rollout_list, rollout_list)
