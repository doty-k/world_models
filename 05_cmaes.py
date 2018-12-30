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

''' Control model training code '''

import os
import datetime
import argparse
from time import sleep

import cv2
import cma
import gym
import numpy as np
import pandas as pd

import torch
from torch.multiprocessing import Process, Queue

from models.models import VAE, MDNRNN, CONTROLLER
from utils.utils import flatten_parameters, load_parameters, crop_and_resize, check_mode

parser = argparse.ArgumentParser()
parser.add_argument('mode', help="The compressed observation sequence you want to train "
                    + "the MDNRNN on. Options are standard, texture, segment.",
                    type=check_mode)
parser.add_argument('processes', help="The number of processes you want to roll out "
                    + "agents with.")
parser.add_argument('model_load_path',
                    nargs='?',
                    default = None,
                    help = "Optional argument specifying path to saved model, if you "
                    + "want to continue training a model")
args = parser.parse_args()


class RolloutGenerator(object):
    ''' Generate a single rollout '''
    def __init__(self, v_path, m_path, device=None):
        ''' Initialize a RolloutGenerator object.

        Parameters
        ----------
        v_path : Union[str, pathlib.Path]
            Path to a saved V model (VAE) checkpoint.
        m_path : Union[str, pathlib.Path]
            Path to a saved M model (memory: LSTM) checkpoint.
        device : Optional[str]
            The device to onto which to push tensors, or None to use CUDA where available.
        '''
        self.vae = VAE().to(device)
        checkpoint_v = torch.load(v_path)
        self.vae.load_state_dict(checkpoint_v['state_dict'])

        self.mdnrnn = MDNRNN().to(device)
        checkpoint_m = torch.load(m_path)
        self.mdnrnn.load_state_dict(checkpoint_m['state_dict'])

        self.control = CONTROLLER().to(device)

        self.env = gym.make('CarRacing-v0')
        self.device = device if device else 'cuda' if torch.cuda.is_available() else 'cpu'


    def pre_process(self, obs):
        ''' Preprocess an input observation.

        Parameters
        ----------
        obs : numpy.ndarray, shape=(R, C, 3)
            The observation (image).
        Returns
        -------
        x : torch.Tensor, shape=(1, 3, 64, 64)
            The cropped-and-resized observation.
        '''
        x = cv2.resize(obs, dsize=(64, 64), interpolation=cv2.INTER_NEAREST)
        x = x.astype('float32') / 255
        x = x.transpose((2, 0, 1))
        x = torch.from_numpy(x).view(-1, 3, 64, 64)
        return x

    def rollout(self, params):
        ''' Perform a single rollout.

        Parameters
        ----------
        params : dict
            controller fully connected layer weights and biases
        Returns
        -------
        total_reward * -1 : float
            The negative reward accumulated during the rollout
        '''
        load_parameters(params, self.control)

        obs = self.env.reset()
        self.env.render()
        (h, c) = [torch.zeros(1, 1, 256).to(self.device) for _ in range(2)]
        total_reward = 0

        for step in range(1000):
            # obs[0] for textured observation
            # obs[1] for semantic segmentation observation
            # obs[2] for standard observation
            obs = self.pre_process(obs[obs_index]).to(self.device)
            # using mean as z vector
            z, _ = self.vae.encode(obs)
            a = self.control.get_action(z, h)
            obs, reward, done, _ = self.env.step(a)

            a = torch.from_numpy(a).view(1, 1, 3).to(self.device)
            x = torch.cat((z.view(1, 1, 32), a), 2)
            _, (h, c) = self.mdnrnn.lstm(x, (h, c))

            total_reward += reward
            if done:
                break
        return -1 * total_reward


def slave_routine(p_queue, r_queue, e_queue, p_index, device=None):
    device = device if device else 'cuda' if torch.cuda.is_available() else 'cpu'
    with torch.no_grad():
        r_gen = RolloutGenerator(v_path, m_path, device)

        while e_queue.empty():
            if p_queue.empty():
                sleep(0.1)
            else:
                s_id, params = p_queue.get()
                r_queue.put((s_id, r_gen.rollout(params)))


def evaluate(solutions, results, rollouts=100):
    ''' Evaluate performance of best controller in generation.

    Parameters
    ----------
    solutions : list(dict)
        model parameters for every model in current generation
    results : list(float)
        negative cumulative rewards for each controller
    rollouts : int
        number of times to rollout controller being evaluated
    Returns
    -------
    best_guess : dict
        parameters for evaluated controller. i.e. controller with current
        minimum negative rewards
    np.mean(restimates) : float
        average reward of controller over all rollouts
    np.std(restimates) : float
        standard deviation of controller reward over all rollouts
    '''
    index_min = np.argmin(results)
    best_guess = solutions[index_min]
    restimates = []

    for s_id in range(rollouts):
        p_queue.put((s_id, best_guess))

    print("Evaluating...")
    for _ in range(rollouts):
        while r_queue.empty():
            sleep(.1)
        restimates.append(r_queue.get()[1])

    return best_guess, np.mean(restimates), np.std(restimates)


v_path = 'trained/vae_train_' + str(args.mode) + '.pth.tar'
m_path = 'trained/mdnrnn_train_' + str(args.mode) + '.pth.tar'
c_save_path = 'trained/cmaes_train_' + str(args.mode) + '.pth.tar'
c_save_dir = 'cmaes_train_' + str(args.mode)
num_workers = int(args.processes)
c_load_path = args.model_load_path

if args.mode == 'standard':
    obs_index = 2
elif args.mode == 'texture':
    obs_index = 0
elif args.mode == 'segment':
    obs_index = 1

n_samples = 16
pop_size = 32
target = 930
log_interval = 1
weight_save_interval = 5
eval_interval = 25
sigma = 0.1

if c_save_dir not in os.listdir('trained'):
    print('Making model-saving directory:' + c_save_dir)
    os.mkdir('trained/' + c_save_dir)

p_queue = Queue()
r_queue = Queue()
e_queue = Queue()

for p_index in range(num_workers):
    Process(target=slave_routine, args=(p_queue, r_queue, e_queue, p_index)).start()

# dummy controller to initialize parameters
controller = CONTROLLER()
cur_best = None
epoch = 0

if c_load_path:
    print('====>>>> controller checkpoint loaded')
    checkpoint_c = torch.load(c_load_path)
    controller.load_state_dict(checkpoint_c['state_dict'])
    cur_best = -checkpoint_c['reward']
    epoch = checkpoint_c['epoch']
    print('====>>>> current best is: ' + str(cur_best))

parameters = controller.parameters()
es = cma.CMAEvolutionStrategy(flatten_parameters(parameters), sigma, {'popsize': pop_size})

while not es.stop():
    if cur_best is not None and (-cur_best) > target:
        print("Already better than target, breaking...")
        break

    r_list = [0] * pop_size
    solutions = es.ask()

    # push parameters to queue
    for s_id, s in enumerate(solutions):
        for _ in range(n_samples):
            p_queue.put((s_id, s))

    # retrieve results
    for _ in range(pop_size * n_samples):
        while r_queue.empty():
            sleep(.1)
        r_s_id, r = r_queue.get()
        r_list[r_s_id] += r / n_samples

    es.tell(solutions, r_list)
    es.disp()

    if epoch % log_interval == 0:
        # saving training log
        print('=======> Generation best is: ' + str(min(r_list)))
        time = datetime.datetime.now().time()
        date = datetime.datetime.now().date()
        t_log = pd.DataFrame([[epoch, -min(r_list), -max(r_list), -sum(r_list)/len(r_list), time, date]],
                             columns = ['generation', 'max_reward', 'min_reward', 'mean_reward', 'time', 'date'])
        with open(c_save_path + '_train.txt', 'a') as f:
            record = t_log.to_json(orient='records')
            f.write(record)
            f.write(os.linesep)

    if epoch % weight_save_interval == 0 and epoch != 0:
        # saving controller weights
        index_min = np.argmin(r_list)
        generation_best = solutions[index_min]
        load_parameters(generation_best, controller)
        torch.save({'epoch': epoch,
                    'reward': -np.min(r_list),
                    'state_dict': controller.state_dict()},
                    'trained/' + c_save_dir + '/' + str(epoch) + '.tar')

    if epoch % eval_interval == 0 and epoch != 0:
        best_params, best, std_best = evaluate(solutions, r_list)
        print('Current evaluation: ' + str(best))
        e_log = pd.DataFrame([[epoch, best, std_best, best_params]],
                             columns = ['generation', 'avg_reward', 'std_reward', 'parameters'])
        with open(c_save_path + '_eval.txt', 'a') as f:
            record = e_log.to_json(orient='records')
            f.write(record)
            f.write(os.linesep)


        # best comparrison is > because working with negative of rewards
        if not cur_best or cur_best > best:
            cur_best = best
            print("Saving new best with value {}±{}...".format(-cur_best, std_best))
            load_parameters(best_params, controller)
            torch.save({'epoch': epoch,
                        'reward': -cur_best,
                        'state_dict': controller.state_dict()},
                       c_save_path + 'best.tar')
        if -best > target:
            print("Terminating controller training with value {}...".format(-best))
            break

    epoch += 1
    print('current epoch: ' + str(epoch))

es.results_pretty()
e_queue.put('EOP')
