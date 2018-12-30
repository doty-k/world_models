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

import gym
import cv2
import numpy as np
import os
import math
from multiprocessing import Pool
import gc
import argparse


parser = argparse.ArgumentParser(description='Script for generating V and M model training data')
parser.add_argument('rollouts', help= "The number of random rollouts to generate")
parser.add_argument('num_processes', help= "The number of processes to use to generate rollouts")
args = parser.parse_args()


def sample_continuous_policy(action_space, seq_len, dt):
    ''' Sample a continuous brownian motion policy
    Parameters
    ----------
    action_space : gym.spaces.box.Box
        Gym environment action space
    seq_len : int
        Length of sequence
    dt : float
        Temporal discretization
    Returns
    -------
    actions : list of np.array
        Action sequence for rollout
    '''
    actions = [action_space.sample()]
    for _ in range(seq_len - 1):
        daction_dt = np.random.randn(*actions[-1].shape)
        actions.append(
            np.clip(actions[-1] + math.sqrt(dt) * daction_dt,
                    action_space.low, action_space.high))
    return actions


def generate_data(rollout):
    ''' Generate observation and action data for a random rollout
    in the CarRacing environment
    Parameters
    ----------
    rollout : Int
        An integer specifying the rollout index
    Saves
    -------
    a_rollout : np.array
        Sequence of actions for the rollout
    texture : np.array
        Sequence of textured observations
    segment : np.array
        Sequence of semantically segmented observations
    standard : np.array
        Sequence of default CarRacing observations
    '''
    env = gym.make('CarRacing-v0')
    obs = env.reset()
    env.env.viewer.window.dispatch_events()

    a_rollout = sample_continuous_policy(env.action_space, seq_len, 1./50)

    # save actions by rollout
    np.save(act_dir + '/' + str(rollout).zfill(4) + '.npy', np.array(a_rollout))

    step = 0
    while True:
        act = a_rollout[step]

        obs, reward, done, _ = env.step(act)
        env.env.viewer.window.dispatch_events()
        texture = cv2.resize(obs[0], dsize=(64, 64),
                             interpolation=cv2.INTER_NEAREST).astype('uint8')
        segment = cv2.resize(obs[1], dsize=(64, 64),
                             interpolation=cv2.INTER_NEAREST).astype('uint8')
        standard = cv2.resize(obs[2], dsize=(64, 64),
                              interpolation=cv2.INTER_NEAREST).astype('uint8')

        # save observation by step
        np.save(texture_dir + '/' + str(rollout).zfill(4) + str(step).zfill(3)
                + '.npy', texture)
        np.save(segment_dir + '/' + str(rollout).zfill(4) + str(step).zfill(3)
                + '.npy', segment)
        np.save(standard_dir + '/' + str(rollout).zfill(4) + str(step).zfill(3)
                + '.npy', standard)

        step +=1

        if done:
            env.close()
            del env, obs, a_rollout, act, reward, texture, segment, standard, step
            gc.collect()
            print(">>>>> Finished rollout {}".format(rollout))
            break


rollouts = int(args.rollouts)
seq_len = 1000
num_processes = int(args.num_processes)

texture_dir = 'data/texture'
segment_dir = 'data/segment'
standard_dir = 'data/standard'
act_dir = 'data/actions'

for directory in [texture_dir, segment_dir, standard_dir,  act_dir]:
    if not os.path.exists(directory):
        os.makedirs(directory)

# check for existing data
vals = set(range(rollouts)) - set(int(i[0:4].lstrip('0')) if i!='0000.npy'
                                  else 0 for i in os.listdir(act_dir))
pool = Pool(processes=num_processes, maxtasksperchild=100)
pool.map(generate_data, vals)
