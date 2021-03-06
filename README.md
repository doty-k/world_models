# World Models

This repository contains code reproducing and extending the Car-Racing
experiments from the paper: [World Models](https://arxiv.org/pdf/1803.10122.pdf)

## Architecture

The World Models architecture consists of three components, detailed below. This
algorithm achieved state of the art results on the OpenAI Gym CarRacing task.

### 1. Vision Model
Made up of a variational autoencoder, the V-model learns a stochastic
latent space representation of the state observations from the environment.
This model is trained on data generated by a continuous random action selection
policy.

### 2. Memory Model
Consists of an mixture density LSTM. The M-model is trained to take
a compressed state representation and action as input and outputs a mixture of
gaussians distribution parameterizing the state representation for the next time
step. This model is trained on sequences of data generated by a continuous
random action selection policy.

### 3. Controller
The controller is a single fully connected layer. It's input is the
concatenation of the current state latent space representation and the hidden
state of the M-model LSTM. The output is an action vector. Controller parameters
are optimized  with CMA-ES.

## Extended Experiments

The CarRacing experiment of the original work is reproduced and extended by
training the architecture on a more visually rich environment. The rendering of
the CarRacing environment has been altered to produce three observation arrays
in a tuple when env.step() is called. The first item in the tuple is a textured
observation who's rendering includes detailed grass and pavement textures. The
second item is a semantically segmented array with three classes: grass, road,
and car. The third item is the standard observation that is usually output by
the environment.

The hypothesis was that the unsupervised training of the
v-model would retain features irrelevant to the task in the compressed state
representation, leading to degraded controller performance. However, the
textured observations were found to have little impact on controller performance
experimentally.

The semantically segmented observations are generated so the V-model can be
trained to perform semantic segmentation. It is possible this supervision could
lead to an increase in performance. That increase is likely to be small or
non-existent for this environment considering the strong performance of the
standard VAE in the textured environment. Currently a controller trained with V
and M models that were trained on segmented data does not learn a better policy
than a randomly initialized controller. This will be debugged when I find some
spare time.

## Prerequisites

* Python 3.5
* PyTorch
* OpenAI Gym
* pycma
* OpenCV
* pandas

## Usage

### Preliminary: Replace rendering code in CarRacing Environment

First car_racing.py and car_dynamics.py in the original CarRacing environment
code must be replaced with the versions of those files in the car_racing
directory of this repository. To do this, navigate to the environment code,
usually located at: (python-path)/site-packages/gym/envs/box2d. Make a directory
called original and move the original car_racing.py and car_dynamics.py there.
Then in box2d create symlinks to these files in this repo.

### 1. Generate random rollouts

``` bash
python 01_collect_data.py --rollouts 500000 --processes 24
```
Generate 500,000 rollouts of observation and corresponding action data. Saves at
data/actions, data/segment, data/standard, data/texture

### 2. Train V-model

``` bash
python 02_vae.py --mode standard --processes 12
```
Train V-model on standard observation data

### 3. Compress observations

``` bash
python 03_compress_obs.py --mode standard --processes 24
```
Create latent space sequence dataset of standard observations using standard
trained V-model to compress

### 4. Train M-model

``` bash
python 04_mdnrnn.py --mode standard --processes 1
```
Train M-model on standard latent space sequence data

### 5. Train Controller

``` bash
python 05_cmaes.py --mode standard --processes 12
```
Train and evaluate controller on standard observations with V-model and M-model
trained with standard data


## Acknowledgments

This work was completed while interning at MIT Lincoln Lab.

The code for brownian motion action selection and controller training
parallelization was inspired by the
[C. Tallec, L. Blier, D. Kalainathan reimplementation](https://github.com/ctallec/world-models)


## License

 DISTRIBUTION STATEMENT A. Approved for public release: distribution unlimited.

 This material is based upon work supported by the Assistant Secretary of Defense for Research and
 Engineering under Air Force Contract No. FA8721-05-C-0002 and/or FA8702-15-D-0001. Any opinions,
 findings, conclusions or recommendations expressed in this material are those of the author(s) and
 do not necessarily reflect the views of the Assistant Secretary of Defense for Research and
 Engineering.

 © 2018 Massachusetts Institute of Technology.

 MIT Proprietary, Subject to FAR52.227-11 Patent Rights - Ownership by the contractor (May 2014)

 The software/firmware is provided to you on an As-Is basis

 Delivered to the U.S. Government with Unlimited Rights, as defined in DFARS Part 252.227-7013 or
 7014 (Feb 2014). Notwithstanding any copyright notice, U.S. Government rights in this work are
 defined by DFARS 252.227-7013 or DFARS 252.227-7014 as detailed above. Use of this work other than
 as specifically authorized by the U.S. Government may violate any copyrights that exist in this
 work.
