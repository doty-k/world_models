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

'''Code for dataloader transformations'''

import torch
from torchvision import transforms, utils
import numpy as np
import torch.nn.functional as F

class ToSeg(object):
    '''convert ndarray segmented image to single channel class map'''

    def __call__(self, image):
        image = image[:,:,1]
        image[image == 102] = 1
        image[image == 204] = 2
        return image

class ToFloat(object):
    '''convert uint8 array to 0->1 float'''

    def __call__(self, image):
        return image.astype('float32') / 255

class SegmentationToTensor(object):
    '''Convert single channel ndarrays to tensor'''

    def __call__(self, image):
        return torch.from_numpy(image).view(1, 64, 64)

class ObservationToTensor(object):
    '''Convert ndarrays to tensors'''

    def __call__(self, image):
        image = image.transpose((2, 0, 1))
        return torch.from_numpy(image).view(3, 64, 64)

class ModelToTensor(object):
    '''Convert ndarray VAE compressed observations to tensors'''

    def __call__(self, model):
        return torch.from_numpy(model).view(300, 32)

class ModelActionToTensor(object):
    '''Convert ndarray VAE compressed observation + action to tensor'''

    def __call__(self, array):
        return torch.from_numpy(array).view(300, 35)

class RandomFlip(object):
    '''With 50% chance, randomly flip array about transverse axis'''

    def __call__(self, array):
        r = np.random.choice(2, 1)
        if r:
            array = np.flip(array, 1).copy()
        return array

