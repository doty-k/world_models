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

''' A set of utility functions. '''

import numpy as np

import torch
import torch.nn.functional as F
from torch.utils.data.sampler import SubsetRandomSampler

def save_checkpoint(state, save_path, is_best):
    ''' Helper function to save a checkpoint.

    Parameters
    ----------
    state : Dict
        A PyTorch state dictionary containing the model state.

    save_path : Union[str, pathlib.Path]
        The path to which to save the checkpoint.

    is_best : bool
        Whether this is the current best model.
    '''
    # filename = Path(save_path)/'.pth.tar'
    filename = save_path + '.pth.tar'
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, save_path + '_best.pth.tar')


def train_test_split(data_class, test_ratio):

    # define indices for train-test split
    num_data = len(data_class)
    indices = list(range(num_data))
    split = int(num_data * test_ratio)

    # random, non-contiguous split
    validation_idx = np.random.choice(indices, size=split, replace=False)
    train_idx = list(set(indices) - set(validation_idx))

    # define samplers
    train_sampler = SubsetRandomSampler(train_idx)
    validation_sampler = SubsetRandomSampler(validation_idx)
    return train_sampler, validation_sampler


def flatten_parameters(params):
    return torch.cat([p.detach().view(-1) for p in params], dim=0).numpy()


def unflatten_parameters(params, example, device):
    params = torch.Tensor(params).to(device)
    idx = 0
    unflattened = []
    for e_p in example:
        unflattened += [params[idx:idx + e_p.numel()].view(e_p.size())]
        idx += e_p.numel()
    return unflattened


def load_parameters(params, controller):
    proto = next(controller.parameters())
    params = unflatten_parameters(
        params, controller.parameters(), proto.device)

    for p, p_0 in zip(controller.parameters(), params):
        p.data.copy_(p_0)


def crop_and_resize(feats, output_shape, *, n_start=None, n_stop=None, n_step=None, k_start=None, k_stop=None,
                    k_step=None, r_start=None, r_stop=None, r_step=None, c_start=None, c_stop=None, c_step=None):
    ''' Perform crop-and-resize on a feature map (a variant of RoI pooling).
    Parameters
    ----------
    feats : torch.Tensor, shape=(N, K, R, C)
        The input feature map on which to perform crop-and-resize.
    output_shape : Tuple[int, int]
        The spatial dimensions of the resized crops.
    n_start : Optional[int]
        The start index of the image dimension.
    n_stop : Optional[int]
        The end index of the image dimension.
    n_step : Optional[int]
        The step size in the image dimension.
    k_start : Optional[int]
        The start index of the channel dimension.
    k_stop : Optional[int]
        The end index of the channel dimension.
    k_step : Optional[int]
        The step size in the channel dimension.
    r_start : Optional[int]
        The start index of the (spatial) row dimension.
    r_stop : Optional[int]
        The end index of the (spatial) row dimension.
    r_step : Optional[int]
        The step size in the (spatial) row dimension.
    c_start : Optional[int]
        The start index of the (spatial) column dimension.
    c_stop : Optional[int]
        The end index of the (spatial) column dimension.
    c_step : Optional[int]
        The step size in the (spatial) column dimension.
    Returns
    -------
    torch.Tensor
        The cropped-and-resized features.
    Notes
    -----
    This function uses the {n,k,r,c}_{start,stop,step} sentinal values to construct slices for indexing into the
    provided feature map.
    '''
    n_slice = slice(n_start, n_stop, n_step)
    k_slice = slice(k_start, k_stop, k_step)
    r_slice = slice(r_start, r_stop, r_step)
    c_slice = slice(c_start, c_stop, c_step)
    crop = feats[(n_slice, k_slice, r_slice, c_slice)].contiguous()
    return F.upsample(crop, output_shape, mode='bilinear', align_corners=False)
