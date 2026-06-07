from itertools import cycle
import torch
import torch.nn as nn


class WeightedSum(nn.Module):
    def __init__(self, num_rows):
        super(WeightedSum, self).__init__()
        # Initialize learnable weights
        self.weights = nn.Parameter(torch.randn(num_rows))

    def forward(self, x):
        # Ensure weights are normalized (optional)
        normalized_weights = self.weights / self.weights.sum()  # torch.softmax(self.weights, dim=0)
        # Compute the weighted sum of the rows
        weighted_sum = torch.matmul(normalized_weights, x)
        return weighted_sum


def wrapped_getattr(self, name, default=None, wrapped_member_name='model'):
    ''' should be called from wrappers of model classes such as ClassifierFreeSampleModel'''

    if isinstance(self, torch.nn.Module):
        # for descendants of nn.Module, name may be in self.__dict__[_parameters/_buffers/_modules] 
        # so we activate nn.Module.__getattr__ first.
        # Otherwise, we might encounter an infinite loop
        try:
            attr = torch.nn.Module.__getattr__(self, name)
        except AttributeError:
            wrapped_member = torch.nn.Module.__getattr__(self, wrapped_member_name)
            attr = getattr(wrapped_member, name, default)
    else:
        # the easy case, where self is not derived from nn.Module
        wrapped_member = getattr(self, wrapped_member_name)
        attr = getattr(wrapped_member, name, default)
    return 

def to_numpy(tensor):
    if torch.is_tensor(tensor):
        return tensor.cpu().numpy()
    elif type(tensor).__module__ != 'numpy':
        raise ValueError("Cannot convert {} to numpy array".format(
            type(tensor)))
    return tensor


def to_torch(ndarray):
    if type(ndarray).__module__ == 'numpy':
        return torch.from_numpy(ndarray)
    elif not torch.is_tensor(ndarray):
        raise ValueError("Cannot convert {} to torch tensor".format(
            type(ndarray)))
    return ndarray


def cleanexit():
    import sys
    import os
    try:
        sys.exit(0)
    except SystemExit:
        os._exit(0)

def load_model_wo_clip(model, state_dict):
    missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
    assert len(unexpected_keys) == 0, f"Got the following unxepected keys {unexpected_keys}"
    assert all([k.startswith('clip_model.') for k in missing_keys]), f"Got the following unxepected keys {missing_keys}"

def freeze_joints(x, joints_to_freeze):
    # Freezes selected joint *rotations* as they appear in the first frame
    # x [bs, [root+n_joints], joint_dim(6), seqlen]
    frozen = x.detach().clone()
    frozen[:, joints_to_freeze, :, :] = frozen[:, joints_to_freeze, :, :1]
    return frozen


class AlternatingIterable:
    def __init__(self, iterable1, iterable2):
        """
        Initialize the generator with two subgenerators.
        Args:
            iterable1 (iterable): The first generator (iteration stops when this is exhausted).
            iterable2 (iterable): The second generator (can continue cycling if needed).
        """
        self.iterable1 = iter(iterable1)
        self.iterable2 = cycle(iterable2)
        self.switch = True  # To alternate between iterable1 and iterable

    def __iter__(self):
        return self

    def __next__(self):
        """
        Yield an item alternately from iterable1 and iterable.
        Stops iteration when iterable1 is exhausted.
        """
        if self.switch:  # Alternate between iterable1 and iterable
            try:
                item = next(self.iterable1)
            except StopIteration:
                raise StopIteration  # Stop when iterable1 is exhausted
        else:
            item = next(self.iterable2)

        self.switch = not self.switch  # Toggle the switch
        return item

    def was_last_other_data(self):
        return self.switch
    
    def __len__(self):
        return 2 * len(self.iterable1)
