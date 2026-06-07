# This code is based on https://github.com/openai/guided-diffusion
"""
Helpers for various likelihood-based losses. These are ported from the original
Ho et al. diffusion models codebase:
https://github.com/hojonathanho/diffusion/blob/1e0dceb3b3495bbe19116a5e1b3596cd0706c543/diffusion_tf/utils.py
"""

import numpy as np
from diffusion.nn import mean_flat, sum_flat
import torch

def angle_l2(angle1, angle2):
    a = angle1 - angle2
    a = (a + (torch.pi/2)) % torch.pi - (torch.pi/2)
    return a ** 2

def diff_l2(a, b):
    return (a - b) ** 2

def masked_l2(a, b, mask, loss_fn=diff_l2, epsilon=1e-8, entries_norm=True):
    # assuming a.shape == b.shape == bs, J, Jdim, seqlen
    # assuming mask.shape == bs, 1, 1, seqlen
    loss = loss_fn(a, b)
    loss = sum_flat(loss * mask.float())  # gives \sigma_euclidean over unmasked elements
    n_entries = a.shape[1]
    if len(a.shape) > 3:
        n_entries *= a.shape[2]
    non_zero_elements = sum_flat(mask)
    if entries_norm:
        # In cases the mask is per frame, and not specifying the number of entries per frame, this normalization is needed,
        # Otherwise set it to False
        non_zero_elements *= n_entries
    # print('mask', mask.shape)
    # print('non_zero_elements', non_zero_elements)
    # print('loss', loss)
    mse_loss_val = loss / (non_zero_elements + epsilon)  # Add epsilon to avoid division by zero
    # print('mse_loss_val', mse_loss_val)
    return mse_loss_val

def normal_kl(mean1, logvar1, mean2, logvar2):
    """
    Compute the KL divergence between two gaussians.

    Shapes are automatically broadcasted, so batches can be compared to
    scalars, among other use cases.
    """
    tensor = None
    for obj in (mean1, logvar1, mean2, logvar2):
        if isinstance(obj,  torch.Tensor):
            tensor = obj
            break
    assert tensor is not None, "at least one argument must be a Tensor"

    # Force variances to be Tensors. Broadcasting helps convert scalars to
    # Tensors, but it does not work for  torch.exp().
    logvar1, logvar2 = [
        x if isinstance(x,  torch.Tensor) else  torch.tensor(x).to(tensor)
        for x in (logvar1, logvar2)
    ]

    return 0.5 * (
        -1.0
        + logvar2
        - logvar1
        +  torch.exp(logvar1 - logvar2)
        + ((mean1 - mean2) ** 2) *  torch.exp(-logvar2)
    )


def approx_standard_normal_cdf(x):
    """
    A fast approximation of the cumulative distribution function of the
    standard normal.
    """
    return 0.5 * (1.0 +  torch.tanh(np.sqrt(2.0 / np.pi) * (x + 0.044715 *  torch.pow(x, 3))))


def discretized_gaussian_log_likelihood(x, *, means, log_scales):
    """
    Compute the log-likelihood of a Gaussian distribution discretizing to a
    given image.

    :param x: the target images. It is assumed that this was uint8 values,
              rescaled to the range [-1, 1].
    :param means: the Gaussian mean Tensor.
    :param log_scales: the Gaussian log stddev Tensor.
    :return: a tensor like x of log probabilities (in nats).
    """
    assert x.shape == means.shape == log_scales.shape
    centered_x = x - means
    inv_stdv =  torch.exp(-log_scales)
    plus_in = inv_stdv * (centered_x + 1.0 / 255.0)
    cdf_plus = approx_standard_normal_cdf(plus_in)
    min_in = inv_stdv * (centered_x - 1.0 / 255.0)
    cdf_min = approx_standard_normal_cdf(min_in)
    log_cdf_plus =  torch.log(cdf_plus.clamp(min=1e-12))
    log_one_minus_cdf_min =  torch.log((1.0 - cdf_min).clamp(min=1e-12))
    cdf_delta = cdf_plus - cdf_min
    log_probs =  torch.where(
        x < -0.999,
        log_cdf_plus,
         torch.where(x > 0.999, log_one_minus_cdf_min,  torch.log(cdf_delta.clamp(min=1e-12))),
    )
    assert log_probs.shape == x.shape
    return log_probs
