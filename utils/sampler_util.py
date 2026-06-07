import numpy as np
import torch
import torch.nn as nn
from copy import deepcopy
from utils.misc import wrapped_getattr
import joblib

# A wrapper model for Classifier-free guidance **SAMPLING** only
# https://arxiv.org/abs/2207.12598
class ClassifierFreeSampleModel(nn.Module):

    def __init__(self, model):
        super().__init__()
        self.models = [model] # This list usage is a weird hack, other wise with direct assignment the model becomes None as an attribute
        assert self.models[0].cond_mask_prob > 0, 'Cannot run a guided diffusion on a model that has not been trained with no conditions'

        # pointers to inner model
        self.rot2xyz = self.models[0].rot2xyz
        self.translation = self.models[0].translation
        self.njoints = self.models[0].njoints
        self.nfeats = self.models[0].nfeats
        self.data_rep = self.models[0].data_rep
        self.cond_mode = self.models[0].cond_mode
        self.encode_text = self.models[0].encode_text

    def forward(self, x, timesteps, y=None):
        cond_mode = self.models[0].cond_mode
        assert cond_mode in ['text', 'action']
        y_uncond = deepcopy(y)
        y_uncond['uncond'] = True
        out = self.models[0](x, timesteps, y)
        out_uncond = self.models[0](x, timesteps, y_uncond)
        return out_uncond + (y['scale'].view(-1, 1, 1, 1) * (out - out_uncond))

    def __getattr__(self, name, default=None):
        # this method is reached only if name is not in self.__dict__.
        return wrapped_getattr(self, name, default=None)
    
    def parameters(self):
        return self.models[0].parameters()
    
    def to(self, device):
        self.models[0].to(device)
        return self
    
    def eval(self):
        self.models[0].eval()
        return self
    
    def train(self):
        self.models[0].train()
        return self


class AutoRegressiveSampler():
    def __init__(self, args, sample_fn, required_frames=196):
        self.sample_fn = sample_fn
        self.args = args
        self.required_frames = required_frames
    
    def sample(self, model, shape, **kargs):
        bs = shape[0]
        n_iterations = (self.required_frames // self.args.pred_len) + int(self.required_frames % self.args.pred_len > 0)
        samples_buf = []
        cur_prefix = deepcopy(kargs['model_kwargs']['y']['prefix'])  # init with data
        dynamic_text_mode = type(kargs['model_kwargs']['y']['text'][0]) == list  # Text changes on the fly - prompt per prediction is provided as a list (instead of a single prompt)
        if self.args.autoregressive_include_prefix:
            samples_buf.append(cur_prefix)
        autoregressive_shape = list(deepcopy(shape))
        autoregressive_shape[-1] = self.args.pred_len
        
        # Autoregressive sampling
        for i in range(n_iterations):
            
            # Build the current kargs
            cur_kargs = deepcopy(kargs)
            cur_kargs['model_kwargs']['y']['prefix'] = cur_prefix
            if dynamic_text_mode:
                cur_kargs['model_kwargs']['y']['text'] = [s[i] for s in kargs['model_kwargs']['y']['text']]
                if model.text_encoder_type == 'bert':
                    cur_kargs['model_kwargs']['y']['text_embed'] = (cur_kargs['model_kwargs']['y']['text_embed'][0][:, :, i], cur_kargs['model_kwargs']['y']['text_embed'][1][:, i])
                else:
                    raise NotImplementedError('DiP model only supports BERT text encoder at the moment. If you implement this, please send a PR!')
            
            # Sample the next prediction
            sample = self.sample_fn(model, autoregressive_shape, **cur_kargs)

            # Buffer the sample
            samples_buf.append(sample.clone()[..., -self.args.pred_len:])

            # Update the prefix
            cur_prefix = sample.clone()[..., -self.args.context_len:]

        full_batch = torch.cat(samples_buf, dim=-1)[..., :self.required_frames]  # 200 -> 196
        return full_batch


class UnconditionalGuidadedSampler(nn.Module):
    def __init__(self, model, time_gap, min_timestep, max_timestep):
        super().__init__()
        self.models = [model] # This list usage is a weird hack, other wise with direct assignment the model becomes None as an attribute
        assert self.models[0] is not None, "Model must be provided"
        self.time_gap = time_gap
        self.min_timestep = min_timestep
        self.max_timestep = max_timestep
        # pointers to inner model
        self.rot2xyz = self.models[0].rot2xyz
        self.translation = self.models[0].translation
        self.njoints = self.models[0].njoints
        self.nfeats = self.models[0].nfeats
        self.data_rep = self.models[0].data_rep
        self.cond_mode = self.models[0].cond_mode
        self.encode_text = self.models[0].encode_text

        self.timesteps_cache = {}

    def reset_timesteps_cache(self):
        self.timesteps_cache = {}

    def forward(self, x, timesteps, y=None):
        assert torch.all(timesteps.float().mean() == timesteps), "Timesteps must be constant"
        scalar_timestep = timesteps.float().mean().item()
        assert scalar_timestep not in self.timesteps_cache, "Timestep must not be in cache already"
        out = self.models[0](x, timesteps, y)
        if scalar_timestep > self.min_timestep and scalar_timestep + self.time_gap in self.timesteps_cache:
            previous_out = self.timesteps_cache[scalar_timestep + self.time_gap]
        else:
            assert scalar_timestep + self.time_gap >= self.max_timestep or scalar_timestep <= self.min_timestep, "Timestep + time_gap is not in cache, but should be."
            previous_out = out
        assert scalar_timestep not in self.timesteps_cache, "Timestep must not be in cache already, you should call reset_timesteps_cache() before sampling again."
        self.timesteps_cache[scalar_timestep] = out
        return previous_out + (y['scale'].view(-1, 1, 1, 1) * (out - previous_out))

    def __getattr__(self, name, default=None):
        # this method is reached only if name is not in self.__dict__.
        return wrapped_getattr(self, name, default=None)
    
    def parameters(self):
        return self.models[0].parameters()
    
    def to(self, device):
        self.models[0].to(device)
        return self
    
    def eval(self):
        self.models[0].eval()
        return self
    
    def train(self):
        self.models[0].train()
        return self