import os
import json
import argparse
from model_2d.mdm_2d import MDM_2D
from diffusion import gaussian_diffusion as gd
from diffusion.respace import SpacedDiffusion, space_timesteps

def get_dims(args):
    if args.dataset in ["video_mdm_synthetic", "video_mdm_synthetic_mvlift", "humanml", "egoexo", "egoexo_uncentered", "fit3d", "fit3d_mvlift"]:
        num_joints = 22
    elif args.dataset == "nba":
        num_joints = 16
    else:
        raise ValueError(f"The model was trained on dataset {args.dataset}, which number of joints is not in code.")
    nfeats = getattr(args, "nfeats", 2)
    return num_joints, nfeats

def get_num_actions(args):
    # Note: This is garbage and assumed it is not used.
    return 10

def add_defualt_diffusion_args(args):
    if not hasattr(args, "lambda_rcxyz"):
        args.lambda_rcxyz = 0.
    if not hasattr(args, "lambda_vel"):
        args.lambda_vel = 0.
    if not hasattr(args, "lambda_root_vel"):
        args.lambda_root_vel = 0.
    if not hasattr(args, "lambda_vel_rcxyz"):
        args.lambda_vel_rcxyz = 0.
    if not hasattr(args, "lambda_fc"):
        args.lambda_fc = 0.
    return args
    

def create_model_2d_and_diffusion_from_path(model_dir_path, **kwargs):
    if not os.path.exists(model_dir_path):
        raise FileNotFoundError(f"Model directory {model_dir_path} not found. Passed as --model_2d_path")
    if not os.path.isdir(model_dir_path):
        raise FileNotFoundError(f"Model directory {model_dir_path} is not a directory. Passed as --model_2d_path")
    args_path = os.path.join(model_dir_path, 'args.json')
    if not os.path.exists(args_path):
        raise FileNotFoundError(f"args.json not found in {model_dir_path}. Passed as --model_2d_path")
    with open(args_path, 'r') as f:
        args = json.load(f)

    # Make args a namespace
    args = argparse.Namespace(**args)
    args = add_defualt_diffusion_args(args)
    
    model, diffusion = create_model_2d_and_diffusion(args, **kwargs)
    model.diffusion = diffusion
    return model


def create_model_2d_and_diffusion(args, **kwargs):
    model_args = get_model_args(args)
    model_args.update(kwargs)
    model = MDM_2D(**model_args)
    diffusion = create_spaced_diffusion(args)
    return model, diffusion


def get_model_args(args):
    return {
        "njoints": get_dims(args)[0],
        "nfeats": get_dims(args)[1],
        "num_actions": get_num_actions(args),
        "latent_dim": args.latent_dim,
        "ff_size": args.ff_size,
        "num_layers": args.layers,
        "num_heads": args.num_heads,
        "dropout": args.dropout,
        "activation": args.activation,
        "cond_mode": args.cond,
        "cond_mask_prob": args.cond_mask_prob,
        "arch": args.arch,
        "emb_trans_dec": args.emb_trans_dec,
        "clip_version": "ViT-B/32",
    }

"""Example args.json file:
{
    "activation": "gelu",
    "arch": "trans_enc",
    "cond": "text",
    "cond_mask_prob": 0.1,
    "data_augmentations": [],
    "data_size": null,
    "data_split": "train",
    "datapath": "./dataset/egoexo_uncentered",
    "dataset": "egoexo_uncentered",
    "device": 0,
    "diffusion_steps": 100,
    "dropout": 0.1,
    "emb_trans_dec": false,
    "eval_during_training": false,
    "ff_size": 1024,
    "latent_dim": 512,
    "layers": 8,
    "lr": 1e-05,
    "model_mean_type": "x_start",
    "nople_schedule": "cosine_tau_2",
    "num_heads": 4,
    "num_steps": 600000,
    "overwrite_model": true,
    "resume_checkpoint": "",
    "save_dir": "save/egoexo_uncentered/attempt_text_3",
    "save_interval": 50000,
    "seed": 0,
    "sigma_small": true,
    "train_batch_size": 64,
    "train_platform_type": "NoPlatform",
    "use_l1": false,
    "velocities_loss": 0
}
"""

def create_spaced_diffusion(args):
    return SpacedDiffusion(
        use_timesteps=space_timesteps(args.diffusion_steps, [args.diffusion_steps]),
        betas=gd.get_named_beta_schedule(args.noise_schedule, args.diffusion_steps, 1.0),
        model_mean_type={"epsilon": gd.ModelMeanType.EPSILON, "x_start": gd.ModelMeanType.START_X, "previous_x": gd.ModelMeanType.PREVIOUS_X}[args.model_mean_type],
        model_var_type=(gd.ModelVarType.FIXED_LARGE if not args.sigma_small else gd.ModelVarType.FIXED_SMALL),
        loss_type=gd.LossType.MSE,
        rescale_timesteps=False,
        args=args,
    )
