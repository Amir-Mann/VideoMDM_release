"""
Use a 3d diffusion model, 2d estimations and cameras to create a 3d sample consistant with the video.
"""

import json
import os
import subprocess

import numpy as np
import torch
from data_loaders.get_data import get_collate_fn, get_dataset
from data_loaders.humanml.scripts.motion_process import get_xyz_hunanml
from diffusion.video_mdm_diffusion import project_points_to_ray_hmlvec
from torch.utils.data import DataLoader
from tqdm import tqdm
from train.train_platforms import (  # required for the eval operation
    ClearmlPlatform,
    NoPlatform,
    TensorboardPlatform,
    WandBPlatform,
    WandBSweepPlatform,
)
from train.training_loop import TrainLoop
from utils import dist_util
from utils.fixseed import fixseed
from utils.model_util import create_model_and_diffusion, load_saved_model
from utils.parser_util import generate_args
from utils.sampler_util import ClassifierFreeSampleModel


def get_weights(t, diffusion):
    weight_x_0 = (
        diffusion.sqrt_alphas_cumprod[t - 1]
        * diffusion.betas[t]
        / (1 - diffusion.alphas_cumprod[t])
    )
    weight_x_t = (
        np.sqrt(1 - diffusion.betas[t])
        * (1 - diffusion.alphas_cumprod[t - 1])
        / (1 - diffusion.alphas_cumprod[t])
    )
    weight_noise = (
        (1 - diffusion.alphas_cumprod[t - 1])
        / (1 - diffusion.alphas_cumprod[t])
        * diffusion.betas[t]
    ) ** 0.5
    return weight_x_0, weight_x_t, weight_noise


def save_samples(samples, model_kwargs, motion_idxs, hml_args, args, data_loader):
    lengths = model_kwargs["y"]["lengths"]
    print(lengths)

    def get_motion_name(length):
        motion_idx = motion_idxs.pop(0)
        motion_name = data_loader.dataset.t2m_dataset.get_name(motion_idx)
        #print(motion_name)
        return motion_name

    samples_xyz = get_xyz_hunanml(
        samples,
        *hml_args,
    ).permute(0, 3, 1, 2).cpu().numpy()
    for sample_xyz, length in zip(
        samples_xyz, lengths
    ):
        #print(sample_xyz.shape)
        #print(length)
        sample_xyz = sample_xyz[:length]
        motion_name = get_motion_name(length)
        np.save(os.path.join(args.save_dir, f"{motion_name}.npy"), sample_xyz)

def main():
    args = generate_args()
    fixseed(args.seed)

    if args.eval_dataset is not None:
        args.dataset = args.eval_dataset

    assert "," not in args.dataset, "Dataset should be a single string"
    print("71", args.dataset)
    model_dir = os.path.dirname(args.model_path)
    model_name = os.path.splitext(os.path.basename(args.model_path))[0]
    args.save_dir =  os.path.join(model_dir, f"{model_name}_lifted_samples_{args.dataset}")
    print("Saving samples to: ", args.save_dir)
    os.makedirs(args.save_dir, exist_ok=True)

    train_platform_type = eval(args.train_platform_type)
    train_platform = train_platform_type(args.save_dir)
    train_platform.report_args(args, name="Args")

    args_path = os.path.join(args.save_dir, "args.json")
    with open(args_path, "w") as fw:
        json.dump(vars(args), fw, indent=4, sort_keys=True)

    dist_util.setup_dist(args.device)

    print("creating data loader...")
    print("87", args.dataset)
    dataset = get_dataset(
        args.dataset, 301, "test", hml_mode="train", device=dist_util.dev()
    )
    collate = get_collate_fn(args.dataset, "train")

    data_loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        drop_last=False,
        collate_fn=collate,
    )

    print("creating model and diffusion...")
    print("103", args.dataset)
    model, diffusion = create_model_and_diffusion(args, data_loader)
    model.eval()
    model.to(dist_util.dev())
    model.rot2xyz.smpl_model.eval()
    load_saved_model(model, args.model_path, use_avg=not args.dont_use_ema)
    if args.guidance_param != 1:
        model = ClassifierFreeSampleModel(model)   # wrapping model with the classifier-free sampler
    print(args.dataset, args.save_dir)

    dataset_name = args.dataset if args.dataset != "humanml" else "HumanML3D"
    args.dataset_path = os.path.join("dataset", dataset_name)

    motion_idxs = list(range(len(data_loader.dataset)))

    print("Sampling...")
    with torch.no_grad():
        for motion_sample, model_kwargs in tqdm(data_loader):
            # add CFG scale to batch
            if args.guidance_param != 1:
                model_kwargs['y']['scale'] = torch.ones(motion_sample.shape[0], device=dist_util.dev()) * args.guidance_param
            #print(model_kwargs.keys())
            #print(model_kwargs["y"].keys())
            motion_sample.to(dist_util.dev())
            x_t = torch.randn_like(motion_sample).to(dist_util.dev())
            extra = model_kwargs.pop("extra")
            dataset_torch_std = torch.tensor(data_loader.dataset.t2m_dataset.std).to(x_t.device).float()
            dataset_torch_mean = torch.tensor(data_loader.dataset.t2m_dataset.mean).to(x_t.device).float()
            hml_args = (dataset_torch_mean, dataset_torch_std)
            for t in range(diffusion.num_timesteps - 1, 0, -1):  # For multistep t* training
                batch_t = torch.tensor([t]).to(dist_util.dev()).repeat(x_t.shape[0])
                model_output = model(x_t, diffusion._scale_timesteps(batch_t), **model_kwargs)
                x_0 = project_points_to_ray_hmlvec(
                    hmlvec=model_output,
                    diffusion_extra=extra,
                    hml_args=hml_args,
                )

                new_noise = torch.randn_like(x_t)
                weight_x_0, weight_x_t, weight_noise = get_weights(t, diffusion)
                x_t = weight_x_0 * x_0 + weight_x_t * x_t + weight_noise * new_noise

            save_samples(x_0, model_kwargs, motion_idxs, hml_args, args, data_loader)
    train_platform.close()

    # Call evaluation
    results_save_path = os.path.join(args.save_dir, "results.csv")
    command = [
        "python", "-m", "eval.eval_pose_estimation",
        args.save_dir,
        "./perm/mdm/model000750000_lifted_samples_fit3d/",
        "../fit3d_train_flat/wham_new_joints",
        "./perm/myft_fit3d_cvpr/model000620164_lifted_samples_fit3d",
        "--out", results_save_path
    ]

    subprocess.run(command, shell=False)


if __name__ == "__main__":
    main()
