# This code is based on https://github.com/openai/guided-diffusion
"""
Train a diffusion model on images.
"""

import os
import json
from utils.fixseed import fixseed
from utils.parser_util import train_args
from utils import dist_util
from train.training_loop import TrainLoop
from data_loaders.get_data import get_dataset_loader, get_datasets_loader, get_hml_mode
from utils.model_util import create_model_and_diffusion
from model_2d.load_2d_model import create_model_2d_and_diffusion_from_path
from train.train_platforms import WandBPlatform, WandBSweepPlatform, ClearmlPlatform, TensorboardPlatform, NoPlatform  # required for the eval operation


def main(get_training_args=train_args):
    args = get_training_args()
    fixseed(args.seed)

    if args.save_dir is None:
        raise FileNotFoundError('save_dir was not specified.')
    elif os.path.exists(args.save_dir) and not args.overwrite:
        raise FileExistsError('save_dir [{}] already exists.'.format(args.save_dir))
    elif not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    train_platform_type = eval(args.train_platform_type)
    train_platform = train_platform_type(args.save_dir)
    train_platform.report_args(args, name='Args')

    args_path = os.path.join(args.save_dir, 'args.json')
    with open(args_path, 'w') as fw:
        json.dump(vars(args), fw, indent=4, sort_keys=True)

    dist_util.setup_dist(args.device)

    print("creating data loader...")
    datasets = get_datasets_loader(name=args.dataset, batch_size=args.batch_size, num_frames=args.num_frames,
                                   split=args.train_split, device=dist_util.dev(), num_workers=args.num_workers,
                                   hml_mode=get_hml_mode(args))
    other_data = None
    assert len(datasets) > 0, "No datasets where loaded"
    data = datasets[0]
    if len(datasets) > 1:
        assert len(datasets) <= 2, "Got a list of datasets, but the implimentation supports only 1 or 2 right now."
        other_data = datasets[1]

    print("creating model and diffusion...")
    model, diffusion = create_model_and_diffusion(args, data)
    model.to(dist_util.dev())
    model.rot2xyz.smpl_model.eval()

    model_2d = None
    if args.model_2d_path is not None:
        print("Loading SDS 2d model...")
        model_2d = create_model_2d_and_diffusion_from_path(args.model_2d_path, distilation_solve_ode=args.distilation_solve_ode,
                                                                                distilation_use_gt_camera=args.distilation_use_gt_camera,
                                                                                cfg_for_distilation=args.cfg_for_distilation)
        model_2d.to(dist_util.dev())
        model_2d.eval()

    print('Total params: %.2fM' % (sum(p.numel() for p in model.parameters_wo_clip()) / 1000000.0))
    print("Training...")
    TrainLoop(args, train_platform, model, diffusion, data, other_data, model_2d).run_loop()
    train_platform.close()

if __name__ == "__main__":
    main()
