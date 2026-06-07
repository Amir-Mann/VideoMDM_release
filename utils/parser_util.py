from argparse import ArgumentParser
import argparse
import os
import json
import numpy as np
from datetime import datetime

def parse_and_load_from_model(parser):
    # args according to the loaded model
    # do not try to specify them from cmd line since they will be overwritten
    add_data_options(parser)
    add_model_options(parser)
    add_diffusion_options(parser)
    args = parser.parse_args()
    args_to_overwrite = []
    for group_name in ['dataset', 'model', 'diffusion']:
        args_to_overwrite += get_args_per_group_name(parser, args, group_name)

    # load args from model
    model_path = args.model_path# get_model_path_from_args()
    args_path = os.path.join(os.path.dirname(model_path), 'args.json')
    assert os.path.exists(args_path), f'Arguments json file at {args_path} was not found!'
    if args.model_path != '':  # if not using external results file
        args = load_args_from_model(args, args_to_overwrite)

    if args.cond_mask_prob == 0:
        args.guidance_param = 1
    
    return args

def load_args_from_model(args, args_to_overwrite):
    model_path = args.model_path#get_model_path_from_args()
    args_path = os.path.join(os.path.dirname(model_path), 'args.json')
    assert os.path.exists(args_path), f'Arguments json file was not found! {args_path}'
    with open(args_path, 'r') as fr:
        model_args = json.load(fr)

    for a in args_to_overwrite:
        if a in model_args.keys():
            setattr(args, a, model_args[a])

        elif 'cond_mode' in model_args: # backward compitability
            unconstrained = (model_args['cond_mode'] == 'no_cond')
            setattr(args, 'unconstrained', unconstrained)

        else:
            print('Warning: was not able to load [{}], using default value [{}] instead.'.format(a, args.__dict__[a]))

    if "," in args.dataset:
        print(f"Warning: model_path was trained on multiple datasets {args.dataset}, using first only ", end="")
        args.dataset = args.dataset.split(",")[0]
        print(f"({args.dataset})")

    if args.cond_mask_prob == 0:
        args.guidance_param = 1
    return args

def get_args_per_group_name(parser, args, group_name):
    for group in parser._action_groups:
        if group.title == group_name:
            group_dict = {a.dest: getattr(args, a.dest, None) for a in group._group_actions}
            return list(argparse.Namespace(**group_dict).__dict__.keys())
    return ValueError('group_name was not found.')

def get_model_path_from_args():
    try:
        dummy_parser = ArgumentParser()
        dummy_parser.add_argument('model_path')
        dummy_args, _ = dummy_parser.parse_known_args()
        return dummy_args.model_path
    except:
        raise ValueError('model_path argument must be specified.')


def add_base_options(parser):
    group = parser.add_argument_group('base')
    group.add_argument("--cuda", default=True, type=bool, help="Use cuda device, otherwise use CPU.")
    group.add_argument("--device", default=0, type=int, help="Device id to use.")
    group.add_argument("--seed", default=10, type=int, help="For fixing random seed.")
    group.add_argument("--batch_size", default=64, type=int, help="Batch size during training.")
    group.add_argument("--train_platform_type", default='WandBPlatform', choices=['NoPlatform', 'ClearmlPlatform', 'TensorboardPlatform', 'WandBPlatform', 'WandBSweepPlatform'], type=str,
                       help="Choose platform to log results. NoPlatform means no logging.")
    group.add_argument("--external_mode", default=False, type=bool, help="For backward cometability, do not change or delete.")


def add_diffusion_options(parser):
    group = parser.add_argument_group('diffusion')
    group.add_argument("--noise_schedule", default='cosine', choices=['linear', 'cosine'] + [f'cosine_tau_{tau}' for tau in [0.25, 0.5, 0.75, 1.25, 1.5, 2.0, 3.0, 1, 2, 3, 1.0]], type=str,
                       help="Noise schedule type")
    group.add_argument("--diffusion_steps", default=50, type=int,
                       help="Number of diffusion steps (denoted T in the paper)")
    group.add_argument("--sigma_small", default=True, type=bool, help="Use smaller sigma values.")
    group.add_argument("--video_mdm", action='store_true',
                       help="Train based on 2d projection of the data instead of the original formulation.")
    group.add_argument("--cam_sample_distance_factor", default=2.5, type=float,
                       help="Distance factor for the camera sampling.")
    group.add_argument("--min_cam_sample_elevation_angle", default=-np.pi/8, type=float,
                       help="Minimum elevation angle for the camera.")
    group.add_argument("--max_cam_sample_elevation_angle", default=np.pi/24, type=float,
                       help="Maximum elevation angle for the camera.")
    group.add_argument("--num_cameras_for_distilation", default=1, type=int,
                       help="Number of cameras for the distilation model.")


def add_model_options(parser):
    group = parser.add_argument_group('model')
    group.add_argument("--arch", default='trans_enc',
                       choices=['trans_enc', 'trans_dec', 'gru'], type=str,
                       help="Architecture types as reported in the paper.")
    group.add_argument("--text_encoder_type", default='clip',
                       choices=['clip', 'bert'], type=str, help="Text encoder type.")
    group.add_argument("--emb_trans_dec", action='store_true',
                       help="For trans_dec architecture only, if true, will inject condition as a class token"
                            " (in addition to cross-attention).")
    group.add_argument("--layers", default=8, type=int,
                       help="Number of layers.")
    group.add_argument("--latent_dim", default=512, type=int,
                       help="Transformer/GRU width.")
    group.add_argument("--cond_mask_prob", default=.1, type=float,
                       help="The probability of masking the condition during training."
                            " For classifier-free guidance learning.")
    group.add_argument("--lambda_rcxyz", default=0.0, type=float, help="Joint positions loss.")
    group.add_argument("--lambda_vel", default=0.0, type=float, help="Joint velocity loss.")
    group.add_argument("--lambda_fc", default=0.0, type=float, help="Foot contact loss.")
    group.add_argument("--lambda_cam", default=7.0, type=float, help="Camera joint loss.")
    group.add_argument("--lambda_cam_vel", default=50.0, type=float, help="Camera velocity loss.")
    group.add_argument("--lambda_cam_complement", default=0.0, type=float, help="Rotation and other factors losses.")
    group.add_argument("--lambda_consistancy", default=0.0, type=float,
                       help="Representation-consistency (L_repr) loss weight between the HumanML3D vector and the 3D motion. "
                            "0 disables it; any non-zero value enables it.")
    group.add_argument("--model_2d_path", default=None, type=str, help="Path to the 2d model checkpoint. None to skip SDS at training.")
    group.add_argument("--lambda_sds", default=0.0, type=float, help="SDS base loss factor per joint joint loss.")
    group.add_argument("--lambda_sds_vel", default=50.0, type=float, help="Ratio between SDS velocity loss and SDS base loss.")
    group.add_argument("--distilation_solve_ode", action='store_true', help="If True, will solve the ODE for the distilation model.")
    group.add_argument("--distilation_use_gt_camera", action='store_true', help="If True, will use the ground truth camera for the distilation model.")
    group.add_argument("--distilation_branched_denoising", action='store_true', help="If True, will use Denoise both in 3d and in 2d using the different denoisers, distiling their results from 2d->3d.")
    group.add_argument("--cfg_for_distilation", default=0.0, type=float, help="The cfg for the distilation model. 0.0 (default) means no cfg.")
    group.add_argument("--mask_frames", action='store_true', help="If true, will fix Rotem's bug and mask invalid frames.")
    group.add_argument("--unconstrained", action='store_true',
                       help="Model is trained unconditionally. That is, it is constrained by neither text nor action. "
                            "Currently tested on HumanAct12 only.")
    group.add_argument("--pos_embed_max_len", default=5000, type=int,
                       help="Pose embedding max length.")
    group.add_argument("--dont_use_ema", action='store_true',
                    help="If the flag is set, will not use EMA model averaging.")


def add_data_options(parser):
    group = parser.add_argument_group('dataset')
    group.add_argument("--dataset", default='humanml', choices=['humanml', 'kit', 'humanact12', 'uestc', 'egoexo', 'video_mdm_synthetic', 'video_mdm_synthetic_mvlift', 'egoexo,humanml', 'nba', 'fit3d', 'fit3d,humanml', 'fit3d_mvlift'], type=str,
                       help="Dataset name (choose from list).")
    group.add_argument("--data_dir", default="", type=str,
                       help="If empty, will use defaults according to the specified dataset.")
    group.add_argument("--pnp_to_find_cameras", action='store_true',
                       help="If True, will use PnP to find the cameras instead of using the stored on disk cameras.")

def add_training_options(parser):
    group = parser.add_argument_group('training')
    group.add_argument("--save_dir", type=str, default=f"saves/unamed_run_{datetime.now().strftime('%Y.%m.%d_%H.%M.%S')}",
                       help="Path to save checkpoints and results.")
    group.add_argument("--overwrite", action='store_true',
                       help="If True, will enable to use an already existing save_dir.")
    group.add_argument("--lr", default=1e-4, type=float, help="Learning rate.")
    group.add_argument("--weight_decay", default=0.0, type=float, help="Optimizer weight decay.")
    group.add_argument("--lr_anneal_steps", default=0, type=int, help="Number of learning rate anneal steps.")
    group.add_argument("--train_split", default='train', type=str,
                       help="Which split to train on.")
    group.add_argument("--eval_batch_size", default=32, type=int,
                       help="Batch size during evaluation loop. Do not change this unless you know what you are doing. "
                            "T2m precision calculation is based on fixed batch size 32.")
    group.add_argument("--eval_split", default='test', choices=['train', 'val', 'test'], type=str,
                       help="Which split to evaluate on during training.")
    group.add_argument("--eval_during_training", action='store_true',
                       help="If True, will run evaluation during training.")
    group.add_argument("--evaluator_dir_path", type=str, default="../MAS/save/egoexo_uncentered/evaluator_l1_wtKL_1/",
                       help="Path to the evaluator directory. It must contain the args.json for the architecture parameters and checkpoint_XXX.pth file(s).")
    group.add_argument("--eval_rep_times", default=3, type=int,
                       help="Number of repetitions for evaluation loop during training.")
    group.add_argument("--eval_num_samples", default=1_000, type=int,
                       help="If -1, will use all samples in the specified split.")
    group.add_argument("--gen_guidance_param", default=2.5, type=float,
                       help="For classifier-free sampling - specifies the s parameter, as defined in the paper.")
    group.add_argument("--avg_model_beta", default=0.9999, type=float, help="Average model beta (for EMA).")
    group.add_argument("--adam_beta2", default=0.999, type=float, help="Adam beta2.")
    group.add_argument("--log_interval", default=1_000, type=int,
                       help="Log losses each N steps")
    group.add_argument("--save_interval", default=50_000, type=int,
                       help="Save checkpoints and run evaluation each N steps")
    group.add_argument("--num_steps", default=600_000, type=int,
                       help="Training will stop after the specified number of steps.")
    group.add_argument("--num_workers", default=2, type=int,
                       help="Number of workers for the data loader.")
    group.add_argument("--num_frames", default=60, type=int,
                       help="Limit for the maximal number of frames. In HumanML3D and KIT this field is ignored.")
    group.add_argument("--resume_checkpoint", default="", type=str,
                       help="If not empty, will start from the specified checkpoint (path to model###.pt file).")
    group.add_argument("--fine_tunning", action='store_true',
                       help="If True, will not load clip and the optimizer state from the checkpoint, allowing to finetune a model checkpoint.")
    group.add_argument("--gen_during_training", action='store_true', help="If True, will generate samples from the model during training.")
    group.add_argument("--t_star_method", default=None, type=str,
                      choices=['none', 'curriculum'] + [f'multistep_b{b}_e{e}_d{d}' for b in range(2) for e, d in [(0, 0), (1, 0), (1, 1)]], 
                      help="The method for the t* sampling. None means no t* sampling. For multistep, b is use buckets, e is calculate loss every step, d is detach x_t each prediction. Recommended: multistep_b1_e1_d1.")
    group.add_argument("--t_star_arg", default=4, type=int, help="The argument for the t* sampling. None means no t* sampling.")
    group.add_argument("--t_star", default=12, type=int, help="The t* value for the t* sampling. None means no t* sampling.")
    group.add_argument("--clip_grad_max_norm", default=1.0, type=float, help="The the maximum grad norm for clipping the gradients.")
    group.add_argument("--gt_supervision", action='store_true', help="If True, will use the ground truth 3d motion for the supervision.")
    group.add_argument("--gt_xyz_supervision", action='store_true', help="If True, will use the ground truth 3d motion for the supervision.")
    group.add_argument("--gt_teacher", action='store_true', help="If True, will use the ground truth 3d motion for the teacher.")
    group.add_argument("--mask_from_score_type", default='square', choices=['square', 'threshold4', 'threshold6', 'linear'], type=str, help="The type of mask from the score.")
    group.add_argument("--multistep_use_guidance", action='store_true', help="If True, will use 2d motion and camera as guidance for the multistep sampling.")
    group.add_argument("--quicker_batch_chance", default=0.0, type=float, help="The chance to shorten all motions to 100 frames. 0.0 means no quicker batching.")
    group.add_argument("--no_distance_weighting", action='store_true', help="If True, will not weight the loss by distance of joints from camera.")
    group.add_argument("--lambda_floor_distance", default=0.0, type=float, help="The lambda for the floor distance loss. Applicable only for video_mdm=True.") 
    group.add_argument("--sanity_check_folder", default=None, type=str, help="If not None, after 1 epoch, will use the spesified folder to save state_dict if empty, or to compare against if exists.")
    group.add_argument("--train_camera_azimuth_corruption", default=0, type=float, help="The maximal angle of azimuth corruption (degrees) to apply to the cameras during training.")


def add_sampling_options(parser):
    group = parser.add_argument_group('sampling')
    group.add_argument("--model_path", required=True, type=str,
                       help="Path to model####.pt file to be sampled.")
    group.add_argument("--output_dir", default='', type=str,
                       help="Path to results dir (auto created by the script). "
                            "If empty, will create dir in parallel to checkpoint.")
    group.add_argument("--num_samples", default=6, type=int,
                       help="Maximal number of prompts to sample, "
                            "if loading dataset from file, this field will be ignored.")
    group.add_argument("--num_repetitions", default=3, type=int,
                       help="Number of repetitions, per sample (text prompt/action)")
    group.add_argument("--guidance_param", default=2.5, type=float,
                       help="For classifier-free sampling - specifies the s parameter, as defined in the paper.")


def add_generate_options(parser):
    group = parser.add_argument_group('generate')
    group.add_argument("--motion_length", default=6.0, type=float,
                       help="The length of the sampled motion [in seconds]. "
                            "Maximum is 9.8 for HumanML3D (text-to-motion), and 2.0 for HumanAct12 (action-to-motion)")
    group.add_argument("--input_text", default='', type=str,
                       help="Path to a text file lists text prompts to be synthesized. If empty, will take text prompts from dataset.")
    group.add_argument("--action_file", default='', type=str,
                       help="Path to a text file that lists names of actions to be synthesized. Names must be a subset of dataset/uestc/info/action_classes.txt if sampling from uestc, "
                            "or a subset of [warm_up,walk,run,jump,drink,lift_dumbbell,sit,eat,turn steering wheel,phone,boxing,throw] if sampling from humanact12. "
                            "If no file is specified, will take action names from dataset.")
    group.add_argument("--text_prompt", default='', type=str,
                       help="A text prompt to be generated. If empty, will take text prompts from dataset.")
    group.add_argument("--action_name", default='', type=str,
                       help="An action name to be generated. If empty, will take text prompts from dataset.")
    group.add_argument("--eval_dataset", default=None, type=str, 
                       help="Which dataset to evaluate on, default is None, meaning use the dataset from the model.")


def add_edit_options(parser):
    group = parser.add_argument_group('edit')
    group.add_argument("--edit_mode", default='in_between', choices=['in_between', 'upper_body'], type=str,
                       help="Defines which parts of the input motion will be edited.\n"
                            "(1) in_between - suffix and prefix motion taken from input motion, "
                            "middle motion is generated.\n"
                            "(2) upper_body - lower body joints taken from input motion, "
                            "upper body is generated.")
    group.add_argument("--text_condition", default='', type=str,
                       help="Editing will be conditioned on this text prompt. "
                            "If empty, will perform unconditioned editing.")
    group.add_argument("--prefix_end", default=0.25, type=float,
                       help="For in_between editing - Defines the end of input prefix (ratio from all frames).")
    group.add_argument("--suffix_start", default=0.75, type=float,
                       help="For in_between editing - Defines the start of input suffix (ratio from all frames).")


def add_evaluation_options(parser):
    group = parser.add_argument_group('eval')
    group.add_argument("--model_path", required=True, type=str,
                       help="Path to model####.pt file to be sampled.")
    group.add_argument("--evaluator_dir_path", default="../MAS/save/egoexo_uncentered/evaluator_l1_wtKL_1/", type=str,
                       help="Path to the evaluator directory.\n"
                            "It must contain the args.json for the architecture parameters and checkpoint_XXX.pth file(s).")
    group.add_argument("--eval_mode", default='partial', choices=['wo_mm', 'mm_short', 'debug', 'partial', 'preview', 'full'], type=str,
                       help="wo_mm (t2m only) - 20 repetitions without multi-modality metric; "
                            "mm_short (t2m only) - 5 repetitions with multi-modality metric; "
                            "debug (video_mdm, a2m only) - short run, less accurate results."
                            "partial (video_mdm) - 20 repetitions, 1000 samples."
                            "preview (video_mdm) - 5 repetitions, 6144 samples."
                            "full (video_mdm, a2m only) - 20 repetitions, 6144 samples.")
    group.add_argument("--guidance_param", default=2.5, type=float,
                       help="For classifier-free sampling - specifies the s parameter, as defined in the paper.")
    group.add_argument("--split", default='val', choices=['val', 'test'], type=str,
                       help="Which split to evaluate on, defualt is val.")
    group.add_argument("--evaluate_training_data", action='store_true',
                       help="If True, will evaluate the training data.")
    group.add_argument("--skip_generation", action='store_true',
                       help="If True, will evaluate the training data.")
    group.add_argument("--mas_data_dir", default=None, type=str, 
                       help="Path to the directory containing the MAS data. None to skip MAS evaluation.")
    group.add_argument("--eval_dataset", default=None, type=str, 
                       help="Which dataset to evaluate on, default is None, meaning use the dataset from the model.")
    group.add_argument("--reconstruct_hmlvec_from_ric", action='store_true',
                       help="If True, will reconstruct the HumanML3D vector from the RIC part of the sampled motion during evaluation.")

def get_cond_mode(args):
    if args.unconstrained:
        cond_mode = 'no_cond'
    elif args.dataset in ['kit', 'humanml', 'egoexo', 'video_mdm_synthetic', 'video_mdm_synthetic_mvlift', 'fit3d', 'fit3d_mvlift', 'nba'] or 'humanml' in args.dataset:
        cond_mode = 'text'
    else:
        cond_mode = 'action'
    return cond_mode


def train_args():
    parser = ArgumentParser()
    add_base_options(parser)
    add_data_options(parser)
    add_model_options(parser)
    add_diffusion_options(parser)
    add_training_options(parser)
    return parser.parse_args()


def generate_args():
    parser = ArgumentParser()
    # args specified by the user: (all other will be loaded from the model)
    add_base_options(parser)
    add_sampling_options(parser)
    add_generate_options(parser)
    args = parse_and_load_from_model(parser)
    cond_mode = get_cond_mode(args)

    if (args.input_text or args.text_prompt) and cond_mode != 'text':
        raise Exception('Arguments input_text and text_prompt should not be used for an action condition. Please use action_file or action_name.')
    elif (args.action_file or args.action_name) and cond_mode != 'action':
        raise Exception('Arguments action_file and action_name should not be used for a text condition. Please use input_text or text_prompt.')

    return args


def edit_args():
    parser = ArgumentParser()
    # args specified by the user: (all other will be loaded from the model)
    add_base_options(parser)
    add_sampling_options(parser)
    add_edit_options(parser)
    return parse_and_load_from_model(parser)


def evaluation_parser():
    parser = ArgumentParser()
    # args specified by the user: (all other will be loaded from the model)
    add_base_options(parser)
    add_evaluation_options(parser)
    return parse_and_load_from_model(parser)