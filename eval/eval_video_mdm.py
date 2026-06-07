import os
import numpy as np
import torch
from utils.parser_util import evaluation_parser
from utils.fixseed import fixseed
from datetime import datetime
from data_loaders.humanml.motion_loaders.model_motion_loaders import get_mdm_loader  # get_motion_loader
from eval.VideoMDM.evaluator import Evaluator
from eval.VideoMDM.mas.eval_mas_dataset import get_mas_loader
from collections import OrderedDict
from utils.model_util import create_model_and_diffusion, load_saved_model
from data_loaders.humanml.utils.metrics import (
    calculate_frechet_distance,
    calculate_activation_statistics,
    calculate_precision,
    calculate_recall,
    calculate_diversity,
)

from data_loaders.humanml.scripts.motion_process import get_xyz_hunanml

from diffusion import logger
from utils import dist_util
from data_loaders.get_data import get_dataset_loader
from data_loaders.tensors import t2m_collate as tensors_t2m_collate
from utils.sampler_util import ClassifierFreeSampleModel
#from video.dataset_info import dataset_torch_mean, dataset_torch_std

torch.multiprocessing.set_sharing_strategy('file_system')


def quick_log(msg, file):
    print(msg)
    print(msg, file=file, flush=True)

def print_recursive_dict_keys(d, depth=0):
        if isinstance(d, dict):
            for key, value in d.items():
                ending = "\n" if isinstance(value, dict) else " "
                print("  " * depth + str(key), end=ending)
                print_recursive_dict_keys(value, depth + 1)
            print()
        elif isinstance(d, (list, tuple)):
            for i, item in enumerate(d):
                print("  " * depth + str(i), end=" ")
                print_recursive_dict_keys(item, depth + 1)
            print()
        else:
            try:
                print(type(d), d.shape)
            except Exception:
                try:
                    print(type(d), len(d), type(d[0]))
                except Exception:
                    print(type(d))

def extract_activation_dict(eval_wrapper: Evaluator, motion_loaders, num_samples_limit):
    """
    Run motion loaders through the eval wrapper to extract latent embeddings.

    Returns
    -------
    dict[str, np.ndarray] : model_name → motion embeddings (N, D)
    """
    activation_dict = OrderedDict({})
    for motion_loader_name, motion_loader in motion_loaders.items():
        print(f"Extracting embeddings for {motion_loader_name}")
        all_motion_embeddings = []
        
        with torch.no_grad():
            motions_count = 0
            for batch in motion_loader:
                
                motions, kwargs = batch
                motions_count += batch[0].shape[0]
                if num_samples_limit and motions_count > num_samples_limit:
                    break
                
                if motion_loader_name == "ground truth":
                    # For ground truth, we need to use the original 2D motion
                    motion_2d = kwargs["extra"]["motion_2d"]
                    motion_embeddings = eval_wrapper.get_motion_embeddings(
                        motion=motion_2d,
                        m_lens=kwargs["y"]["lengths"]
                    )
                else:
                    if motions.shape[-1] != 3:
                        #std = torch.tensor(dataset_torch_std).to(motions.device).float()
                        #mean = torch.tensor(dataset_torch_mean).to(motions.device).float()
                        std = torch.tensor(motion_loader.dataset.t2m_dataset.std).to(motions.device).float()
                        mean = torch.tensor(motion_loader.dataset.t2m_dataset.mean).to(motions.device).float()
                        
                        #print("motions.shape", motions.shape)
                        #print("lengths", kwargs["y"]["lengths"])
                        motion_3d = get_xyz_hunanml(motions, mean, std) # [bs, njoints, 3, nframes]
                        motion_3d = motion_3d[..., :torch.max(kwargs["y"]["lengths"])]
                        motion_3d = motion_3d.permute(0, 3, 1, 2) # [bs, nframes, njoints, 3]
                    else:
                        motion_3d = motions
                    motion_embeddings = eval_wrapper.get_motion_embeddings(
                        motion=motion_3d,
                        m_lens=kwargs["y"]["lengths"]
                    )
                all_motion_embeddings.append(motion_embeddings.cpu().numpy())
        activation_dict[motion_loader_name] = np.concatenate(all_motion_embeddings, axis=0)
        if not all_motion_embeddings:
            print(f"[WARN] No embeddings found for {motion_loader_name}")
    return activation_dict

def evaluate_fid_precision_and_recall(eval_wrapper, groundtruth_loader, activation_dict, file, num_samples_limit):
    recall_dict = OrderedDict({})
    precision_dict = OrderedDict({})
    fid_dict = OrderedDict({})
    
    gt_motion_embeddings = []
    print('========== Evaluating FID ==========')
    motions_count = 0
    with torch.no_grad():
        for idx, batch in enumerate(groundtruth_loader):
            if num_samples_limit and motions_count + batch[0].shape[0] > num_samples_limit:
                break
            motions_count += batch[0].shape[0]
            motions, kwargs = batch
            motion_2d = kwargs["extra"]["motion_2d"]
            m_lens = kwargs["y"]["lengths"]
            motion_embeddings = eval_wrapper.get_motion_embeddings(
                motion=motion_2d,
                m_lens=m_lens
            )
            gt_motion_embeddings.append(motion_embeddings.cpu().numpy())
    gt_motion_embeddings = np.concatenate(gt_motion_embeddings, axis=0)
    gt_mu, gt_cov = calculate_activation_statistics(gt_motion_embeddings)

    # print(gt_mu)
    for model_name, motion_embeddings in activation_dict.items():
        mu, cov = calculate_activation_statistics(motion_embeddings)
        # print(mu)
        
        recall = calculate_recall(motion_embeddings, gt_motion_embeddings)
        precision = calculate_precision(motion_embeddings, gt_motion_embeddings)
        fid = calculate_frechet_distance(gt_mu, gt_cov, mu, cov)

        quick_log(f'---> [{model_name}] Recall: {recall:.4f}', file=file)
        quick_log(f'---> [{model_name}] Precision: {precision:.4f}', file=file) 
        quick_log(f'---> [{model_name}] FID: {fid:.4f}', file=file)

        recall_dict[model_name] = recall
        precision_dict[model_name] = precision
        fid_dict[model_name] = fid
        
    return recall_dict, precision_dict, fid_dict


def evaluate_diversity(activation_dict, file, diversity_times):
    eval_dict = OrderedDict({})
    print('========== Evaluating Diversity ==========')
    for model_name, motion_embeddings in activation_dict.items():
        diversity = calculate_diversity(motion_embeddings, diversity_times)
        eval_dict[model_name] = diversity
        quick_log(f'---> [{model_name}] Diversity: {diversity:.4f}', file=file)
    return eval_dict


def get_metric_statistics(values, replication_times):
    mean = np.mean(values, axis=0)
    std = np.std(values, axis=0)
    conf_interval = 1.96 * std / np.sqrt(replication_times)
    return mean, conf_interval


def evaluation(eval_wrapper, gt_loader, eval_motion_loaders, log_file, replication_times, diversity_times, mm_num_times, num_samples_limit, run_mm=False, log_function=print):
    with open(log_file, 'w') as f:
        all_metrics = OrderedDict({'Matching Score': OrderedDict({}),
                                   'R_precision': OrderedDict({}),
                                   'Recall': OrderedDict({}),
                                   'Precision': OrderedDict({}),
                                   'FID': OrderedDict({}),
                                   'Diversity': OrderedDict({}),
                                   'MultiModality': OrderedDict({})})
        for replication in range(replication_times):
            motion_loaders = {}
            mm_motion_loaders = {}
            motion_loaders['ground truth'] = gt_loader
            for motion_loader_name, motion_loader_getter in eval_motion_loaders.items():
                motion_loader, mm_motion_loader = motion_loader_getter()
                motion_loaders[motion_loader_name] = motion_loader
                mm_motion_loaders[motion_loader_name] = mm_motion_loader

            quick_log(f'==================== Replication {replication} ====================', file=f)
            #quick_log(f'Time: {datetime.now()}', file=f)
            #mat_score_dict, R_precision_dict, acti_dict = evaluate_matching_score(eval_wrapper, motion_loaders, f)
            acti_dict = extract_activation_dict(eval_wrapper, motion_loaders, num_samples_limit)
            quick_log(f'Time: {datetime.now()}', file=f)
            recall_dict, precision_dict, fid_score_dict = evaluate_fid_precision_and_recall(eval_wrapper, gt_loader, acti_dict, f, num_samples_limit)

            quick_log(f'Time: {datetime.now()}', file=f)
            div_score_dict = evaluate_diversity(acti_dict, f, diversity_times)

            if run_mm:
                quick_log(f'Time: {datetime.now()}', file=f)
                mm_score_dict = evaluate_multimodality(eval_wrapper, mm_motion_loaders, f, mm_num_times)

            quick_log(f'!!! DONE !!!', file=f)

            for key, item in recall_dict.items():
                if key not in all_metrics['Recall']:
                    all_metrics['Recall'][key] = [item]
                else:
                    all_metrics['Recall'][key] += [item]

            for key, item in precision_dict.items():
                if key not in all_metrics['Precision']:
                    all_metrics['Precision'][key] = [item]
                else:
                    all_metrics['Precision'][key] += [item]
            
            for key, item in fid_score_dict.items():
                if key not in all_metrics['FID']:
                    all_metrics['FID'][key] = [item]
                else:
                    all_metrics['FID'][key] += [item]
            

            for key, item in div_score_dict.items():
                if key not in all_metrics['Diversity']:
                    all_metrics['Diversity'][key] = [item]
                else:
                    all_metrics['Diversity'][key] += [item]
            if run_mm:
                for key, item in mm_score_dict.items():
                    if key not in all_metrics['MultiModality']:
                        all_metrics['MultiModality'][key] = [item]
                    else:
                        all_metrics['MultiModality'][key] += [item]


        # print(all_metrics['Diversity'])
        gt_fid = None
        test_fid = None
        mean_dict = {}
        to_report_strings = []
        for metric_name, metric_dict in all_metrics.items():
            quick_log('========== %s Summary ==========' % metric_name, file=f)
            for model_name, values in metric_dict.items():
                # print(metric_name, model_name)
                mean, conf_interval = get_metric_statistics(np.array(values), replication_times)
                mean_dict[metric_name + '_' + model_name] = mean
                # print(mean, mean.dtype)
                if isinstance(mean, np.float64) or isinstance(mean, np.float32):
                    quick_log(f'---> [{model_name}] Mean: {mean:.4f} CInterval: {conf_interval:.4f}', file=f)
                    if metric_name == "FID":
                        if model_name == "ground truth":
                            gt_fid = mean
                        if model_name in "test||vald":
                            test_fid = mean
                    to_report_strings.append(f'{metric_name.rjust(10)} {model_name[:4]} : {mean:05.2f}±{conf_interval:.2f}')
                elif isinstance(mean, np.ndarray):
                    line = f'---> [{model_name}]'
                    for i in range(len(mean)):
                        line += '(top %d) Mean: %.4f CInt: %.4f;' % (i+1, mean[i], conf_interval[i])
                    quick_log(line, file=f)
        try:
            strings = '\n'.join(sorted(to_report_strings, key= lambda x: x.split(':')[0][::-1]))
            log_function(f"\n     For report:\n{strings}\n")
            log_function(f"\n     FID:\nGT  : {gt_fid}\nTest:{test_fid}\n")
        except Exception as e:
            print(e)
        return mean_dict


if __name__ == '__main__':
    import builtins

    _original_print = print

    def guarded_print(*args, **kwargs):
        from io import StringIO
        import sys

        buf = StringIO()
        kwargs_copy = kwargs.copy()
        kwargs_copy["file"] = buf
        _original_print(*args, **kwargs_copy)

        output = buf.getvalue()
        lines = output.count("\n")

        if lines > 10:
            raise RuntimeError(f"Too much printed output: {lines} lines")
        _original_print(*args, **kwargs)

    # Replace built-in print
    builtins.print = guarded_print
    args = evaluation_parser()
    fixseed(args.seed)
    args.batch_size = 32 # This must be 32! Don't change it! otherwise it will cause a bug in R precision calc!

    name = os.path.basename(os.path.dirname(args.model_path))
    niter = os.path.basename(args.model_path).replace('model', '').replace('.pt', '')
    dataset = args.eval_dataset if args.eval_dataset else args.dataset
    log_file = os.path.join(os.path.dirname(args.model_path), 'eval_{}_{}_{}'.format(dataset, name, niter))
    if args.evaluate_training_data:
        log_file += '_traindata'
    if args.mas_data_dir is not None:
        log_file += '_masdata'
    if args.guidance_param != 1.:
        log_file += f'_gscale{args.guidance_param}'
    log_file += f'_{args.eval_mode}'


    print(f'Will save to log file [{log_file}]')

    print(f'Eval mode [{args.eval_mode}]')
    if args.eval_mode == 'debug':
        num_samples_limit = 512  # None means no limit (eval over all dataset)
        diversity_times = 300
        replication_times = 5 
    elif args.eval_mode == 'partial':
        num_samples_limit = 3072
        diversity_times = 300
        replication_times = 20
    elif args.eval_mode == 'preview':
        num_samples_limit = 6144
        diversity_times = 300
        replication_times = 5
    elif args.eval_mode == 'full':
        num_samples_limit = 6144
        diversity_times = 300
        replication_times = 20
    elif args.eval_mode == 'mm_short':
        raise NotImplementedError('mm short is not implemented yet!')
    else:
        raise ValueError()


    dist_util.setup_dist(args.device)
    logger.configure()

    logger.log(f"creating data loader, using dataset {dataset}...")

    gt_loader = get_dataset_loader(name=dataset, batch_size=args.batch_size, num_frames=None, split=args.split, hml_mode='2d')
    gen_loader = get_dataset_loader(name=dataset, batch_size=args.batch_size, num_frames=None, split=args.split, hml_mode='2d')
    num_actions = gen_loader.dataset.num_actions

    eval_motion_loaders = {
    }
    if not args.skip_generation:
        
        logger.log("Creating model and diffusion...")
        model, diffusion = create_model_and_diffusion(args, gen_loader)

        logger.log(f"Loading checkpoints from [{args.model_path}]...")
        load_saved_model(model, args.model_path, use_avg=not args.dont_use_ema)

        if args.guidance_param != 1:
            model = ClassifierFreeSampleModel(model)   # wrapping model with the classifier-free sampler
        model.to(dist_util.dev())
        model.eval()  # disable random masking
        name = args.split if args.split != "val" else "vald"
        eval_motion_loaders[name] = lambda: get_mdm_loader(
            model, diffusion, args.batch_size,
            gen_loader, 0, 0, gt_loader.dataset.opt.max_motion_length, num_samples_limit, args.guidance_param,
            collate_fn_to_use=tensors_t2m_collate
        )
    
    if args.evaluate_training_data:
        eval_motion_loaders['train'] = lambda: (get_dataset_loader(name=dataset, batch_size=args.batch_size, num_frames=None, split='train', hml_mode='train'), None)
    
    if args.mas_data_dir is not None:
        eval_motion_loaders['mas'] = lambda: (get_mas_loader(args), None)

    eval_wrapper = Evaluator(args.dataset, dist_util.dev(), args.evaluator_dir_path)
    evaluation(eval_wrapper, gt_loader, eval_motion_loaders, log_file, replication_times, diversity_times, 0, num_samples_limit, run_mm=False)
