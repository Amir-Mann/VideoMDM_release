import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from utils.parser_util import evaluation_parser
from utils.fixseed import fixseed
from datetime import datetime
from data_loaders.humanml.motion_loaders.model_motion_loaders import get_mdm_loader
from data_loaders.humanml.networks.evaluator_wrapper import EvaluatorMDMWrapper
from data_loaders.humanml.utils.metrics import (
    calculate_frechet_distance,
    calculate_activation_statistics,
    calculate_diversity,
    calculate_multimodality,
)
from eval.unconstrained.metrics.kid import polynomial_mmd_averages
from diffusion.torch_process_motions import process_file_torch, get_skeleton_kwargs_from_dataset
from collections import OrderedDict
from utils.model_util import create_model_and_diffusion, load_saved_model
from diffusion import logger
from utils import dist_util
from data_loaders.get_data import get_dataset, get_collate_fn
from utils.sampler_util import ClassifierFreeSampleModel

torch.multiprocessing.set_sharing_strategy('file_system')

_FIT3D_DATASETS = {'fit3d', 'fit3d_mvlift'}
_GT_DIR = 'dataset/fit3d/smpl_gt_poses'
_SPLIT_FILE = 'dataset/fit3d/{split}.txt'
_MODEL_MEAN_PATH = 'dataset/fit3d/Mean.npy'    # fit3d training norm — used to denorm generated samples
_MODEL_STD_PATH = 'dataset/fit3d/Std.npy'
_EVAL_MEAN_PATH = 'dataset/video_t2m_mean.npy'  # t2m evaluator norm — used for all inputs to the evaluator
_EVAL_STD_PATH = 'dataset/video_t2m_std.npy'


# ---------------------------------------------------------------------------
# GT dataset
# ---------------------------------------------------------------------------

class Fit3DGTDataset(Dataset):
    def __init__(self, split, mean, std):
        split_path = _SPLIT_FILE.format(split=split)
        with open(split_path, 'r') as f:
            names = [l.strip() for l in f if l.strip()]

        self.mean = mean
        self.std = std
        self.skel_kwargs = get_skeleton_kwargs_from_dataset('fit3d')

        self.motions = []
        self.lengths = []
        for name in names:
            npy_path = os.path.join(_GT_DIR, name + '.npy')
            if not os.path.exists(npy_path):
                print(f'[GT] Warning: missing {npy_path}, skipping')
                continue
            joints = np.load(npy_path)              # (T, 22, 3) metres
            if joints.ndim != 3 or joints.shape[2] != 3 or joints.shape[0] < 9:
                print(f'[GT] Warning: bad shape {joints.shape} for {name}, skipping')
                continue
            try:
                t = torch.from_numpy(joints.astype(np.float32)).unsqueeze(0)  # (1,T,22,3)
                feat, _, _, _ = process_file_torch(t, **self.skel_kwargs)     # (1,T-1,263)
                feat = feat.squeeze(0).numpy()                                 # (T-1,263)
                feat = (feat - self.mean) / self.std
                self.motions.append(feat)
                self.lengths.append(feat.shape[0])
            except Exception as e:
                print(f'[GT] Warning: process_file_torch failed for {name}: {e}, skipping')

        print(f'[GT] Loaded {len(self.motions)} / {len(names)} clips from {split_path}')

    def __len__(self):
        return len(self.motions)

    def __getitem__(self, idx):
        return self.motions[idx], self.lengths[idx]


def fit3d_gt_collate(batch):
    motions, lengths = zip(*batch)
    max_len = max(lengths)
    padded = np.zeros((len(motions), max_len, 263), dtype=np.float32)
    for i, m in enumerate(motions):
        padded[i, :m.shape[0]] = m
    return torch.from_numpy(padded), torch.tensor(lengths, dtype=torch.long)


# ---------------------------------------------------------------------------
# Lifted predictions dataset (WHAM / mvlift pre-computed)
# ---------------------------------------------------------------------------

class LiftedMotionsDataset(Dataset):
    """Pre-computed lifted motions from a directory of per-clip .npy files.

    Auto-detects format:
      (T, J, 3): raw joint positions in metres → process_file_torch → normalize with t2m evaluator stats
      (T, 263): raw HumanML features from process_file → normalize with t2m evaluator stats
    """
    def __init__(self, preds_dir, split, mean, std):
        split_path = _SPLIT_FILE.format(split=split)
        with open(split_path, 'r') as f:
            names = [l.strip() for l in f if l.strip()]

        self.mean = mean
        self.std = std
        self.skel_kwargs = get_skeleton_kwargs_from_dataset('fit3d')

        self.motions = []
        self.lengths = []
        for name in names:
            npy_path = os.path.join(preds_dir, name + '.npy')
            if not os.path.exists(npy_path):
                print(f'[Lifted] Warning: missing {npy_path}, skipping')
                continue
            arr = np.load(npy_path).astype(np.float32)
            try:
                if arr.ndim == 3 and arr.shape[2] == 3:
                    if arr.shape[0] < 9:
                        print(f'[Lifted] Warning: too short ({arr.shape[0]} frames) for {name}, skipping')
                        continue
                    t = torch.from_numpy(arr).unsqueeze(0)
                    feat, _, _, _ = process_file_torch(t, **self.skel_kwargs)
                    feat = feat.squeeze(0).numpy()
                    feat = (feat - self.mean) / self.std
                elif arr.ndim == 2 and arr.shape[1] == 263:
                    feat = (arr - self.mean) / self.std
                else:
                    print(f'[Lifted] Warning: unexpected shape {arr.shape} for {name}, skipping')
                    continue
                self.motions.append(feat)
                self.lengths.append(feat.shape[0])
            except Exception as e:
                print(f'[Lifted] Warning: failed for {name}: {e}, skipping')

        print(f'[Lifted:{os.path.basename(preds_dir.rstrip("/"))}] '
              f'Loaded {len(self.motions)} / {len(names)} clips from {preds_dir}')

    def __len__(self):
        return len(self.motions)

    def __getitem__(self, idx):
        return self.motions[idx], self.lengths[idx]


# ---------------------------------------------------------------------------
# Embedding extraction
# ---------------------------------------------------------------------------

def get_gt_embeddings(eval_wrapper, gt_dataset, batch_size, device):
    loader = DataLoader(gt_dataset, batch_size=batch_size, shuffle=False,
                        collate_fn=fit3d_gt_collate, drop_last=False, num_workers=2)
    all_emb = []
    with torch.no_grad():
        for motions, m_lens in loader:
            emb = eval_wrapper.get_motion_embeddings(motions.to(device), m_lens)
            all_emb.append(emb.cpu().numpy())
    return np.concatenate(all_emb, axis=0)


def extract_gen_embeddings(eval_wrapper, motion_loader, renorm_fn=None):
    all_emb = []
    with torch.no_grad():
        for batch in motion_loader:
            _, _, _, _, motions, m_lens, _ = batch
            if renorm_fn is not None:
                motions = renorm_fn(motions)
            emb = eval_wrapper.get_motion_embeddings(motions, m_lens)
            all_emb.append(emb.cpu().numpy())
    return np.concatenate(all_emb, axis=0)


# ---------------------------------------------------------------------------
# Metric functions
# ---------------------------------------------------------------------------

def evaluate_fid(gt_emb, gen_emb, file):
    print('========== Evaluating FID ==========')
    gt_mu, gt_cov = calculate_activation_statistics(gt_emb)
    gen_mu, gen_cov = calculate_activation_statistics(gen_emb)
    fid = calculate_frechet_distance(gt_mu, gt_cov, gen_mu, gen_cov)
    msg = f'---> FID: {fid:.4f}  (N_gt={len(gt_emb)}, N_gen={len(gen_emb)}, unreliable at this scale — use KID)'
    print(msg)
    print(msg, file=file, flush=True)
    return fid


def evaluate_kid(gt_emb, gen_emb, file, n_subsets=100):
    print('========== Evaluating KID ==========')
    N = min(len(gt_emb), len(gen_emb))
    mmds, _ = polynomial_mmd_averages(
        gt_emb, gen_emb, n_subsets=n_subsets, subset_size=N, replace=True, ret_var=True
    )
    kid_mean, kid_std = float(mmds.mean()), float(mmds.std())
    msg = f'---> KID: {kid_mean:.6f} ± {kid_std:.6f}  (N={N}, bootstrap n_subsets={n_subsets})'
    print(msg)
    print(msg, file=file, flush=True)
    return kid_mean, kid_std


def evaluate_diversity(gen_emb, file, diversity_times):
    print('========== Evaluating Diversity ==========')
    diversity_times = min(diversity_times, len(gen_emb) - 1)
    diversity = calculate_diversity(gen_emb, diversity_times)
    msg = f'---> Diversity: {diversity:.4f}'
    print(msg)
    print(msg, file=file, flush=True)
    return diversity


def evaluate_multimodality(eval_wrapper, mm_motion_loaders, file, mm_num_times, renorm_fn=None):
    eval_dict = OrderedDict({})
    print('========== Evaluating MultiModality ==========')
    for model_name, mm_motion_loader in mm_motion_loaders.items():
        mm_motion_embeddings = []
        with torch.no_grad():
            for idx, batch in enumerate(mm_motion_loader):
                motions, m_lens = batch
                mm_motions = motions[0]  # (mm_num_repeats, T, 263)
                if renorm_fn is not None:
                    mm_motions = renorm_fn(mm_motions)
                motion_emb = eval_wrapper.get_motion_embeddings(mm_motions, m_lens[0])
                mm_motion_embeddings.append(motion_emb.unsqueeze(0))
        if len(mm_motion_embeddings) == 0:
            multimodality = 0
        else:
            mm_motion_embeddings = torch.cat(mm_motion_embeddings, dim=0).cpu().numpy()
            multimodality = calculate_multimodality(mm_motion_embeddings, mm_num_times)
        msg = f'---> [{model_name}] Multimodality: {multimodality:.4f}'
        print(msg)
        print(msg, file=file, flush=True)
        eval_dict[model_name] = multimodality
    return eval_dict


def get_metric_statistics(values, replication_times):
    mean = np.mean(values, axis=0)
    std = np.std(values, axis=0)
    conf_interval = 1.96 * std / np.sqrt(replication_times)
    return mean, conf_interval


# ---------------------------------------------------------------------------
# Main evaluation loop
# ---------------------------------------------------------------------------

def evaluation(eval_wrapper, gt_emb, eval_motion_loaders, log_file, replication_times,
               diversity_times, mm_num_times, run_mm=False, extra_embs=None, renorm_fn=None):
    """Run evaluation loop.

    extra_embs: dict[str, np.ndarray] of pre-computed embeddings (e.g. GT, lifted sources).
    These are fixed — FID/Diversity CI will be ~0 across replications; KID will have tiny
    bootstrap-noise CI. This is expected and correct for deterministic sources.

    renorm_fn: callable(motions_tensor) -> motions_tensor that converts generated motions
    from model (fit3d) normalization to evaluator (t2m) normalization before embedding.
    """
    with open(log_file, 'w') as f:
        print(f'Log file: {log_file}', file=f, flush=True)
        all_metrics = OrderedDict({
            'FID': OrderedDict({}),
            'KID_mean': OrderedDict({}),
            'KID_std': OrderedDict({}),
            'Diversity': OrderedDict({}),
            'MultiModality': OrderedDict({}),
        })

        for replication in range(replication_times):
            motion_loaders = {}
            mm_motion_loaders = {}
            for name, loader_getter in eval_motion_loaders.items():
                motion_loader, mm_motion_loader = loader_getter()
                motion_loaders[name] = motion_loader
                mm_motion_loaders[name] = mm_motion_loader

            print(f'==================== Replication {replication} ====================')
            print(f'==================== Replication {replication} ====================', file=f, flush=True)
            print(f'Time: {datetime.now()}', file=f, flush=True)

            for model_name, motion_loader in motion_loaders.items():
                print(f'--- [{model_name}] ---')
                print(f'--- [{model_name}] ---', file=f, flush=True)
                gen_emb = extract_gen_embeddings(eval_wrapper, motion_loader, renorm_fn)

                fid = evaluate_fid(gt_emb, gen_emb, f)
                kid_mean, kid_std = evaluate_kid(gt_emb, gen_emb, f)
                diversity = evaluate_diversity(gen_emb, f, diversity_times)

                for key, val in [('FID', fid), ('KID_mean', kid_mean),
                                  ('KID_std', kid_std), ('Diversity', diversity)]:
                    all_metrics[key].setdefault(model_name, []).append(val)

            if extra_embs:
                for name, emb in extra_embs.items():
                    print(f'--- [{name}] ---')
                    print(f'--- [{name}] ---', file=f, flush=True)
                    fid = evaluate_fid(gt_emb, emb, f)
                    kid_mean, kid_std = evaluate_kid(gt_emb, emb, f)
                    diversity = evaluate_diversity(emb, f, diversity_times)
                    for key, val in [('FID', fid), ('KID_mean', kid_mean),
                                      ('KID_std', kid_std), ('Diversity', diversity)]:
                        all_metrics[key].setdefault(name, []).append(val)

            if run_mm:
                mm_score_dict = evaluate_multimodality(eval_wrapper, mm_motion_loaders, f, mm_num_times, renorm_fn)
                for key, item in mm_score_dict.items():
                    all_metrics['MultiModality'].setdefault(key, []).append(item)

            print('!!! DONE !!!', file=f, flush=True)

        # Summary
        mean_dict = {}
        report_lines = []
        for metric_name, metric_dict in all_metrics.items():
            print(f'========== {metric_name} Summary ==========')
            print(f'========== {metric_name} Summary ==========', file=f, flush=True)
            for model_name, values in metric_dict.items():
                mean, conf = get_metric_statistics(np.array(values), replication_times)
                mean_dict[f'{metric_name}_{model_name}'] = mean
                line = f'---> [{model_name}] Mean: {mean:.4f}  CInterval: {conf:.4f}'
                print(line)
                print(line, file=f, flush=True)
                report_lines.append(f'{metric_name.rjust(12)} {model_name}: {mean:07.4f} ± {conf:.4f}')

        summary = '\n'.join(sorted(report_lines, key=lambda x: x.split(':')[0][::-1]))
        print(f'\n     For report:\n{summary}\n')
        print(f'\n     For report:\n{summary}\n', file=f, flush=True)

    return mean_dict


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    import sys
    import argparse as _ap

    # Pre-parse --lifted_motions_dirs before evaluation_parser (which calls parse_and_load_from_model)
    _pre = _ap.ArgumentParser(add_help=False)
    _pre.add_argument('--lifted_motions_dirs', nargs='+', default=None,
                      help='Directories of pre-computed lifted motions (e.g. WHAM, mvlift). '
                           'Each directory is evaluated as a separate source alongside generation.')
    _pre_args, _remaining = _pre.parse_known_args(sys.argv[1:])
    sys.argv = [sys.argv[0]] + _remaining  # strip --lifted_motions_dirs before evaluation_parser sees it

    args = evaluation_parser()
    args.lifted_motions_dirs = _pre_args.lifted_motions_dirs
    fixseed(args.seed)

    # dataset guard — args.dataset is loaded from model's args.json
    if args.dataset not in _FIT3D_DATASETS:
        print(f'Warning: --dataset={args.dataset!r} is not a fit3d variant. '
              f'Overriding to "fit3d". This script is for fit3d models only.')
        args.dataset = 'fit3d'

    args.batch_size = 32

    name = os.path.basename(os.path.dirname(args.model_path))
    niter = os.path.basename(args.model_path).replace('model', '').replace('.pt', '')
    log_file = os.path.join(os.path.dirname(args.model_path),
                            f'eval_fit3d_{name}_{niter}_{args.eval_mode}')
    if args.guidance_param != 1.:
        log_file += f'_gscale{args.guidance_param}'
    log_file += '.log'

    print(f'Will save to log file [{log_file}]')
    print(f'Eval mode [{args.eval_mode}]')

    if args.eval_mode == 'debug':
        replication_times = 3
        diversity_times = 100
        run_mm = False
        mm_num_samples = 0
        mm_num_repeats = 0
        mm_num_times = 0
    elif args.eval_mode == 'wo_mm':
        replication_times = 10
        diversity_times = 100
        run_mm = False
        mm_num_samples = 0
        mm_num_repeats = 0
        mm_num_times = 0
    elif args.eval_mode == 'mm_short':
        replication_times = 5
        diversity_times = 100
        run_mm = True
        mm_num_samples = 20
        mm_num_repeats = 30
        mm_num_times = 10
    else:
        raise ValueError(f'Unknown eval_mode: {args.eval_mode!r}')

    dist_util.setup_dist(args.device)
    logger.configure()

    model_mean = np.load(_MODEL_MEAN_PATH)
    model_std = np.load(_MODEL_STD_PATH)
    eval_mean = np.load(_EVAL_MEAN_PATH)
    eval_std = np.load(_EVAL_STD_PATH)

    # Renorm generated motions from fit3d training norm → t2m evaluator norm.
    # Root velocity and foot contact dims have ~25x larger std in fit3d stats vs t2m stats,
    # so the evaluator is effectively blind to those features without this correction.
    _model_mean_t = torch.from_numpy(model_mean).float()
    _model_std_t = torch.from_numpy(model_std).float()
    _eval_mean_t = torch.from_numpy(eval_mean).float()
    _eval_std_t = torch.from_numpy(eval_std).float()
    def renorm_fn(motions):
        motions = motions.cpu().float()
        raw = motions * _model_std_t + _model_mean_t   # undo fit3d norm
        return (raw - _eval_mean_t) / _eval_std_t      # apply t2m norm

    logger.log(f'Loading GT data for split={args.split} ...')
    gt_dataset = Fit3DGTDataset(split=args.split, mean=eval_mean, std=eval_std)

    logger.log(f'Creating gen data loader (dataset={args.dataset}, split={args.split}) ...')
    _gen_dataset = get_dataset(name=args.dataset, num_frames=None, split=args.split, hml_mode='2d')
    _gen_collate = get_collate_fn(args.dataset, hml_mode='2d')
    gen_loader = DataLoader(
        _gen_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=2, drop_last=False, collate_fn=_gen_collate,
    )

    logger.log('Creating model and diffusion ...')
    model, diffusion = create_model_and_diffusion(args, gen_loader)

    logger.log(f'Loading checkpoint from [{args.model_path}] ...')
    load_saved_model(model, args.model_path, use_avg=not args.dont_use_ema)

    if args.guidance_param != 1:
        model = ClassifierFreeSampleModel(model)
    model.to(dist_util.dev())
    model.eval()

    eval_wrapper = EvaluatorMDMWrapper('humanml', dist_util.dev())

    logger.log('Pre-computing GT embeddings (fixed across replications) ...')
    gt_emb = get_gt_embeddings(eval_wrapper, gt_dataset, args.batch_size, dist_util.dev())
    print(f'GT embeddings: {gt_emb.shape}')

    # GT is always included as a reference source (FID=0, KID≈0, Diversity=target value)
    extra_embs = {'gt': gt_emb}
    if args.lifted_motions_dirs:
        for preds_dir in args.lifted_motions_dirs:
            lname = os.path.basename(preds_dir.rstrip('/'))
            logger.log(f'Loading lifted motions [{lname}] from [{preds_dir}] ...')
            lifted_ds = LiftedMotionsDataset(preds_dir, args.split, eval_mean, eval_std)
            lifted_emb = get_gt_embeddings(eval_wrapper, lifted_ds, args.batch_size, dist_util.dev())
            print(f'Lifted embeddings [{lname}]: {lifted_emb.shape}')
            extra_embs[lname] = lifted_emb

    split_key = args.split if args.split != 'val' else 'vald'
    eval_motion_loaders = {
        split_key: lambda: get_mdm_loader(
            model, diffusion, args.batch_size,
            gen_loader, mm_num_samples, mm_num_repeats,
            gen_loader.dataset.opt.max_motion_length,
            num_samples_limit=None,
            scale=args.guidance_param,
        )
    }

    evaluation(eval_wrapper, gt_emb, eval_motion_loaders, log_file,
               replication_times, diversity_times, mm_num_times, run_mm=run_mm,
               extra_embs=extra_embs, renorm_fn=renorm_fn)
