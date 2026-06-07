#!/usr/bin/env python3
# Evaluate 3D human pose predictions (GT cameras assumed).
# Metrics: MPJPE, PA-MPJPE (per-frame Procrustes sim(3)), PCK@50/100mm, Accel (m/s^2).
# Optional: KID (Kernel Inception Distance) via --kid flag.
# Inputs: .npz/.npy files containing 'joints' (T,J,3). Optional alternative keys: 'poses'.
# No mesh/vertex metrics; no pandas; optional CSV output.

import argparse
import os
import re
import csv
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for saving
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# -------------------------- I/O helpers --------------------------


mpjpe_histogram = []
mpjpe_q90_histogram = {}
mpjpe_q99_histogram = {}
mpjpe_max_histogram = {}
mpjpe_q90_inverse_time_histogram = {}
mpjpe_q99_inverse_time_histogram = {}
mpjpe_max_inverse_time_histogram = {}

def compute_mpjpe(pred_joints_mm, gt_joints_mm):
    diff = pred_joints_mm - gt_joints_mm
    #print(diff.shape) [Time, Joints, 3]
    mpjpe = np.linalg.norm(diff, axis=-1).mean()
    mpjpe_histogram.append(mpjpe)
    norms = np.linalg.norm(diff, axis=-1)  # (T,J)
    q90_indices = list(map(tuple, np.argwhere(norms > np.percentile(norms, 90))))   # [(i0,j0), (i1,j1), ...]
    q99_indices = list(map(tuple, np.argwhere(norms > np.percentile(norms, 99))))
    max_idx = np.unravel_index(np.argmax(norms), norms.shape)  # e.g. (i, j)
    T = norms.shape[0]
    for idx_pair in q90_indices:
        mpjpe_q90_histogram[idx_pair] = mpjpe_q90_histogram.get(idx_pair, 0) + 1
        inv_pair = (T - idx_pair[0], idx_pair[1])
        mpjpe_q90_inverse_time_histogram[inv_pair] = mpjpe_q90_inverse_time_histogram.get(inv_pair, 0) + 1
    for idx_pair in q99_indices:
        mpjpe_q99_histogram[idx_pair] = mpjpe_q99_histogram.get(idx_pair, 0) + 1
        inv_pair = (T - idx_pair[0], idx_pair[1])
        mpjpe_q99_inverse_time_histogram[inv_pair] = mpjpe_q99_inverse_time_histogram.get(inv_pair, 0) + 1
    mpjpe_max_histogram[max_idx] = mpjpe_max_histogram.get(max_idx, 0) + 1
    mpjpe_max_inverse_time_histogram[inv_pair] = mpjpe_max_inverse_time_histogram.get(inv_pair, 0) + 1

    return mpjpe


def plot_mpjpe_histogram(save_path):
    plt.figure(figsize=(10, 6))
    plt.hist(mpjpe_histogram, bins=50, alpha=0.7, color='blue')
    plt.title('MPJPE Distribution')
    plt.xlabel('MPJPE (mm)')
    plt.ylabel('Frequency')
    plt.grid(True)
    plt.savefig(save_path)
    plt.close()

def plot_3d_histogram(histogram, title, save_dir):
    # Plot histogram as 2 2D bar charts (one for time, one for joints)
    time_counts = {}
    joint_counts = {}
    for (time_idx, joint_idx), count in histogram.items():
        time_counts[time_idx] = time_counts.get(time_idx, 0) + count
        joint_counts[joint_idx] = joint_counts.get(joint_idx, 0) + count

    sum_counts = sum(histogram.values())
    time_counts = {k: v / sum_counts for k, v in time_counts.items()}
    joint_counts = {k: v / sum_counts for k, v in joint_counts.items()}

    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.bar(time_counts.keys(), time_counts.values(), color='orange')
    plt.title(f'{title} - Time Distribution')
    plt.xlabel('Time Index')
    plt.ylabel('Count')
    plt.subplot(1, 2, 2)
    plt.bar(joint_counts.keys(), joint_counts.values(), color='green')
    plt.title(f'{title} - Joint Distribution')
    plt.xlabel('Joint Index')
    plt.ylabel('Count')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"{title}_2d.png"))
    plt.close()

    # Plot histogram as 3D bar chart
    # Convert histogram dict to arrays
    pairs = np.array(list(histogram.keys()))  # (N, 2)
    counts = np.array(list(histogram.values()))  # (N,)

    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(111, projection='3d')
    ax.bar3d(pairs[:, 0], pairs[:, 1], np.zeros_like(counts), 1, 1, counts, shade=True)
    ax.set_title(title)
    ax.set_xlabel('Time Index')
    ax.set_ylabel('Joint Index')
    ax.set_zlabel('Count')
    plt.savefig(os.path.join(save_dir, f"{title}_3d.png"))
    plt.close()


def procrustes_align_sim3(X, Y):
    """Return Y aligned to X by similarity (scale, rotation, translation).
    X, Y: (J,3)
    """
    Xc = X - X.mean(axis=0, keepdims=True)
    Yc = Y - Y.mean(axis=0, keepdims=True)
    H = Yc.T @ Xc
    U, S, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T
    # Reflection handling
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1.0
        R = Vt.T @ U.T
    varY = (Yc ** 2).sum()
    s = np.trace(np.diag(S)) / varY if varY > 0 else 1.0
    t = X.mean(axis=0) - s * (R @ Y.mean(axis=0))
    Y_aligned = (s * (Y @ R.T)) + t
    return Y_aligned

def compute_pa_mpjpe(pred_joints_mm, gt_joints_mm):
    T = min(len(gt_joints_mm), len(pred_joints_mm))
    # Shapes: (T,J,3)
    total = 0.0
    for t in range(T):
        aligned = procrustes_align_sim3(gt_joints_mm[t], pred_joints_mm[t])
        total += np.linalg.norm(aligned - gt_joints_mm[t], axis=-1).mean()
    return total / max(T, 1)

def compute_pck(pred_joints_mm, gt_joints_mm, threshold_mm):
    dists = np.linalg.norm(pred_joints_mm - gt_joints_mm, axis=-1)  # (T,J)
    correct = (dists < threshold_mm).mean()
    return 100.0 * float(correct)

def compute_accel_error_mps2(pred_joints, gt_joints, dt_seconds, unit='mm'):
    """Acceleration error averaged over joints & valid timesteps; output in m/s^2.
    pred_joints, gt_joints are in 'unit' (mm or m).
    """
    to_m = 0.001 if unit == 'mm' else 1.0
    pred_m = pred_joints * to_m
    gt_m   = gt_joints * to_m
    # second finite difference
    def second_diff(A):
        return A[2:] - 2*A[1:-1] + A[:-2]
    acc_pred = second_diff(pred_m) / (dt_seconds * dt_seconds)
    acc_gt   = second_diff(gt_m) / (dt_seconds * dt_seconds)
    return np.linalg.norm(acc_pred - acc_gt, axis=-1).mean()

# -------------------------- Utilities --------------------------

def ensure_mm(arr, unit):
    return arr * 1000.0 if unit == 'm' else arr

def trim_to_common(gt_joints, pred_joints):
    #assert (gt_joints.shape[0] - pred_joints.shape[0]) ** 2 < 10 ** 2, f"GT and pred joints must have the same number of frames, got {gt_joints.shape[0]} and {pred_joints.shape[0]}"
    assert gt_joints.shape[1] == pred_joints.shape[1], f"GT and pred joints must have the same number of joints, got {gt_joints.shape[1]} and {pred_joints.shape[1]}"
    T = min(gt_joints.shape[0], pred_joints.shape[0])
    J = min(gt_joints.shape[1], pred_joints.shape[1])
    return gt_joints[:T, :J, :], pred_joints[:T, :J, :], T, J

def format_table(rows, headers):
    # Compute column widths
    widths = [len(h) for h in headers]
    for row in rows:
        for i, cell in enumerate(row):
            widths[i] = max(widths[i], len(cell))
    # Header and separator
    header_line = ' | '.join(h.ljust(widths[i]) for i, h in enumerate(headers))
    sep_line = '-+-'.join('-' * widths[i] for i in range(len(headers)))
    lines = [header_line, sep_line]
    # Rows
    for row in rows:
        lines.append(' | '.join(str(cell).ljust(widths[i]) for i, cell in enumerate(row)))
    return '\n'.join(lines)

# -------------------------- Log ---------------------------

class Logger:
    def __init__(self, filepath):
        self.filepath = filepath
        self.file = open(filepath, 'a')

    def log(self, message):
        print(message)
        self.file.write(message + '\n')
        self.file.flush()

    def close(self):
        self.file.close()


# -------------------------- KID helpers --------------------------

def joints_to_hml_features(joints_m, device, mean, std, skel_kwargs):
    """Convert (T, 22, 3) metres to normalized (T-1, 263) HumanML features.

    Returns (features, length) where features is (T-1, 263) float32 numpy
    and length is T-1.
    """
    import torch
    from diffusion.torch_process_motions import process_file_torch
    t = torch.from_numpy(joints_m.astype(np.float32)).unsqueeze(0).to(device)  # (1,T,22,3)
    data, _, _, _ = process_file_torch(t, **skel_kwargs)                        # (1,T-1,263)
    data = data.squeeze(0).cpu().numpy()                                         # (T-1,263)
    return (data - mean) / std, data.shape[0]


def collect_embeddings(eval_wrapper, features_list, lengths, batch_size=32):
    """Run a list of (T_i, 263) normalized features through the motion encoder.

    Returns (N, 512) numpy array of embeddings.
    """
    import torch
    T_max = max(lengths)
    N = len(features_list)
    padded = np.zeros((N, T_max, 263), dtype=np.float32)
    for i, feat in enumerate(features_list):
        padded[i, :feat.shape[0]] = feat
    m_lens = torch.tensor(lengths, dtype=torch.long)
    all_emb = []
    for s in range(0, N, batch_size):
        e = min(s + batch_size, N)
        emb = eval_wrapper.get_motion_embeddings(
            torch.from_numpy(padded[s:e]), m_lens[s:e]
        )
        all_emb.append(emb.cpu().numpy())
    return np.concatenate(all_emb, axis=0)


def evaluate_kid(gt_emb, pred_emb, n_subsets=100):
    """Bootstrap KID: subset_size=N with replace=True (full-population bootstrap)."""
    from eval.unconstrained.metrics.kid import polynomial_mmd_averages
    N = min(len(gt_emb), len(pred_emb))
    mmds, _ = polynomial_mmd_averages(
        gt_emb, pred_emb,
        n_subsets=n_subsets, subset_size=N, replace=True, ret_var=True,
    )
    return float(mmds.mean()), float(mmds.std())


# -------------------------- Main --------------------------
def main():
    parser = argparse.ArgumentParser(description='Evaluate 3D human pose predictions (GT cameras).')
    parser.add_argument('preds', type=str, nargs='+', help='One or more prediction files')
    parser.add_argument('--split_path', type=str, default="dataset/fit3d/test.txt", help='Path to directory containing prediction files')
    parser.add_argument('--gt', type=str, default="dataset/fit3d/smpl_gt_poses", help='Path to GT .npy (contains joints (T,J,3))')
    parser.add_argument('--unit', type=str, choices=['mm', 'm'], default='m', help='Coordinate units in files')
    parser.add_argument('--fps', type=float, default=20.0, help='Frames per second for Accel')
    parser.add_argument('--out', type=str, default=None, help='Optional CSV output path')
    parser.add_argument('--kid', action='store_true', help='Compute KID (slow; requires HumanML encoder)')
    parser.add_argument('--device', type=str, default='cuda', help='Device for KID encoder (not used for lifting)')
    parser.add_argument('--kid_norm_dir', type=str, default='dataset', help='Dir containing t2m_mean.npy and t2m_std.npy for KID normalisation')
    args = parser.parse_args()

    # ---- KID setup (once before the model loop) ----
    kid_enabled = args.kid
    if kid_enabled:
        import torch
        from data_loaders.humanml.networks.evaluator_wrapper import EvaluatorMDMWrapper
        from diffusion.torch_process_motions import get_skeleton_kwargs_from_dataset

        hml_mean = np.load(os.path.join(args.kid_norm_dir, 't2m_mean.npy'))
        hml_std  = np.load(os.path.join(args.kid_norm_dir, 't2m_std.npy'))
        skel_kwargs = get_skeleton_kwargs_from_dataset('fit3d')
        # 'humanml' resolves to the t2m checkpoint (dim_pose=263)
        eval_wrapper = EvaluatorMDMWrapper('humanml', args.device)

    results = []
    with open(args.split_path, 'r') as f:
        split = f.read().strip().split('\n')

    for pred_path in args.preds:
        model_name = os.path.basename(pred_path)
        if model_name.startswith("model000"):
            model_name = pred_path.split(os.sep)[-2]
            date_regex = "\d{4}.\d{2}.\d{2}_\d{2}.\d{2}"
            if re.search(date_regex, model_name):
                model_name = model_name[:-17]
        success_count = 0
        mpjpe_val = 0
        pa_mpjpe_val = 0
        pck50 = 0
        pck100 = 0
        accel = 0

        gt_features, gt_lengths     = [], []
        pred_features, pred_lengths = [], []

        # ---- OFF-BY-1 DIAGNOSTIC START ----
        mpjpe_shift_neg1 = 0  # pred 1 frame behind gt: gt[1:] vs pred[:-1]
        mpjpe_shift_pos1 = 0  # pred 1 frame ahead of gt: gt[:-1] vs pred[1:]
        mpjpe_shift_neg2 = 0  # pred 2 frames behind gt: gt[2:] vs pred[:-2]
        mpjpe_shift_pos2 = 0  # pred 2 frames ahead of gt: gt[:-2] vs pred[2:]
        shift_count = 0
        # ---- OFF-BY-1 DIAGNOSTIC END ----

        for fname in split:
            gt_joints = np.load(os.path.join(args.gt, fname + ".npy"))
            if gt_joints.ndim != 3 or gt_joints.shape[2] != 3:
                print(f"Warning: GT joints must be (T,J,3), got {gt_joints.shape}")
                continue

            pred_file_path = os.path.join(pred_path, fname + ".npy")
            if not os.path.exists(pred_file_path):
                print(f"Warning: Pred file {pred_file_path} does not exist")
                continue

            pred_joints = np.load(pred_file_path)
            if pred_joints.ndim != 3 or pred_joints.shape[2] != 3:
                print(f"Warning: Pred joints must be (T,J,3), got {pred_joints.shape}")
                continue

            success_count += 1
            # Trim to common (T,J)
            gt_common, pred_common, T, J = trim_to_common(gt_joints, pred_joints)

            # Convert to mm for MPJPE/PA/PCK; keep original unit for accel calc
            gt_mm = ensure_mm(gt_common.copy(), args.unit)
            pred_mm = ensure_mm(pred_common.copy(), args.unit)

            mpjpe = compute_mpjpe(pred_mm, gt_mm)
            if mpjpe > 500:
                print(f"Warning: MPJPE for {fname} is {mpjpe}")
            mpjpe_val += mpjpe
            pa_mpjpe_val += compute_pa_mpjpe(pred_mm, gt_mm)
            pck50 += compute_pck(pred_mm, gt_mm, 50.0)
            pck100 += compute_pck(pred_mm, gt_mm, 100.0)
            accel += compute_accel_error_mps2(pred_common, gt_common, dt_seconds=1.0/args.fps, unit=args.unit)

            # ---- OFF-BY-1 DIAGNOSTIC START ----
            if T >= 4:
                shift_count += 1
                mpjpe_shift_neg1 += np.linalg.norm(gt_mm[1:] - pred_mm[:-1], axis=-1).mean()
                mpjpe_shift_pos1 += np.linalg.norm(gt_mm[:-1] - pred_mm[1:], axis=-1).mean()
                mpjpe_shift_neg2 += np.linalg.norm(gt_mm[2:] - pred_mm[:-2], axis=-1).mean()
                mpjpe_shift_pos2 += np.linalg.norm(gt_mm[:-2] - pred_mm[2:], axis=-1).mean()
            # ---- OFF-BY-1 DIAGNOSTIC END ----

            if kid_enabled:
                if gt_common.shape[1] != 22:
                    print(f"Warning: KID skipped for {fname}: GT has {gt_common.shape[1]} joints, expected 22")
                elif pred_common.shape[1] != 22:
                    print(f"Warning: KID skipped for {fname}: pred has {pred_common.shape[1]} joints, expected 22")
                else:
                    gt_m   = gt_common   if args.unit == 'm' else gt_common   / 1000.0
                    pred_m = pred_common if args.unit == 'm' else pred_common / 1000.0
                    gt_feat,   gt_len   = joints_to_hml_features(gt_m,   args.device, hml_mean, hml_std, skel_kwargs)
                    pred_feat, pred_len = joints_to_hml_features(pred_m, args.device, hml_mean, hml_std, skel_kwargs)
                    if gt_len < 8:
                        print(f"Warning: KID skipped for {fname}: GT has {gt_len} frames after HML conversion (min 8)")
                    elif pred_len < 8:
                        print(f"Warning: KID skipped for {fname}: pred has {pred_len} frames after HML conversion (min 8)")
                    else:
                        gt_features.append(gt_feat)
                        gt_lengths.append(gt_len)
                        pred_features.append(pred_feat)
                        pred_lengths.append(pred_len)

        # ---- OFF-BY-1 DIAGNOSTIC START ----
        if shift_count > 0:
            print(f'[Off-by-1 diagnostic] {model_name}:')
            print(f'  MPJPE normal  (gt[0:]  vs pred[0:] ): {mpjpe_val / success_count:.2f} mm')
            print(f'  MPJPE shift-1 (gt[1:]  vs pred[:-1]): {mpjpe_shift_neg1 / shift_count:.2f} mm  (pred 1 frame behind gt)')
            print(f'  MPJPE shift+1 (gt[:-1] vs pred[1:] ): {mpjpe_shift_pos1 / shift_count:.2f} mm  (pred 1 frame ahead of gt)')
            print(f'  MPJPE shift-2 (gt[2:]  vs pred[:-2]): {mpjpe_shift_neg2 / shift_count:.2f} mm  (pred 2 frames behind gt)')
            print(f'  MPJPE shift+2 (gt[:-2] vs pred[2:] ): {mpjpe_shift_pos2 / shift_count:.2f} mm  (pred 2 frames ahead of gt)')
        # ---- OFF-BY-1 DIAGNOSTIC END ----

        if kid_enabled and len(gt_features) != len(pred_features):
            print(f"Warning: GT/pred KID feature count mismatch ({len(gt_features)} vs {len(pred_features)}) — unexpected, check motion preparation.")

        result = {
            'Model': model_name,
            'MPJPE (mm)': f"{mpjpe_val / success_count:.2f}",
            'PA-MPJPE (mm)': f"{pa_mpjpe_val / success_count:.2f}",
            'PCK@50mm (%)': f"{pck50 / success_count:.2f}",
            'PCK@100mm (%)': f"{pck100 / success_count:.2f}",
            'Accel (m/s^2)': f"{accel / success_count:.3f}",
            'KID mean': 'N/A',
            'KID std':  'N/A',
        }

        if kid_enabled and len(pred_features) > 0:
            print(f'========== Computing KID for {model_name} ==========')
            gt_embeddings   = collect_embeddings(eval_wrapper, gt_features,   gt_lengths)
            pred_embeddings = collect_embeddings(eval_wrapper, pred_features, pred_lengths)
            print(f'GT embeddings: {gt_embeddings.shape}  Pred embeddings: {pred_embeddings.shape}')
            N_kid = min(len(gt_embeddings), len(pred_embeddings))
            kid_mean, kid_std = evaluate_kid(gt_embeddings, pred_embeddings)
            result['KID mean'] = f"{kid_mean:.4f}"
            result['KID std']  = f"{kid_std:.4f} (N={N_kid}, bootstrap, exploratory)"

        results.append(result)

    headers = ['Model', 'MPJPE (mm)', 'PA-MPJPE (mm)', 'PCK@50mm (%)', 'PCK@100mm (%)', 'Accel (m/s^2)']
    if kid_enabled:
        headers += ['KID mean', 'KID std']
    rows = [[r[h] for h in headers] for r in results]
    print(format_table(rows, headers))

    total = sum(mpjpe_histogram)
    sorted_histogram = sorted(mpjpe_histogram)
    precentiles_to_print = [10, 25, 50, 75, 90, 95, 99]
    temp_sum = 0
    for i, mpjpe in enumerate(sorted_histogram):
        temp_sum += mpjpe
        if temp_sum >= total * precentiles_to_print[0] / 100:
            print(f"Percentile {precentiles_to_print[0]}: {i} {i / len(sorted_histogram):.2f} {mpjpe}")
            precentiles_to_print.pop(0)
            if not precentiles_to_print:
                break

    plot_mpjpe_histogram(os.path.join(os.path.dirname(args.out) if args.out else '.', "mpjpe_histogram.png"))
    plot_3d_histogram(mpjpe_q90_histogram, "MPJPE Q90 Histogram", os.path.dirname(args.out) if args.out else '.')
    plot_3d_histogram(mpjpe_q90_inverse_time_histogram, "MPJPE Q90 Inverse Time Histogram", os.path.dirname(args.out) if args.out else '.')
    plot_3d_histogram(mpjpe_q99_histogram, "MPJPE Q99 Histogram", os.path.dirname(args.out) if args.out else '.')
    plot_3d_histogram(mpjpe_q99_inverse_time_histogram, "MPJPE Q99 Inverse Time Histogram", os.path.dirname(args.out) if args.out else '.')

    if args.out:
        with open(args.out, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(headers)
            for r in results:
                writer.writerow([r[h] for h in headers])
        print(f"\nSaved CSV to: {args.out}")

if __name__ == '__main__':
    main()
