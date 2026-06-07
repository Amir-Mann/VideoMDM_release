import os
import re
import subprocess
from collections import OrderedDict
from datetime import datetime

import numpy as np
import torch
from tqdm import tqdm

from data_loaders.humanml.scripts.motion_process import recover_from_ric
from data_loaders.tensors import collate
from diffusion import logger
from utils import dist_util
from utils.fixseed import fixseed
from utils.model_util import create_model_and_diffusion, load_saved_model
from utils.parser_util import evaluation_parser
from utils.sampler_util import ClassifierFreeSampleModel, UnconditionalGuidadedSampler

# from video.dataset_info import dataset_torch_mean, dataset_torch_std

torch.multiprocessing.set_sharing_strategy("file_system")
NBA_DATASET_DIR = (
    "/home/amir.mann/MAS/dataset/nba/motions"  # MAS assumes same location
)
SOURCE_FPS = 60


def quick_log(msg, file):
    print(msg)
    print(msg, file=file, flush=True)


def generate_and_save_motions(
    model,
    diffusion,
    num_samples,
    saved_motions_dir,
    guidance_param,
    device,
    batch_size,
    fps=30,
):
    assert SOURCE_FPS % fps == 0, "Source FPS must be divisible by target FPS"
    """Generate motions using the model and save them to files."""
    os.makedirs(saved_motions_dir, exist_ok=True)

    gt_motion_lengths = []

    # Find all .npy files and read only their frame counts
    npy_files = sorted([f for f in os.listdir(NBA_DATASET_DIR) if f.endswith(".npy")])
    # Limit generation to min of requested samples and available ground truth motions
    num_samples = min(num_samples, len(npy_files))
    print(
        f"Generating {num_samples} motions ({len(npy_files)} available GT motions)..."
    )

    print(f"Reading motion lengths from {NBA_DATASET_DIR}...")
    for npy_file in tqdm(npy_files[:num_samples]):
        file_path = os.path.join(NBA_DATASET_DIR, npy_file)
        motion_data = np.load(file_path, mmap_mode="r")
        assert (
            len(motion_data.shape) == 3 and motion_data.shape[2] == 3
        ), f"Unexpected shape {motion_data.shape} in {npy_file}"
        # Format is (frames, joints, 3)
        gt_motion_lengths.append(motion_data.shape[0] // (SOURCE_FPS // fps))

    print(f"Found {len(gt_motion_lengths)} motion files with lengths")

    model.eval()
    sample_fn = diffusion.p_sample_loop

    all_motions = []

    # Generate in batches
    num_batches = (num_samples + batch_size - 1) // batch_size

    tqdm_iterator = tqdm(range(num_batches), desc="Generating motions")
    for batch_idx in tqdm_iterator:
        current_batch_size = min(batch_size, num_samples - batch_idx * batch_size)
        if current_batch_size <= 0:
            break

        # Get the indices for this batch
        batch_start_idx = batch_idx * batch_size
        batch_indices = list(
            range(
                batch_start_idx, min(batch_start_idx + current_batch_size, num_samples)
            )
        )

        # Get target lengths for this batch
        batch_target_lengths = [gt_motion_lengths[idx] for idx in batch_indices]
        # Use max_frames for generation (model needs fixed size), then trim afterwards
        n_frames = max(batch_target_lengths)

        #print(
        #    f"Generating batch {batch_idx + 1}/{num_batches} ({current_batch_size} samples, max length {n_frames})..."
        #)
        tqdm_iterator.set_description(f"Generating batch, current max length: {n_frames}")

        collate_args = []
        text_prompt = "a person is playing basketball."
        for i in range(current_batch_size):
            cur_frames = batch_target_lengths[i]
            collate_args.append(
                {
                    "inp": torch.zeros(cur_frames),
                    "tokens": None,
                    "lengths": cur_frames,
                    "text": text_prompt,
                }
            )
        _, model_kwargs = collate(collate_args)

        # Move to device
        model_kwargs["y"] = {
            key: val.to(device) if torch.is_tensor(val) else val
            for key, val in model_kwargs["y"].items()
        }

        # Add CFG scale if needed
        if guidance_param != 1:
            model_kwargs["y"]["scale"] = (
                torch.ones(current_batch_size, device=device) * guidance_param
            )

        # Encode text
        if "text" in model_kwargs["y"].keys():
            model_kwargs["y"]["text_embed"] = model.encode_text(
                model_kwargs["y"]["text"]
            )

        # Generate sample
        motion_shape_batch = (current_batch_size, model.njoints, model.nfeats, n_frames)
        with torch.no_grad():
            if isinstance(model, UnconditionalGuidadedSampler):
                model.reset_timesteps_cache()
            sample = sample_fn(
                model,
                motion_shape_batch,
                clip_denoised=False,
                model_kwargs=model_kwargs,
                skip_timesteps=0,
                init_image=None,
                progress=False,
                dump_steps=None,
                noise=None,
                const_noise=False,
            )

        # Recover XYZ positions from HumanML3D vector representation
        if model.data_rep == "hml_vec":
            n_joints = int((sample.shape[1] + 1) // 12)
            sample_copy = sample.cpu().numpy()

            # Get the dataset transform - need to load mean/std for denormalization
            mean_path = "dataset/nba/Mean.npy"
            std_path = "dataset/nba/Std.npy"

            if os.path.exists(mean_path) and os.path.exists(std_path):
                mean = np.load(mean_path)
                std = np.load(std_path)
                # Convert to torch tensor for transform
                sample_torch = (
                    sample.cpu().permute(0, 2, 3, 1).float()
                )  # [bs, nframes, njoints, nfeats]
                sample_torch = (
                    sample_torch * torch.from_numpy(std).float()
                    + torch.from_numpy(mean).float()
                )
                # Recover from RIC representation
                sample_ric = recover_from_ric(sample_torch, n_joints)
                sample = sample_ric.view(-1, *sample_ric.shape[2:]).permute(0, 2, 3, 1)
            else:
                # Fallback: direct recovery without denormalization
                sample_perm = sample.cpu().permute(0, 2, 3, 1).float()
                sample_ric = recover_from_ric(sample_perm, n_joints)
                sample = sample_ric.view(-1, *sample_ric.shape[2:]).permute(0, 2, 3, 1)

        # Convert to XYZ using rot2xyz
        rot2xyz_pose_rep = (
            "xyz" if model.data_rep in ["xyz", "hml_vec"] else model.data_rep
        )
        rot2xyz_mask = (
            None
            if rot2xyz_pose_rep == "xyz"
            else model_kwargs["y"]["mask"].reshape(current_batch_size, n_frames).bool()
        )

        with torch.no_grad():
            sample_xyz = model.rot2xyz(
                x=sample,
                mask=rot2xyz_mask,
                pose_rep=rot2xyz_pose_rep,
                glob=True,
                translation=True,
                jointstype="smpl",
                vertstrans=True,
                betas=None,
                beta=0,
                glob_rot=None,
                get_rotations_back=False,
            )

        for i in range(current_batch_size):
            motion = sample_xyz[i].cpu().numpy()

            # rot2xyz may return different shapes, ensure it's (frames, joints, 3)
            # Expected format: (frames, joints, 3) like motionbert_predictions
            if len(motion.shape) == 3:
                # Check if we need to transpose
                # If shape is (joints, frames, 3), transpose to (frames, joints, 3)
                if motion.shape[0] < motion.shape[1] and motion.shape[2] == 3:
                    # Likely (joints, frames, 3) -> transpose to (frames, joints, 3)
                    motion = motion.transpose(1, 0, 2)
                # If shape is (joints, 3, frames), transpose to (frames, joints, 3)
                elif motion.shape[1] == 3 and motion.shape[2] > motion.shape[0]:
                    motion = motion.transpose(2, 0, 1)

            # Samples are generated in equal length (the max in the batch), so we need to trim the motion to the target length
            motion = motion[: batch_target_lengths[i]]

            # # Turn upside down: not needed since mas flips the motion during evaluation
            # motion_lowest_part = np.min(motion[:, :, 1])
            # motion_highest_part = np.max(motion[:, :, 1])
            # motion[:, :, 1] = (
            #     -motion[:, :, 1] + motion_lowest_part + motion_highest_part
            # )

            # Verify final shape is (frames, joints, 3)
            assert (
                len(motion.shape) == 3 and motion.shape[2] == 3
            ), f"Unexpected motion shape {motion.shape}"
            all_motions.append(motion)

        #print(f"Note - turned upside down the generated motions")
        #print(f"Generated {len(all_motions)}/{num_samples} motions so far...")

    # Save each motion as a separate .npy file in the format expected by MAS evaluator
    # Format: (frames, joints, 3) per file, like dataset/nba/motionbert_predictions/
    # Each motion is already in shape [frames, njoints, 3] from sample_xyz
    print(f"Saving {len(all_motions)} individual motion files to {saved_motions_dir}...")

    for motion_idx, motion in enumerate(all_motions):
        # Motion is already in shape [frames, njoints, 3] and trimmed to length
        # Ensure it matches the reference format: (frames, joints, 3) as float32
        motion_save = motion.astype(np.float32)

        # Verify shape is (frames, joints, 3)
        if len(motion_save.shape) != 3 or motion_save.shape[2] != 3:
            # If shape is wrong, try to fix it
            if len(motion_save.shape) == 2:
                # Assume it's already flattened incorrectly
                raise ValueError(
                    f"Unexpected motion shape: {motion_save.shape}, expected (frames, joints, 3)"
                )
            # If shape is [frames, 3, joints] or similar, transpose
            if motion_save.shape[1] == 3 and motion_save.shape[2] != 3:
                motion_save = motion_save.transpose(0, 2, 1)  # [frames, joints, 3]

        # Save as individual file: {motion_idx}.npy
        # Using simple numbering: 0.npy, 1.npy, 2.npy, etc. to match motionbert_predictions format
        motion_file = os.path.join(saved_motions_dir, f"{motion_idx}.npy")
        np.save(motion_file, motion_save)

    if all_motions:
        example_shape = all_motions[0].shape
        print(
            f"Saved {len(all_motions)} motions as individual .npy files to {saved_motions_dir}"
        )
        print(
            f"Each file has shape {example_shape} (frames, joints, 3) matching motionbert_predictions format"
        )
    return saved_motions_dir


def run_external_evaluation(
    evaluator_path, model_path, saved_samples_path, num_views=5, replication_times=5, evaluate_on_training_data=False
):
    """Run the external evaluation subprocess from MAS directory."""
    MAS_DIR = "/home/amir.mann/MAS2"
    MAS_EVALUATOR_SCRIPT = "eval.evaluate"  # Relative path from MAS directory
    MAS_PYTHON_PATH = "/home/amir.mann/miniconda3/envs/mas/bin/python"

    # Convert paths to absolute paths if they're relative
    evaluator_path_abs = os.path.abspath(evaluator_path)
    model_path_abs = os.path.abspath(model_path)
    saved_samples_path_abs = os.path.abspath(saved_samples_path)

    command = [
        MAS_PYTHON_PATH,
        "-m",
        MAS_EVALUATOR_SCRIPT,
        "--evaluator_path",
        evaluator_path_abs,
        "--model_path",
        model_path_abs,
        "--subjects",
        "saved_samples",
        "--saved_samples_path",
        saved_samples_path_abs,
        "--num_views",
        str(num_views),
        "--num_eval_iterations",
        str(replication_times),
        
        #"--vis_subjects", "saved_samples",
        #"--num_visualize_samples", "2",
    ]
    if evaluate_on_training_data:
        command.append("--evaluate_on_training_data")

    print(f"Running external evaluation from MAS directory: {' '.join(command)}")
    print(f"Working directory: {MAS_DIR}")
    try:
        result = subprocess.run(
            command,
            cwd=MAS_DIR,  # Run from MAS directory
            capture_output=True,
            text=True,
            timeout=600,  # 10 minutes timeout
        )
        print(f"External evaluation stdout: {result.stdout}")
        print(f"External evaluation stderr: {result.stderr}")
        return result.returncode == 0
    except subprocess.TimeoutExpired:
        print("External evaluation timed out")
        return False
    except Exception as e:
        print(f"Error running external evaluation: {e}")
        return False


def read_evaluation_results(output_file):
    """Read FID, diversity, precision, and recall from the output file.

    Returns both values and confidence intervals.

    Expected format:
    Line 1: Header with column names (fid, diversity, precision, recall)
    Line 2+: model_name value±ci value±ci value±ci value±ci
    """
    results = {
        "fid": None,
        "diversity": None,
        "precision": None,
        "recall": None,
        "fid_ci": None,
        "diversity_ci": None,
        "precision_ci": None,
        "recall_ci": None,
    }

    if not os.path.exists(output_file):
        print(f"Output file not found: {output_file}")
        return results

    print(f"Reading evaluation results from {output_file}")
    with open(output_file, "r") as f:
        lines = f.readlines()

    # Parse each data line: model_name value±ci value±ci value±ci value±ci
    for line in lines:
        line = line.strip()
        if not line:
            continue

        # Skip header line
        if (
            line.startswith("fid")
            or "fid" in line.lower()
            and "diversity" in line.lower()
        ):
            continue

        # Split by whitespace - format: model_name fid±ci diversity±ci precision±ci recall±ci
        parts = line.split()
        if len(parts) < 5:
            continue

        # Extract model name (first part) and metrics (remaining 4 parts)
        model_name = parts[0]
        metrics = parts[1:5]  # fid, diversity, precision, recall

        # Parse each metric value±confidence_interval
        for idx, metric_name in enumerate(["fid", "diversity", "precision", "recall"]):
            if idx < len(metrics):
                metric_str = metrics[idx]
                # Extract both value and confidence interval
                if "±" in metric_str:
                    parts_metric = metric_str.split("±")
                    value_str = parts_metric[0].strip()
                    ci_str = parts_metric[1].strip() if len(parts_metric) > 1 else "0"
                else:
                    value_str = metric_str.strip()
                    ci_str = "0"

                # Parse the value (handles both regular floats and scientific notation)
                try:
                    value = float(value_str)
                    results[metric_name] = value
                    # Parse confidence interval
                    ci_value = float(ci_str)
                    results[f"{metric_name}_ci"] = ci_value
                    print(f"Found {metric_name}: {value} ± {ci_value}")
                except ValueError:
                    print(
                        f"Warning: Could not parse {metric_name} value: {value_str} or CI: {ci_str}"
                    )

        # Only process first data line (assuming one model result per file)
        break

    return results


def evaluation(
    log_file,
    log_function=print,
    external_results=None,
):
    with open(log_file, "w") as f:
        all_metrics = OrderedDict(
            {
                "Matching Score": OrderedDict({}),
                "R_precision": OrderedDict({}),
                "Recall": OrderedDict({}),
                "Precision": OrderedDict({}),
                "FID": OrderedDict({}),
                "Diversity": OrderedDict({}),
                "MultiModality": OrderedDict({}),
            }
        )

        # Store confidence intervals from external evaluation
        external_conf_intervals = {}

        # Use external evaluation results from MAS
        quick_log("Using external evaluation results", file=f)
        for model_name, metrics in external_results.items():
            all_metrics["FID"][model_name] = [metrics["fid"]]
            external_conf_intervals[("FID", model_name)] = metrics["fid_ci"]
            all_metrics["Diversity"][model_name] = [metrics["diversity"]]
            external_conf_intervals[("Diversity", model_name)] = metrics["diversity_ci"]
            all_metrics["Precision"][model_name] = [metrics["precision"]]
            external_conf_intervals[("Precision", model_name)] = metrics["precision_ci"]
            all_metrics["Recall"][model_name] = [metrics["recall"]]
            external_conf_intervals[("Recall", model_name)] = metrics["recall_ci"]

        # print(all_metrics['Diversity'])
        gt_fid = None
        test_fid = None
        mean_dict = {}
        to_report_strings = []
        for metric_name, metric_dict in all_metrics.items():
            quick_log("========== %s Summary ==========" % metric_name, file=f)
            for model_name, values in metric_dict.items():
                # print(metric_name, model_name)
                # Use external confidence interval from MAS evaluation
                mean = np.mean(np.array(values))
                conf_interval = external_conf_intervals[(metric_name, model_name)]
                mean_dict[metric_name + "_" + model_name] = mean
                # print(mean, mean.dtype)
                if isinstance(mean, np.float64) or isinstance(mean, np.float32):
                    quick_log(
                        f"---> [{model_name}] Mean: {mean:.4f} CInterval: {conf_interval:.4f}",
                        file=f,
                    )
                    if metric_name == "FID":
                        if model_name == "ground truth":
                            gt_fid = mean
                        if model_name in "test||vald":
                            test_fid = mean
                    to_report_strings.append(
                        f"{metric_name.rjust(10)} {model_name[:4]} : {mean:05.2f}±{conf_interval:.2f}"
                    )
                elif isinstance(mean, np.ndarray):
                    line = f"---> [{model_name}]"
                    for i in range(len(mean)):
                        line += "(top %d) Mean: %.4f CInt: %.4f;" % (
                            i + 1,
                            mean[i],
                            conf_interval[i],
                        )
                    quick_log(line, file=f)
        try:
            strings = "\n".join(
                sorted(to_report_strings, key=lambda x: x.split(":")[0][::-1])
            )
            log_function(f"\n     For report:\n{strings}\n")
            log_function(f"\n     FID:\nGT  : {gt_fid}\nTest:{test_fid}\n")
        except Exception as e:
            print(e)
        return mean_dict


if __name__ == "__main__":
    TIME_GAP = 5
    MIN_TIMESTEP = 5
    args = evaluation_parser()
    fixseed(args.seed)

    name = os.path.basename(os.path.dirname(args.model_path))
    niter = os.path.basename(args.model_path).replace("model", "").replace(".pt", "")
    dataset = args.eval_dataset if args.eval_dataset else args.dataset
    log_file = os.path.join(
        os.path.dirname(args.model_path), "eval_{}_{}_{}".format(dataset, name, niter)
    )
    if args.evaluate_training_data:
        log_file += "_traindata"
    if args.mas_data_dir is not None:
        log_file += "_masdata"
    if args.guidance_param != 1.0:
        log_file += f"_gscale{args.guidance_param}"
    log_file += f"_{args.eval_mode}"

    print(f"Will save to log file [{log_file}]")

    print(f"Eval mode [{args.eval_mode}]")
    if args.eval_mode == "debug":
        num_samples_limit = 1024
        replication_times = 1
    elif args.eval_mode == "full":
        num_samples_limit = (
            100000  # will be limited by the number of available GT motions
        )
        replication_times = 20

    dist_util.setup_dist(args.device)
    logger.configure()

    # Load args from model directory to match architecture
    import json

    model_args_path = os.path.join(os.path.dirname(args.model_path), "args.json")
    if os.path.exists(model_args_path):
        with open(model_args_path, "r") as f:
            model_args_dict = json.load(f)
        # Update args with model args for missing keys
        for key, value in model_args_dict.items():
            if not hasattr(args, key):
                setattr(args, key, value)

    # Generate motions and save them for external MAS evaluation
    logger.log("Creating model and diffusion...")
    # Create a dummy dataset-like object for model creation
    # We don't need actual data, just need the model architecture
    from collections import namedtuple

    # Create minimal dummy data for model initialization
    DummyDataset = namedtuple("DummyDataset", ["num_actions"])
    dummy_data = namedtuple("Data", ["dataset"])(DummyDataset(num_actions=1))

    # Load args from saved model
    if "dataset" not in vars(args) or args.dataset != "nba":
        args.dataset = model_args_dict.get("dataset", "nba")

    model, diffusion = create_model_and_diffusion(args, dummy_data)

    logger.log(f"Loading checkpoints from [{args.model_path}]...")
    load_saved_model(model, args.model_path, use_avg=not args.dont_use_ema)

    if args.guidance_param != 1:
        model = UnconditionalGuidadedSampler(
            model,
            time_gap=TIME_GAP,
            min_timestep=MIN_TIMESTEP,
            max_timestep=diffusion.num_timesteps,
        )  # wrapping model with the classifier-free sampler
    model.to(dist_util.dev())
    model.eval()  # disable random masking

    num_samples_to_generate = num_samples_limit
    saved_motions_dir = os.path.join(
        os.path.dirname(args.model_path), "saved_motions_for_eval"
    )

    print(f"Generating {num_samples_to_generate} motions...")
    saved_path = generate_and_save_motions(
        model,
        diffusion,
        num_samples_to_generate,
        saved_motions_dir,
        args.guidance_param,
        dist_util.dev(),
        args.batch_size,
        fps=30,
    )

    # Run external evaluation
    evaluator_path = args.evaluator_dir_path
    external_output_file = os.path.join(
        os.path.dirname(args.model_path),
        f"{os.path.basename(args.model_path).replace('.pt', '')}_eval_uniform.txt",
    )

    print(f"Running external evaluation...")
    success = run_external_evaluation(
        evaluator_path=evaluator_path,
        model_path=args.model_path,
        saved_samples_path=saved_path,
        num_views=5,
        replication_times=replication_times,
        evaluate_on_training_data=args.evaluate_training_data,
    )

    if not success:
        print("ERROR: External MAS evaluation failed.")
        print(f"Expected output file: {external_output_file}")
        exit(1)

    print(f"External evaluation completed. Reading results from {external_output_file}")
    external_results = read_evaluation_results(external_output_file)

    # Verify we got results
    if not any(
        v is not None
        for v in [
            external_results.get("fid"),
            external_results.get("diversity"),
            external_results.get("precision"),
            external_results.get("recall"),
        ]
    ):
        print("ERROR: No valid results found in external evaluation output.")
        print(f"Expected output file: {external_output_file}")
        exit(1)

    external_results_dict = {"generated": external_results}
    print(f"Using external evaluation results")

    evaluation(
        log_file,
        external_results=external_results_dict,
    )
    print(f"External MAS evaluation complete. Results saved to: {log_file}")
