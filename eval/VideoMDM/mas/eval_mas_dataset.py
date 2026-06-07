import os
import re
import time
from collections import OrderedDict

import numpy as np
import torch
from torch.utils.data import Dataset

from diffusion.torch_process_motions import process_file_torch
from data_loaders.humanml.common.quaternion import (
    qbetween,
    qrot
)

MIN_SEEDS = 83
__global_bbox_scale_hml = np.sqrt(3) # For 3 dimentions, manually checked that it is correct


def calculate_bbox_scale(motion):
    # motion: (seq_len, njoints, 3)
    mins_per_frame = np.min(motion, axis=1) # (seq_len, 3)
    maxs_per_frame = np.max(motion, axis=1) # (seq_len, 3)
    diffs_per_frame = maxs_per_frame - mins_per_frame # (seq_len, 3)
    diff_squared = diffs_per_frame ** 2 # (seq_len, 3)
    bbox_sizes = np.sqrt(np.sum(diff_squared, axis=1)) # (seq_len,)
    value = np.mean(bbox_sizes) # scalar
    return value


def torch_normlize_motion(motion):
    B, T, J, _ = motion.shape # (B, T, J, 3)
    translation_before_rotation = torch.zeros((B, 3)) # (B, 3)
    translation_before_rotation[:, [0, 2]] = -motion[:, 0, 0, [0, 2]] # Root starts at origin on XZ plane
    translation_before_rotation[:, 1] = -torch.amin(motion[:, :, :, 1], dim=(1, 2)) # Minimal joint starts at y=0
    motion += translation_before_rotation.unsqueeze(1).unsqueeze(2) # (B, T, J, 3)

    root_pos_init = motion[:, 0] # (B, J, 3)

    # Rotate the motion to be facing postive Z.
    r_hip, l_hip, sdr_r, sdr_l = [11, 16, 5, 8]
    across1 = root_pos_init[:, r_hip] - root_pos_init[:, l_hip] # (B, 3)
    across2 = root_pos_init[:, sdr_r] - root_pos_init[:, sdr_l] # (B, 3)
    across = across1 + across2 # (B, 3)
    across = across / torch.sqrt((across ** 2).sum(dim=-1)).unsqueeze(-1) # (B, 3)

    # forward (B, 3), rotate around y-axis
    fw = torch.tensor([[0, 1, 0]]).float().expand(B, 3)
    forward_init = torch.cross(fw, across, dim=-1) # (B, 3)
    # forward (B, 3)
    forward_init = forward_init / torch.sqrt((forward_init ** 2).sum(dim=-1)).unsqueeze(-1) # (B, 3)

    target = torch.tensor([[0, 0, 1]]).float().expand(B, 3) # (B, 3)
    root_quat_init = qbetween(forward_init, target) # (B, 4)
    root_quat_init = root_quat_init.unsqueeze(1).unsqueeze(2) # (B, 1, 1, 4)
    root_quat_init = root_quat_init.expand(B, T, J, 4) # (B, T, J, 4)
    motion = qrot(root_quat_init, motion) # (B, T, J, 3)
    return motion


class LazyMASDataset(Dataset):
    def __init__(self, base_dir, sequence_length=None, num_samples_per_file=640, cache_size=10, shuffle=True, w_vectorizer=None, norm_params=None):
        self.base_dir = base_dir
        self.sequence_length = sequence_length
        self.w_vectorizer = w_vectorizer  # Required for HumanML evaluation format
        self.norm_params = norm_params
        assert os.path.exists(base_dir), f"MAS data directory does not exist: {base_dir}"
        assert os.path.isdir(base_dir), f"MAS data directory is not a directory: {base_dir}"
        assert len(os.listdir(base_dir)) > 0, f"MAS data directory is empty: {base_dir}"
        self.seeds = list(map(lambda file_name: int(re.findall("\d+", file_name)[0]), os.listdir(base_dir)))
        assert len(self.seeds) >= MIN_SEEDS, f"Found only {len(self.seeds)} files in {base_dir}, expecting at least {MIN_SEEDS}"
        if shuffle:
            np.random.shuffle(self.seeds)

        self.index = [
            (seed, i)
            for seed in self.seeds
            for i in range(num_samples_per_file)
        ]
        self.cache_size = cache_size
        self.file_cache = OrderedDict()  # seed -> data

    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx):
        seed, i = self.index[idx]

        # Move to front if in cache
        if seed in self.file_cache:
            data = self.file_cache.pop(seed)
            self.file_cache[seed] = data
        else:
            # Load and evict if over cache size
            path = os.path.join(self.base_dir, f"mas_seed_{seed}_results.npy")
            data = np.load(path, allow_pickle=True).item()
            self.file_cache[seed] = data
            if len(self.file_cache) > self.cache_size:
                self.file_cache.popitem(last=False)  # remove oldest
            
        data = self.file_cache[seed]
        motion = torch.tensor(data["motions"][i], dtype=torch.float32)
        length = int(data["model_kwargs"]["y"]["lengths"][i])
        text = data["model_kwargs"]["y"]["text"][i]
        # Padd to sequence length
        if self.sequence_length is not None:
            if self.sequence_length - motion.shape[0] > 0:
                motion = torch.cat([
                    motion,
                    torch.zeros(self.sequence_length - motion.shape[0], *motion.shape[1:], device=motion.device)
                ], dim=0)
            else:
                print("Warning: Truncating motion to sequence length",
                    motion.shape, self.sequence_length)
                motion = motion[:self.sequence_length]
        
        # Prepare HumanML evaluation format if w_vectorizer is provided
        humanml_eval_batch = None
        if self.w_vectorizer is not None:
            # Extract tokens from text (assuming tokens are stored in the data)
            # If tokens are not available, we need to tokenize the text
            tokens_text = text.split("#")[1].strip()
            tokens = tokens_text.split(' ')
            # Apply HumanML-style fixed-length padding/cropping
            max_text_len = 20  # Standard HumanML max text length
            tokens = tokens[:max_text_len]
            tokens = ['sos/OTHER'] + tokens + ['eos/OTHER']
            sent_len = len(tokens)
            tokens = tokens + ['unk/OTHER'] * (max_text_len + 2 - sent_len)    
            # Convert motion to numpy for HumanML format
            motion_np = motion.cpu().numpy()

            # Generate word embeddings and POS one-hots
            pos_one_hots = []
            word_embeddings = []
            for token in tokens:
                try:
                    word_emb, pos_oh = self.w_vectorizer[token]
                    pos_one_hots.append(pos_oh[None, :])
                    word_embeddings.append(word_emb[None, :])
                except (KeyError, TypeError):
                    # Fallback for unknown tokens or invalid token format
                    print(f"Warning: Unknown token '{token}', using zero vectors")
                    pos_one_hots.append(np.zeros((1, 15)))  # 15 POS tags
                    word_embeddings.append(np.zeros((1, 300)))  # 300-dim embeddings
            
            pos_one_hots = np.concatenate(pos_one_hots, axis=0)
            word_embeddings = np.concatenate(word_embeddings, axis=0)
            
            # Create HumanML evaluation batch format
            humanml_eval_batch = (
                word_embeddings,  # (22, 300) - fixed length
                pos_one_hots,     # (22, 15) - fixed length
                text,             # str - caption
                sent_len,         # int - actual sentence length (before padding)
                motion_np,        # (max_len, 263) - motion data
                length,           # int - m_length
                '_'.join(tokens),  # str - tokens string
                self.norm_params
            )
        return motion, length, text, seed, humanml_eval_batch


def mas_collate_fn(batch):
    # batch: list of (motion, length, text, seed)
    motions, lengths, texts, _ = zip(*batch)

    lengths = torch.tensor(lengths, dtype=torch.long)
    max_len = lengths.max().item()
    
    # Truncate each motion to the new max_len
    trimmed_motions = [
        motion[:max_len] for motion in motions
    ]
    motions = torch.stack(trimmed_motions, dim=0)  # (B, T, J, 3)

    model_kwargs = {
        "y": {
            "lengths": lengths,
            "text": list(texts),
        }
    }

    return motions, model_kwargs


def humanml_eval_collate_fn(batch):
    # batch: list of (motion, length, text, seed, humanml_eval_batch)
    # Extract the HumanML evaluation batches
    humanml_eval_batches = [item[4] for item in batch if item[4] is not None]
    
    if not humanml_eval_batches:
        raise ValueError("No HumanML evaluation batches found. Make sure w_vectorizer is provided to LazyMASDataset.")
    
    # Sort by sent_len (index 3) like the standard collate_fn does
    humanml_eval_batches.sort(key=lambda x: x[3], reverse=True)
    
    # All sequences now have fixed length (22), so we can directly stack
    word_embeddings = torch.stack([torch.tensor(item[0]) for item in humanml_eval_batches])
    pos_one_hots = torch.stack([torch.tensor(item[1]) for item in humanml_eval_batches])
    captions = [item[2] for item in humanml_eval_batches]
    sent_lens = torch.tensor([item[3] for item in humanml_eval_batches], dtype=torch.long)
    
    # Ensure motion data has correct shape (batch_size, seq_len, features)
    motion_tensors = []
    for item in humanml_eval_batches:
        motion = item[4]
        motion *= __global_bbox_scale_hml / calculate_bbox_scale(motion)
        motion_tensor = torch.tensor(motion)  # (seq_len, njoints, 3)
        motion_tensor = motion_tensor.unsqueeze(0)
        motion_tensors.append(motion_tensor)
    motions = torch.cat(motion_tensors, dim=0)  # (batch_size, seq_len, njoints, 3)
    motions = torch_normlize_motion(motions)
    motions, _, _, _ = process_file_torch(motions) # (batch_size, seq_len, features)
    mean, std = item[7]
    motions = (motions - mean) / std

    m_lens = torch.tensor([item[5] for item in humanml_eval_batches], dtype=torch.long)
    tokens_strs = [item[6] for item in humanml_eval_batches]
    
    return word_embeddings, pos_one_hots, captions, sent_lens, motions, m_lens, tokens_strs


def get_mas_loader(args, collate_fn=mas_collate_fn, w_vectorizer=None, norm_params=None):
    assert args.mas_data_dir is not None, "MAS data directory must be specified"
    dataset = LazyMASDataset(
        base_dir=args.mas_data_dir,
        w_vectorizer=w_vectorizer,  # Pass w_vectorizer for HumanML evaluation support
        norm_params=norm_params,
    )
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=0,
    )


"""
# Example check for dataset validity
import os
import numpy as np
from tqdm import tqdm
data_path = "/home/amir.mann/VideoMDM/eval/VideoMDM/mas/synthetic_2025.09.03/generated"
expected_keys = ["motions", "model_kwargs"]
samples_per_file_counts = {}
for f in tqdm(os.listdir(data_path)):
    if f.endswith(".npy"):
        data = np.load(os.path.join(data_path, f), allow_pickle=True).item()
        if not all(key in data for key in expected_keys):
            print(f"Warning: {f} does not contain all expected keys")
            continue
        samples = data["motions"].shape[0]
        if samples < 640:
            print(f"Warning: {f} has less than 640 samples")
            continue
        if samples not in samples_per_file_counts:
            samples_per_file_counts[samples] = 0
        samples_per_file_counts[samples] += 1
print(f"Samples per file counts: {samples_per_file_counts}")

# Clean up rouge sample Warning: mas_seed_0_results.npy has less than 640 samples

os.remove(os.path.join(data_path, "mas_seed_0_results.npy"))
"""