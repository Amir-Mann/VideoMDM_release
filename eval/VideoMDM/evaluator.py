import os
import re
import json
import argparse
import numpy as np
import torch
from utils.math_utils import perspective_projection_batch_angles
from diffusion.video_mdm_diffusion import norm_motion_2d
from torch import nn
from model.mdm import InputProcess, OutputProcess, PositionalEncoding
from data_loaders.humanml.networks.evaluator_wrapper import EvaluatorMDMWrapper


described_names = set()
def describe(x, name="var"):
    if name in described_names:
        return
    described_names.add(name)
    print(f"{name}: shape={getattr(x, 'shape', None)}, type={type(x)}", end="")
    if isinstance(x, torch.Tensor):
        print(f", device={x.device}")
    else:
        print()


def get_dim() -> tuple:
    """Get the number of joints and features."""
    # Assuming a fixed skeleton for this example
    njoints = 22
    nfeats = 2
    return njoints, nfeats


def sample_vertical_angle() -> float:
    """Fixed elevation ≈ 11.25 °  (π/16 rad)."""
    return np.pi / 16


def sample_distance() -> float:
    """Fixed camera distance."""
    return 7.0


class VAE(nn.Module):
    def __init__(self, args, encoder, decoder):
        super().__init__()
        self.args = args
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, motion, m_lens):
        mu, sigma = self.encoder(motion, m_lens)
        latent = mu + torch.exp(sigma / 2) * torch.randn_like(mu)
        recon = self.decoder(latent, m_lens)
        return {"mu": mu, "sigma": sigma, "recon_motion": recon}


def get_seq_mask(lengths):
    lengths = lengths.view(-1, 1)  # [bs, 1]
    positions = torch.arange(lengths.max(), device=lengths.device).view(1, -1)  # [1, nframes+1]
    return positions >= lengths  # [nframes+1, bs]


class TransformerEncoder(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.set_arch()
        self.setup_device()

    def set_arch(self):
        self.njoints, self.nfeats = get_dim()
        self.input_dim = self.njoints * self.nfeats
        self.latent_dim, self.num_layers, self.num_heads, self.ff_size, self.dropout, self.activation = self.args.e_latent_dim, self.args.e_num_layers, self.args.e_num_heads, self.args.e_ff_size, self.args.e_dropout, self.args.e_activation

        self.input_process = InputProcess("xyz", self.input_dim, self.latent_dim)

        self.sequence_pos_encoder = PositionalEncoding(self.latent_dim, self.dropout)

        seqTransEncoderLayer = nn.TransformerEncoderLayer(d_model=self.latent_dim, nhead=self.num_heads, dim_feedforward=self.ff_size, dropout=self.dropout, activation=self.activation)
        self.seqTransEncoder = nn.TransformerEncoder(seqTransEncoderLayer, num_layers=self.num_layers)

        self.mu_query = nn.Parameter(torch.randn([self.latent_dim]))
        self.sigma_query = nn.Parameter(torch.randn([self.latent_dim]))

    def setup_device(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.to(self.device)

    def forward(self, x, m_lens):
        #describe(x, "TransformerEncoder.forward, x")
        #describe(m_lens, "TransformerEncoder.forward, m_lens")
        bs, njoints, nfeats, nframes = x.shape  # [bs, njoints, nfeats, nframes]
        assert njoints == self.njoints and nfeats == self.nfeats
        x = self.input_process(x)  # [nframes, bs, d]
        x = torch.cat((self.mu_query.expand(x[[0]].shape), self.sigma_query.expand(x[[0]].shape), x), axis=0)
        x = self.sequence_pos_encoder(x)  # [nframes+2, bs, d]

        # create a bigger mask, to allow to attend to mu and sigma
        
        mask = get_seq_mask(m_lens + 2)
        x = self.seqTransEncoder(x, src_key_padding_mask=mask)
        mu = x[0]
        logvar = x[1]

        return mu, logvar


class TransformerDecoder(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.set_arch()
        self.setup_device()

    def set_arch(self):
        self.njoints, self.nfeats = get_dim()
        self.input_dim = self.njoints * self.nfeats
        self.latent_dim, self.num_layers, self.num_heads, self.ff_size, self.dropout, self.activation = self.args.e_latent_dim, self.args.e_num_layers, self.args.e_num_heads, self.args.e_ff_size, self.args.e_dropout, self.args.e_activation

        self.output_process = OutputProcess("xyz", self.input_dim, self.latent_dim, self.njoints, self.nfeats)

        self.sequence_pos_encoder = PositionalEncoding(self.latent_dim, self.dropout)

        seqTransDecoderLayer = nn.TransformerDecoderLayer(d_model=self.latent_dim, nhead=self.num_heads, dim_feedforward=self.ff_size, dropout=self.dropout, activation=self.activation)
        self.seqTransDecoder = nn.TransformerDecoder(seqTransDecoderLayer, num_layers=self.num_layers)

    def setup_device(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.to(self.device)

    def forward(self, latent, m_lens):
        bs, latent_dim = latent.shape  # [bs, d]
        nframes = m_lens.max()
        assert latent_dim == self.latent_dim
        x = torch.zeros([nframes, bs, self.latent_dim], device=self.device)  # [nframes, bs, d]
        x = self.sequence_pos_encoder(x)  # [nframes, bs, d]

        x = self.seqTransDecoder(tgt=x, memory=latent.unsqueeze(0), tgt_key_padding_mask=get_seq_mask(m_lens))
        x = self.output_process(x)  # [nframes, bs, nfeats*njoints]

        return x

_CKPT_RE = re.compile(r"^checkpoint_(\d+)\.pth$")

def _load_args(json_path: str) -> argparse.Namespace:
    with open(json_path, "r") as f:
        cfg = json.load(f)
    return argparse.Namespace(**cfg)


def _find_latest_ckpt(path: str) -> str:
    ckpts = []
    for fname in os.listdir(path):
        m = _CKPT_RE.match(fname)
        if m:
            ckpts.append((int(m.group(1)), fname))
    assert ckpts, f"No checkpoints found in {path}"
    ckpts.sort()
    return os.path.join(path, ckpts[-1][1])   # highest step

def create_evaluator(path: str) -> VAE:
    """
    Build a VAE and load the latest checkpoint found in *path*.

    *path* must contain:
      • args.json
      • at least one checkpoint_XXXXX.pth
    """
    assert os.path.isdir(path), f"{path!r} is not a directory"

    args_file = os.path.join(path, "args.json")
    assert os.path.isfile(args_file), "args.json not found in directory"

    args = _load_args(args_file)

    # attach the path to the newest checkpoint
    args.model_path = _find_latest_ckpt(path)

    # make sure the device string is usable
    if torch.cuda.is_available():
        idx = getattr(args, "device", 0)
        args.device = f"cuda:{idx}"
    else:
        args.device = "cpu"

    # ---- build architecture --------------------------------------------------
    vae = VAE(args, TransformerEncoder(args), TransformerDecoder(args))

    # ---- load weights --------------------------------------------------------
    ckpt = torch.load(args.model_path, map_location="cpu")
    state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    vae.load_state_dict(state, strict=True)

    return vae


def _sample_hor_angle(mode: str = "uniform") -> float:
    """Pick a horizontal camera angle (radians)."""
    if mode == "uniform":
        return np.random.uniform(-np.pi, np.pi)
    if mode == "side":
        return np.pi / 2
    if mode == "hybrid":
        return np.random.uniform(np.pi / 2 - np.pi / 4, np.pi / 2 + np.pi / 4)
    raise ValueError(f"Unknown angle-mode '{mode}'")


class Evaluator(EvaluatorMDMWrapper):
    """
    Text pipeline = original HumanML3D wrapper  
    Motion pipeline = our 2-D VAE encoder
    """

    def __init__(
        self,
        dataset: str,
        device: str,
        vae_checkpoint_dir: str,
        angle_mode="uniform",
    ):
        #super().__init__(dataset, device)                     # keeps text encoder
        self.vae = create_evaluator(vae_checkpoint_dir).to(device).eval()
        self.angle_mode = angle_mode
        self.device = device

    # ------------------------------------------------------------------
    # override motion embedding only – base class keeps text routines
    # ------------------------------------------------------------------
    @torch.no_grad()
    def get_motion_embeddings(self, motion, m_lens):
        """
        motion  : (B, T, J, 3) **or** (B, T, J, 2)
        m_lens  : (B,)  tensor of sequence lengths
        Returns : (B, D) motion embeddings from the VAE encoder
        """
        motion = motion.to(self.device)
        # C = 3 → project,  C = 2 → already 2-D
        if motion.shape[-1] == 3:
            motion_2d = self._project(motion)          # (B, T, J, 2)
        else:
            motion_2d = motion                         # (B, T, J, 2)

        m_lens = m_lens.to(self.device)        
        motion_2d = norm_motion_2d(motion_2d, m_lens)
        #describe(motion_2d, "Evaluator.get_motion_embeddings, motion_2d")
        #describe(m_lens, "Evaluator.get_motion_embeddings, m_lens")

        return self.vae.encoder(motion_2d.permute(0, 2, 3, 1), m_lens)[0] # Returns mu, sigma. Only mu is used.

    def _project(self, motion: torch.Tensor) -> torch.Tensor:
        """Project 3-D motion to 2-D from a random camera."""
        b = motion.shape[0]
        hor = torch.tensor([_sample_hor_angle(self.angle_mode) for _ in range(b)], dtype=torch.float32, device=motion.device)
        ver = torch.tensor([sample_vertical_angle() for _ in range(b)], dtype=torch.float32, device=motion.device)
        dist = torch.tensor([sample_distance() for _ in range(b)], dtype=torch.float32, device=motion.device)
        proj_2d, _ = perspective_projection_batch_angles(
            motion.permute(0, 2, 3, 1), hor, ver, dist
        )   # → (B, J, 2, T)

        # reorder to (B, T, J, 2)
        proj_2d = proj_2d.permute(0, 3, 1, 2)
        return proj_2d
