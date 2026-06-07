import numpy as np
import torch
import torch.nn as nn


class MDM_2D(nn.Module):
    def __init__(
        self,
        njoints,
        nfeats,
        num_actions,
        latent_dim=256,
        ff_size=1024,
        num_layers=8,
        num_heads=4,
        dropout=0.1,
        activation="gelu",
        clip_dim=512,
        arch="trans_enc",
        emb_trans_dec=False,
        clip_version=None,
        cond_mode="no_cond",
        cond_mask_prob=0.0,
        distilation_solve_ode=False,
        distilation_use_gt_camera=False,
        cfg_for_distilation=0.0,
        **kargs,
    ):
        super().__init__()

        self.njoints = njoints
        self.nfeats = nfeats
        self.num_actions = num_actions
        self.latent_dim = latent_dim
        self.ff_size = ff_size
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.dropout = dropout
        self.activation = activation
        self.clip_dim = clip_dim
        self.cond_mode = cond_mode
        self.cond_mask_prob = cond_mask_prob
        self.arch = arch
        self.input_feats = self.njoints * self.nfeats
        self.gru_emb_dim = self.latent_dim if self.arch == "gru" else 0
        self.distilation_solve_ode = distilation_solve_ode
        self.distilation_use_gt_camera = distilation_use_gt_camera
        self.cfg_for_distilation = cfg_for_distilation
        
        self.input_process = InputProcess(self.input_feats + self.gru_emb_dim, self.latent_dim)

        self.sequence_pos_encoder = PositionalEncoding(self.latent_dim, self.dropout)
        self.emb_trans_dec = emb_trans_dec
        if self.arch == "trans_enc":
            seqTransEncoderLayer = nn.TransformerEncoderLayer(d_model=self.latent_dim, nhead=self.num_heads, dim_feedforward=self.ff_size, dropout=self.dropout, activation=self.activation)
            self.seqTransEncoder = nn.TransformerEncoder(seqTransEncoderLayer, num_layers=self.num_layers)

        elif self.arch == "trans_dec":
            seqTransDecoderLayer = nn.TransformerDecoderLayer(d_model=self.latent_dim, nhead=self.num_heads, dim_feedforward=self.ff_size, dropout=self.dropout, activation=activation)
            self.seqTransDecoder = nn.TransformerDecoder(seqTransDecoderLayer, num_layers=self.num_layers)

        elif self.arch == "gru":
            self.gru = nn.GRU(self.latent_dim, self.latent_dim, num_layers=self.num_layers, batch_first=True)

        else:
            raise ValueError("Please choose a valid architecture [trans_enc, trans_dec, gru]")

        self.embed_timestep = TimestepEmbedder(self.latent_dim, self.sequence_pos_encoder)

        if self.cond_mode != "no_cond":
            if "text" in self.cond_mode:
                self.embed_text = nn.Linear(self.clip_dim, self.latent_dim)
                print("Loading CLIP...")
                self.clip_version = clip_version
                self.clip_model = self.load_and_freeze_clip(clip_version)
            if "action" in self.cond_mode:
                self.embed_action = EmbedAction(self.num_actions, self.latent_dim)

        self.output_process = OutputProcess(self.input_feats, self.latent_dim, self.njoints, self.nfeats)
        self.clip_cache = {}

    def parameters_wo_clip(self):
        return [p for name, p in self.named_parameters() if not name.startswith("clip_model.")]

    def load_and_freeze_clip(self, clip_version):
        import clip

        clip_model, clip_preprocess = clip.load(clip_version, device="cpu", jit=False)  # Must set jit=False for training
        clip.model.convert_weights(clip_model)  # Actually this line is unnecessary since clip by default already on float16

        # Freeze CLIP weights
        clip_model.eval()
        for p in clip_model.parameters():
            p.requires_grad = False
        
        def encode_text(raw_text):
            device = next(self.parameters()).device
            # Check if we need to re-run the model
            if any(t not in self.clip_cache for t in raw_text):
                # Tokenize entire batch once
                default_ctx = 77
                max_len = 50
                ctx = max_len + 2
                assert ctx < default_ctx

                tokens = clip.tokenize(raw_text, context_length=ctx, truncate=True).to(device)
                pad = torch.zeros(
                    tokens.size(0),
                    default_ctx - ctx,
                    dtype=tokens.dtype,
                    device=device
                )
                tokens = torch.cat([tokens, pad], dim=1)

                # Encode once
                embs = self.clip_model.encode_text(tokens).float()  # (B, D)
                # Cache only the new ones
                for txt, emb in zip(raw_text, embs):
                    self.clip_cache[txt] = emb.detach()
            # Return a stacked batch of embeddings
            return torch.stack([self.clip_cache[t] for t in raw_text], dim=0)

        self.encode_text = encode_text
        return clip_model

    def mask_cond(self, cond, force_mask=False):
        bs, d = cond.shape
        if force_mask:
            return torch.zeros_like(cond)
        elif self.training and self.cond_mask_prob > 0.0:
            mask = torch.bernoulli(torch.ones(bs, device=cond.device) * self.cond_mask_prob).view(bs, 1)  # 1-> use null_cond, 0-> use real cond
            return cond * (1.0 - mask)
        else:
            return cond

    def get_seq_mask(self, xseq, y):
        lengths = y["lengths"].to(xseq.device).view(-1, 1)  # [bs, 1]
        positions = torch.arange(xseq.size(0), device=xseq.device).view(1, -1)  # [1, seqlen+1]
        return positions > lengths  # [seqlen+1, bs]

    def forward(self, x, timesteps, y=None):
        """
        x: [batch_size, njoints, nfeats, max_frames], denoted x_t in the paper
        timesteps: [batch_size] (int)
        """
        # Check if CFG is enabled
        cfg_scale = self.cfg_for_distilation
        
        if cfg_scale > 0.0 and y is not None and "uncond" not in y:
            # CFG: Call forward twice and combine results
            # 1. Conditional prediction (normal)
            conditional_output = self._forward_impl(x, timesteps, y)
            
            # 2. Unconditional prediction (force mask all conditions)
            y_uncond = y.copy() if y is not None else {}
            y_uncond["uncond"] = True
            unconditional_output = self._forward_impl(x, timesteps, y_uncond)
            
            # 3. Apply CFG formula: uncond + cfg_scale * (cond - uncond)
            return unconditional_output + cfg_scale * (conditional_output - unconditional_output)
        else:
            # No CFG, use normal forward pass
            return self._forward_impl(x, timesteps, y)

    def _forward_impl(self, x, timesteps, y=None):
        """
        x: [batch_size, njoints, nfeats, max_frames], denoted x_t in the paper
        timesteps: [batch_size] (int)
        """
        bs, njoints, nfeats, nframes = x.shape
        emb = self.embed_timestep(timesteps)  # [1, bs, d]

        force_mask = y.get("uncond", False)
        if "text" in self.cond_mode:
            enc_text = self.encode_text(y["text"])
            emb += self.embed_text(self.mask_cond(enc_text, force_mask=force_mask))
        if "action" in self.cond_mode:
            action_emb = self.embed_action(y["action"])
            emb += self.mask_cond(action_emb, force_mask=force_mask)

        if self.arch == "gru":
            x_reshaped = x.reshape(bs, njoints * nfeats, 1, nframes)
            emb_gru = emb.repeat(nframes, 1, 1)  # [#frames, bs, d]
            emb_gru = emb_gru.permute(1, 2, 0)  # [bs, d, #frames]
            emb_gru = emb_gru.reshape(bs, self.latent_dim, 1, nframes)  # [bs, d, 1, #frames]
            x = torch.cat((x_reshaped, emb_gru), axis=1)  # [bs, d+joints*feat, 1, #frames]

        x = self.input_process(x)

        if self.arch == "trans_enc":
            # adding the timestep embed
            xseq = torch.cat((emb, x), axis=0)  # [seqlen+1, bs, d]
            xseq = self.sequence_pos_encoder(xseq)  # [seqlen+1, bs, d]
            output = self.seqTransEncoder(xseq, src_key_padding_mask=self.get_seq_mask(xseq, y))[1:]  # [seqlen, bs, d]

        elif self.arch == "trans_dec":
            if self.emb_trans_dec:
                xseq = torch.cat((emb, x), axis=0)
            else:
                xseq = x
            xseq = self.sequence_pos_encoder(xseq)  # [seqlen+1, bs, d]
            if self.emb_trans_dec:
                output = self.seqTransDecoder(tgt=xseq, memory=emb)[1:]  # [seqlen, bs, d] # FIXME - maybe add a causal mask
            else:
                output = self.seqTransDecoder(tgt=xseq, memory=emb)
        elif self.arch == "gru":
            xseq = x
            xseq = self.sequence_pos_encoder(xseq)  # [seqlen, bs, d]
            output, _ = self.gru(xseq)

        output = self.output_process(output)  # [bs, njoints, nfeats, nframes]
        return output


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)

        self.register_buffer("pe", pe)

    def forward(self, x):
        # not used in the final model
        x = x + self.pe[: x.shape[0], :]
        return self.dropout(x)


class TimestepEmbedder(nn.Module):
    def __init__(self, latent_dim, sequence_pos_encoder):
        super().__init__()
        self.latent_dim = latent_dim
        self.sequence_pos_encoder = sequence_pos_encoder

        time_embed_dim = self.latent_dim
        self.time_embed = nn.Sequential(
            nn.Linear(self.latent_dim, time_embed_dim),
            nn.SiLU(),
            nn.Linear(time_embed_dim, time_embed_dim),
        )

    def forward(self, timesteps):
        return self.time_embed(self.sequence_pos_encoder.pe[timesteps]).permute(1, 0, 2)


class InputProcess(nn.Module):
    def __init__(self, input_feats, latent_dim):
        super().__init__()
        self.input_feats = input_feats
        self.latent_dim = latent_dim
        self.poseEmbedding = nn.Linear(self.input_feats, self.latent_dim)

    def forward(self, x):
        bs, njoints, nfeats, nframes = x.shape
        x = x.permute((3, 0, 1, 2)).reshape(nframes, bs, njoints * nfeats)

        x = self.poseEmbedding(x)  # [seqlen, bs, d]
        return x


class OutputProcess(nn.Module):
    def __init__(self, input_feats, latent_dim, njoints, nfeats):
        super().__init__()
        self.input_feats = input_feats
        self.latent_dim = latent_dim
        self.njoints = njoints
        self.nfeats = nfeats
        self.poseFinal = nn.Linear(self.latent_dim, self.input_feats)

    def forward(self, output):
        nframes, bs, d = output.shape
        output = self.poseFinal(output)  # [seqlen, bs, 150]
        output = output.reshape(nframes, bs, self.njoints, self.nfeats)
        output = output.permute(1, 2, 3, 0)  # [bs, njoints, nfeats, nframes]
        return output


class EmbedAction(nn.Module):
    def __init__(self, num_actions, latent_dim):
        super().__init__()
        self.action_embedding = nn.Parameter(torch.randn(num_actions, latent_dim))

    def forward(self, input):
        idx = input[:, 0].to(torch.long)  # an index array must be long
        output = self.action_embedding[idx]
        return output
