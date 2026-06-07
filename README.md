# VideoMDM

> ⚠️ **This is a temporary README** — a complete one will follow.

Official code for **"VideoMDM: Towards 3D Human Motion Generation From 2D Supervision"**
([project page](https://videomdm.github.io/) · [paper](https://videomdm.github.io/VideoMDM.pdf)).

VideoMDM trains 3D human-motion diffusion models (text-to-motion) using only **2D pose
supervision** extracted from monocular video — no 3D ground truth. It is built on top of
MDM (Human Motion Diffusion Model).

## Usage

This repository follows the structure and conventions of the upstream **MDM** repository by
Guy Tevet. For environment setup, dependency/asset downloads, and the general
training / sampling / evaluation workflow, please follow the upstream instructions there:

**https://github.com/GuyTevet/motion-diffusion-model**

(A VideoMDM-specific README with the exact commands and added options will be added later.)

## Pretrained checkpoints

Our trained checkpoints for the three datasets in the paper are hosted on the Hugging Face Hub:

**https://huggingface.co/AmirMann/VideoMDM**

| Dataset | Lifter / teacher | Folder |
|---|---|---|
| HumanML3D | MVLift   | `HUMANML3D_VIDEOMDM_ON_MVLIFT/` |
| Fit3D     | WHAM     | `FIT3D_VIDEOMDM_ON_WHAM/`       |
| NBA       | ElePose  | `NBA_VIDEOMDM_ON_ELEPOSE/`      |

Each folder contains the model weights (`model*.pt`) and its `args.json`. Download with:

```bash
pip install -U huggingface_hub
hf download AmirMann/VideoMDM --local-dir ./save
```

This places the checkpoints under `./save/<FOLDER>/`. To use one, point `--model_path` at its
`.pt` file (the matching `args.json` in the same folder is loaded automatically), e.g.:

```bash
python -m sample.generate --model_path ./save/HUMANML3D_VIDEOMDM_ON_MVLIFT/model000600091.pt
```

## Data

The data-processing code and the processed datasets are **not** part of this initial release.
They will be made available later, subject to the licenses of the underlying datasets
(e.g. AMASS and Fit3D).

## License

Released under the [MIT License](LICENSE). Note that dependencies and datasets (CLIP,
SMPL/SMPL-X, AMASS, Fit3D, NBA, etc.) carry their own licenses that must also be followed.
