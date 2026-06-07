from torch.utils.data import DataLoader
from data_loaders.tensors import collate as all_collate
from data_loaders.tensors import t2m_collate, video_t2m_collate
from data_loaders.humanml.data.dataset import collate_fn as t2m_eval_collate

def get_dataset_class(name):
    if name == "amass":
        from .amass import AMASS
        return AMASS
    elif name == "uestc":
        from .a2m.uestc import UESTC
        return UESTC
    elif name == "humanact12":
        from .a2m.humanact12poses import HumanAct12Poses
        return HumanAct12Poses
    elif name == "humanml":
        from data_loaders.humanml.data.dataset import HumanML3D
        return HumanML3D
    elif name == "egoexo":
        from data_loaders.humanml.data.dataset import EgoExoDataset
        return EgoExoDataset
    elif name == "video_mdm_synthetic":
        from data_loaders.humanml.data.dataset import SyntheticVideoMDMDataset
        return SyntheticVideoMDMDataset
    elif name == "video_mdm_synthetic_mvlift":
        from data_loaders.humanml.data.dataset import SyntheticVideoMDMDataset
        class SyntheticVideoMDMDatasetMVLift(SyntheticVideoMDMDataset):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, datapath='./dataset/video_mdm_synthetic_mvlift_opt.txt', **kwargs)
        return SyntheticVideoMDMDatasetMVLift
    elif name == "fit3d":
        from data_loaders.humanml.data.dataset import Fit3DDataset
        return Fit3DDataset
    elif name == "fit3d_mvlift":
        from data_loaders.humanml.data.dataset import Fit3DDataset
        class Fit3DDatasetMVLift(Fit3DDataset):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, datapath='./dataset/fit3d_mvlift_opt.txt', **kwargs)
        return Fit3DDatasetMVLift
    elif name == "nba":
        from data_loaders.humanml.data.dataset import NBADataset
        return NBADataset
    elif name == "kit":
        from data_loaders.humanml.data.dataset import KIT
        return KIT
    else:
        raise ValueError(f'Unsupported dataset name [{name}]')

def get_collate_fn(name, hml_mode='train'):
    if hml_mode == 'gt':
        return t2m_eval_collate
    if name in ["humanml", "kit"]:
        return t2m_collate
    if name in ["egoexo", "video_mdm_synthetic", "video_mdm_synthetic_mvlift","nba", "fit3d", "fit3d_mvlift"]:
        if hml_mode == 'text_only':
            return t2m_collate
        return video_t2m_collate
    else:
        return all_collate


def get_dataset(name, num_frames, split='train', hml_mode='train', device='cuda'):
    DATA = get_dataset_class(name)
    if name in ["humanml", "kit", "egoexo", "video_mdm_synthetic", "video_mdm_synthetic_mvlift", "nba", "fit3d", "fit3d_mvlift"]:
        dataset = DATA(split=split, num_frames=num_frames, mode=hml_mode, device=device)
    else:
        dataset = DATA(split=split, num_frames=num_frames, device=device)
    return dataset


def get_datasets_loader(name, batch_size, num_frames, split='train', hml_mode='train', device='cuda', num_workers=2):
    if "," in split or "," in name:
        splits = split.split(",")
        names = name.split(",")
        num_datasets = max(len(splits), len(names))
        if len(splits) == 1:
            splits = [split] * num_datasets
        if len(names) == 1:
            names = [name] * num_datasets
        assert len(names) == len(splits), f"Got diferent number of splits ({split}) and names ({name}), {len(names)} != {len(splits)}"
        collates = [get_collate_fn(name, hml_mode) for name in names]
        datasets = [get_dataset(name, num_frames, split=split, hml_mode=hml_mode, device=device) for split, name in zip(splits, names)]
    else:
        datasets = [get_dataset(name, num_frames, split, hml_mode, device=device)]
        collates = [get_collate_fn(name, hml_mode)]
    

    loaders = [
        DataLoader(
            dataset, batch_size=batch_size, shuffle=True,
            num_workers=num_workers, drop_last=True, collate_fn=collate
        ) for dataset, collate in zip(datasets, collates)
    ]

    return loaders

def get_dataset_loader(name, batch_size, num_frames, split='train', hml_mode='train', device='cuda', num_workers=2):
    dataset = get_dataset(name, num_frames, split, hml_mode, device=device)
    collate = get_collate_fn(name, hml_mode)

    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, drop_last=True, collate_fn=collate
    )

    return loader

def get_hml_mode(args):
    if args.pnp_to_find_cameras:
        return 'pnptrain'
    else:
        return 'train'