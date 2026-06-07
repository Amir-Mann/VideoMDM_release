import os
from argparse import Namespace
import re
from os.path import join as pjoin
from data_loaders.humanml.utils.word_vectorizer import POS_enumerator


def is_float(numStr):
    flag = False
    numStr = str(numStr).strip().lstrip('-').lstrip('+')    # 去除正数(+)、负数(-)符号
    try:
        reg = re.compile(r'^[-+]?[0-9]+\.[0-9]+$')
        res = reg.match(str(numStr))
        if res:
            flag = True
    except Exception as ex:
        print("is_float() - error: " + str(ex))
    return flag


def is_number(numStr):
    flag = False
    numStr = str(numStr).strip().lstrip('-').lstrip('+')    # 去除正数(+)、负数(-)符号
    if str(numStr).isdigit():
        flag = True
    return flag


def get_opt(opt_path, device):
    opt = Namespace()
    opt_dict = vars(opt)

    skip = ('-------------- End ----------------',
            '------------ Options -------------',
            '\n')
    print('Reading', opt_path)
    with open(opt_path) as f:
        for line in f:
            if line.strip() not in skip:
                # print(line.strip())
                key, value = line.strip().split(': ')
                if value in ('True', 'False'):
                    opt_dict[key] = bool(value)
                elif is_float(value):
                    opt_dict[key] = float(value)
                elif is_number(value):
                    opt_dict[key] = int(value)
                else:
                    opt_dict[key] = str(value)

    # print(opt)
    opt_dict['which_epoch'] = 'latest'
    opt.save_root = pjoin(opt.checkpoints_dir, opt.dataset_name, opt.name)
    opt.model_dir = pjoin(opt.save_root, 'model')
    opt.meta_dir = pjoin(opt.save_root, 'meta')

    if opt.dataset_name == 't2m':
        opt.data_root = './dataset/HumanML3D'
        opt.motion_dir = pjoin(opt.data_root, 'new_joint_vecs')
        opt.text_dir = pjoin(opt.data_root, 'texts')
        opt.joints_num = 22
        opt.dim_pose = 263
        opt.max_motion_length = 196
    elif opt.dataset_name == 'video_t2m':
        opt.motion_dir = pjoin(opt.data_root, 'new_joint_vecs')
        opt.text_dir = pjoin(opt.data_root, 'texts')
        opt.camera_dir = pjoin(opt.data_root, 'cameras')
        opt.motion_2d_dir = pjoin(opt.data_root, '2d_motions')
        opt.scores_2d_dir = pjoin(opt.data_root, '2d_scores')
        opt.joints_num = 22
        opt.dim_pose = 263
        opt.max_motion_length = 196
    elif opt.dataset_name == 'video_t2m_mvlift_hook':
        if "./dataset/video_mdm_synthetic" == opt.data_root:
            Y_CORRECTED = False
            if Y_CORRECTED:
                opt.dataset_name += "_y_correct"
                opt.motion_dir = pjoin('mvlift_release', 'lifted_motions_y_corrected')
            else:
                opt.motion_dir = pjoin('mvlift_release', 'lifted_motions')
        elif "./dataset/fit3d" == opt.data_root:
            opt.motion_dir = pjoin("./dataset/fit3d", "mvlift_lifted_vecs")
        else:
            raise ValueError(f"Unknown data_root {opt.data_root} for video_t2m_mvlift_hook dataset. Expected either './dataset/video_mdm_synthetic' or './dataset/fit3d'.")
        assert os.path.exists(opt.motion_dir), f"Motion directory {opt.motion_dir} not found. This is the directory where the lifted motions from MVLift should be stored. Passed as --data_dir for video_t2m_mvlift_hook dataset."
        opt.text_dir = pjoin(opt.data_root, 'texts')
        opt.camera_dir = pjoin(opt.data_root, 'cameras')
        opt.motion_2d_dir = pjoin(opt.data_root, '2d_motions')
        opt.scores_2d_dir = pjoin(opt.data_root, '2d_scores')
        opt.joints_num = 22
        opt.dim_pose = 263
        opt.max_motion_length = 196
    elif opt.dataset_name == 'kit':
        opt.data_root = './dataset/KIT-ML'
        opt.motion_dir = pjoin(opt.data_root, 'new_joint_vecs')
        opt.text_dir = pjoin(opt.data_root, 'texts')
        opt.joints_num = 21
        opt.dim_pose = 251
        opt.max_motion_length = 196
    else:
        raise KeyError('Dataset not recognized')

    opt.dim_word = 300
    opt.num_classes = 200 // opt.unit_length
    opt.dim_pos_ohot = len(POS_enumerator)
    opt.is_train = False
    opt.is_continue = False
    opt.device = device

    return opt