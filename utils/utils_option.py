import os
from collections import OrderedDict
from datetime import datetime
import json
import re
import glob
import sys
import datetime
import logging


'''
# --------------------------------------------
# https://github.com/cszn
# https://github.com/xinntao/BasicSR
# --------------------------------------------
'''

# --------------------------------------------
# report the training process in a log file
# --------------------------------------------
def log(*args, **kwargs):
    print(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S:"), *args, **kwargs)

def logger_info(logger_name, log_path='default_logger.log'):
    log = logging.getLogger(logger_name)
    if log.hasHandlers():
        print('LogHandlers exist!')
    else:
        print('LogHandlers setup!')
        level = logging.INFO
        formatter = logging.Formatter('%(asctime)s.%(msecs)03d : %(message)s', datefmt='%y-%m-%d %H:%M:%S')
        fh = logging.FileHandler(log_path, mode='a')
        fh.setFormatter(formatter)
        log.setLevel(level)
        log.addHandler(fh)
        # sh = logging.StreamHandler()
        # sh.setFormatter(formatter)
        # log.addHandler(sh)

# --------------------------------------------
# get time
# --------------------------------------------
def get_timestamp():
    return datetime.datetime.now().strftime('_%y%m%d_%H%M%S')

# --------------------------------------------
# process opt
# --------------------------------------------
def parse(opt_path, is_train=True):
    # ----------------------------------------
    # remove comments starting with '//'
    # ----------------------------------------
    json_str = ''
    with open(opt_path, 'r') as f:
        for line in f:
            line = line.split('//')[0] + '\n'
            json_str += line

    # ----------------------------------------
    # initialize opt
    # ----------------------------------------
    opt = json.loads(json_str, object_pairs_hook=OrderedDict)
    opt['opt_path'] = opt_path
    opt['is_train'] = is_train

    # ----------------------------------------
    # datasets
    # ----------------------------------------
    for phase, dataset in opt['datasets'].items():
        phase = phase.split('_')[0]
        dataset['phase'] = phase
        dataset['n_channels'] = opt['n_channels']  # broadcast
        if phase == 'train':
            dataset['dataroot_Label'] = os.path.join(opt['dataroot'], 'train_Label')
            dataset['dataroot_Input'] = os.path.join(opt['dataroot'], 'train_Input')
        if phase == 'test':
            dataset['dataroot_Label'] = os.path.join(opt['dataroot'], 'val_Label')
            dataset['dataroot_Input'] = os.path.join(opt['dataroot'], 'val_Input')
        if 'dataroot_Label' in dataset and dataset['dataroot_Label'] is not None:
            dataset['dataroot_Label'] = os.path.expanduser(dataset['dataroot_Label']) # use 'expanduser' to get the ~ path
        if 'dataroot_Input' in dataset and dataset['dataroot_Input'] is not None:
            dataset['dataroot_Input'] = os.path.expanduser(dataset['dataroot_Input'])

    # ----------------------------------------
    # path
    # ----------------------------------------
    for key, path in opt['path'].items():
        if path and key in opt['path']:
            opt['path'][key] = os.path.expanduser(path)

    path_task = os.path.join(opt['path']['root'], opt['task'])
    opt['path']['task'] = path_task
    opt['path']['log'] = path_task
    opt['path']['options'] = os.path.join(path_task, 'options')

    if is_train:
        opt['path']['models'] = os.path.join(path_task, 'models')
        opt['path']['images'] = os.path.join(path_task, 'images')
    else:  # test
        opt['path']['images'] = os.path.join(path_task, 'test_images')

    # ----------------------------------------
    # GPU devices
    # ----------------------------------------
    # gpu_list = ','.join(str(x) for x in opt['gpu_ids'])
    # os.environ['CUDA_VISIBLE_DEVICES'] = gpu_list
    # print('export CUDA_VISIBLE_DEVICES=' + gpu_list)
    return opt



# --------------------------------------------
# continue training from the last cheakpoint
# --------------------------------------------
def find_last_checkpoint(save_dir):
    """
    Args: 
        save_dir: model folder
    Return:
        init_iter: iteration number
        init_path: model path
    """
    file_list = glob.glob(os.path.join(save_dir, '*.pth'))
    if file_list:
        iter_exist = []
        for file_ in file_list:
            iter_current = re.findall(r"(\d+).pth", file_)
            iter_exist.append(int(iter_current[0]))
        init_iter = max(iter_exist)
        init_path = os.path.join(save_dir, '{}.pth'.format(init_iter))
    else:
        init_iter = 0
        init_path = None
    return init_iter, init_path



# --------------------------------------------
# convert the opt into json file
# --------------------------------------------
def save(opt):
    opt_path = opt['opt_path']
    opt_path_copy = opt['path']['options']
    dirname, filename_ext = os.path.split(opt_path)
    filename, ext = os.path.splitext(filename_ext)
    dump_path = os.path.join(opt_path_copy, filename+get_timestamp()+ext)
    with open(dump_path, 'w') as dump_file:
        json.dump(opt, dump_file, indent=2)



# --------------------------------------------
# dict to string for logger
# --------------------------------------------
def dict2str(opt, indent_l=1):
    msg = ''
    for k, v in opt.items():
        if isinstance(v, dict):
            msg += ' ' * (indent_l * 2) + k + ':[\n'
            msg += dict2str(v, indent_l + 1)
            msg += ' ' * (indent_l * 2) + ']\n'
        else:
            msg += ' ' * (indent_l * 2) + k + ': ' + str(v) + '\n'
    return msg



# --------------------------------------------
# convert OrderedDict to NoneDict,
# return None for missing key
# --------------------------------------------
def dict_to_nonedict(opt):
    if isinstance(opt, dict):
        new_opt = dict()
        for key, sub_opt in opt.items():
            new_opt[key] = dict_to_nonedict(sub_opt)
        return NoneDict(**new_opt)
    elif isinstance(opt, list):
        return [dict_to_nonedict(sub_opt) for sub_opt in opt]
    else:
        return opt


class NoneDict(dict):
    def __missing__(self, key):
        return None
