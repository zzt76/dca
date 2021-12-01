import os
import glob
import yaml
import random
import numpy as np
import cv2
import torch


def setup_runtime(args):
    """Load configs, initialize CUDA, CuDNN and the random seeds."""

    cfgs = {}
    cfgs.update(args)

    if torch.cuda.is_available():
        torch.backends.cudnn.enabled = True
        torch.backends.cudnn.benchmark = True

    # Setup random seeds for reproducibility
    if cfgs['seed']:
        random.seed(cfgs['seed'])
        np.random.seed(cfgs['seed'])
        torch.manual_seed(cfgs['seed'])
        torch.cuda.manual_seed_all(cfgs['seed'])  # It's safe even if cuda is not available

    # Load config
    if cfgs['config'] is not None and os.path.isfile(cfgs['config']):
        with open(cfgs['config']) as f:
            cfg = yaml.safe_load(f)
        cfgs.update(cfg)

    print(f"Loading config from {cfgs['config']}")
    print(f"Environment: {cfgs['device']}: {cfgs['device_id'] if cfgs['device']=='gpu' and cfgs['multi_gpu'] else None,}",
          f"seed: {cfgs['seed']} num_workers: {cfgs['num_workers']}",
          f"dataset: {cfgs['dataset']}")
    return cfgs


def dump_yaml(path, cfgs):
    print(f"Saving configs to {path}")
    xmkdir(os.path.dirname(path))
    with open(path, 'w') as f:
        return yaml.safe_dump(cfgs, f)


def xmkdir(path):
    """Create directory PATH recursively if it does not exist."""
    os.makedirs(path, exist_ok=True)


def clean_checkpoint(checkpoint_dir, keep_num=2):
    if keep_num > 0:
        names = list(sorted(
            glob.glob(os.path.join(checkpoint_dir, 'checkpoint*.pth'))
        ))
        if len(names) > keep_num:
            for name in names[:-keep_num]:
                print(f"Deleting obslete checkpoint file {name}")
                os.remove(name)


def save_images(out_fold, imgs, prefix='', suffix='', sep_folder=True, ext='.png', is_test=False):
    if sep_folder:
        out_fold = os.path.join(out_fold, suffix.split('-')[0])
    xmkdir(out_fold)
    prefix = prefix + '_' if prefix else ''
    suffix = '_' + suffix if suffix else ''
    offset = len(glob.glob(os.path.join(out_fold, prefix+'*'+suffix+ext))) + 1
    # b c h w -> b h w c
    imgs = imgs.transpose(0, 2, 3, 1)

    for i, img in enumerate(imgs):
        if 'depth' in suffix or 'pred' in suffix:
            if is_test:
                # im_out = np.uint16(np.clip(img*10000., 0, 10000))  # to [0,10000]
                im_out = np.uint16(np.clip(img*65535., 0, 65535))
            else:
                im_out = np.uint16(np.clip(img*65535., 0, 65535))
        else:
            im_out = np.uint8(np.clip(img*255., 0, 255))
            im_out = cv2.cvtColor(im_out, cv2.COLOR_RGB2BGR)

        if is_test:
            cv2.imwrite(os.path.join(out_fold, suffix.lstrip('_')+ext), im_out)
        else:
            cv2.imwrite(os.path.join(out_fold, prefix+'%05d' % (i+offset)+suffix+ext), im_out)


class RunningAverage():
    def __init__(self):
        self.avg = 0
        self.count = 0

    def append(self, value):
        self.avg = (value + self.count * self.avg) / (self.count + 1)
        self.count += 1

    def get_value(self):
        return self.avg


class RunningAverageDict():
    def __init__(self):
        self._dict = None

    def update(self, new_dict):
        if self._dict is None:
            self._dict = dict()
            for key, value in new_dict.items():
                self._dict[key] = RunningAverage()

        for key, value in new_dict.items():
            self._dict[key].append(value)

    def get_value(self):
        return {key: value.get_value() for key, value in self._dict.items()}
