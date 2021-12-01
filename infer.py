import os
from collections import OrderedDict

import torch
import torchvision.transforms.functional as tf
import numpy as np
from PIL import Image
from tqdm import tqdm

from dca.network import DCA


class Inference():
    def __init__(self, pretrained_path='pretrained/dca_vari.pth', device=0):
        self.min_depth = 1e-3
        self.max_depth = 10
        self.saving_factor = 65535
        self.device = torch.device(device)

        model = DCA()
        model = self.load_checkpoint(model, pretrained_path)
        model.eval()
        self.model = model.to(self.device)

    def predict_dir(self, im_dir, save_dir):
        paths = os.listdir(im_dir)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
            print(f"Created directory '{save_dir}'.")
        for path in tqdm(paths):
            path = os.path.join(im_dir, path)
            name = os.path.basename(path).split('.')[0] + '-pred.png'
            save_path = os.path.join(save_dir, name)
            self.predict_path(path, save_path, verbose=False)
        print(f"Predicted depth maps of folder '{im_dir}' are saved in '{save_dir}'")

    def predict_path(self, path, save_path=None, verbose=True):
        input_im = Image.open(path)
        input_im = torch.unsqueeze(tf.to_tensor(input_im), dim=0)

        pred = self.predict(input_im)
        pred = Image.fromarray(pred)

        if save_path is None:
            name = os.path.basename(path).split('.')[0] + '-pred.png'
            save_path = os.path.join(os.path.dirname(path), name)
        pred.save(save_path)
        if verbose:
            print(f"Predicted depth is saved '{save_path}'")

    @torch.no_grad()
    def predict(self, input):
        input_im = input.to(self.device)
        pred = self.model(input_im)

        input_flip = tf.hflip(input_im)
        pred_flip = self.model(input_flip)
        pred = (pred + tf.hflip(pred_flip)) * 0.5

        pred = torch.clamp(pred, min=self.min_depth, max=self.max_depth)
        pred = np.uint16(pred.cpu().numpy()[0][0] / 10.0 * 65535.0)
        return pred

    def load_checkpoint(self, model, pretrained_path):
        print(f"Loading checkpoint from {pretrained_path}")
        cp = torch.load(pretrained_path, map_location=self.device)
        if 'model' in cp:
            state_dict = cp['model']
        else:
            state_dict = cp
        state_dict = self.mul_to_single_checkpoint(state_dict)
        model.load_state_dict(state_dict, strict=True)  # load model weights
        return model

    def mul_to_single_checkpoint(self, state_dict):
        new_state_dict = OrderedDict()
        for i, j in state_dict.items():
            if i.startswith('module.'):
                name = i.replace('module.', '')
            else:
                name = i
            new_state_dict[name] = j
        return new_state_dict


if __name__ == '__main__':
    infer = Inference(pretrained_path='pretrained/dca_vari.pth', device=torch.device(0))
    infer.predict_path(path='test_images/485_c1_SunMorning_Indoor_Environment0188.jpg')
    infer.predict_dir(im_dir='test_images', save_dir='results')
