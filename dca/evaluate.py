import os
from collections import OrderedDict
import tqdm
import numpy as np

from .dataloader import DataLoader
from . import network as network
from . import utils
import torch
import torchvision.transforms.functional as tf
import datetime


class Eval():
    def __init__(self, cfgs):
        ### trainer ###
        self.device = cfgs.get('device', 'gpu')
        self.num_epochs = cfgs.get('num_epochs', 30)
        self.run_train = cfgs.get('run_train', False)
        self.device_id = torch.device(int(cfgs.get('device_id', 0)[0]))
        ### dataset ###
        self.dataset = cfgs.get('dataset', 'vari')  # training whether in vari or nyu dataset
        self.test_list = cfgs.get('test_list', './data/vari/test.txt')
        ### dataloader ###
        self.height = cfgs.get('input_height', None)
        self.width = cfgs.get('input_width', None)
        self.min_depth = cfgs.get('min_depth', 0.001)
        self.max_depth = cfgs.get('max_depth', 10.0)
        self.batch_size = cfgs.get('batch_size', 16)
        self.num_workers = cfgs.get('num_workers', 8)
        self.do_aug = cfgs.get('do_aug', False)
        self.do_flip = cfgs.get('do_flip', True)
        ### checkpoint ###
        self.checkpoint_path = cfgs.get('checkpoint_path', None)
        ### save dir ###
        self.test_result_dir = cfgs.get('test_result_dir', None)
        self.test_result_dir = os.path.join(self.test_result_dir, self.dataset+'_'+os.path.basename(self.checkpoint_path[:-4])+datetime.datetime.now().strftime('_%m%d-%H:%M'))
        os.makedirs(self.test_result_dir, exist_ok=True)
        print(f"Saving testing results to {self.test_result_dir}")
        ### result log ###
        self.log_name = self.dataset + '_' + os.path.basename(self.checkpoint_path[:-4]) + datetime.datetime.now().strftime('_%m%d-%H:%M') + '.log'

        ######################### Load data ############################
        data_loader = DataLoader(self.dataset, self.height, self.width, self.batch_size, self.num_workers,
                                 train_list=None, val_list=None, test_list=self.test_list,
                                 run_train=self.run_train, do_aug=self.do_aug)
        self.test_loader = data_loader.get_test_loader()

        ######################### Load model ############################
        model = network.DCA(cout=1)
        if self.device == 'gpu':
            self.model = model.to(self.device_id)
        elif self.device == 'cpu':
            self.device_id = torch.device(self.device)
            self.model = model.to(self.device_id)

    @torch.no_grad()
    def test(self):
        """Perform testing."""
        self.load_checkpoint()
        print(f"Starting validation {self.checkpoint_path}")

        self.metrics = utils.RunningAverageDict()

        self.model.eval()
        self.num_iters = len(self.test_loader)
        self.sum_l1_depth = 0
        self.log = open(os.path.join('./results', self.log_name), mode='w')
        for input in tqdm.tqdm(self.test_loader):
            self.depth_pred = self.predict(input)

            self.input_im = self.input_im.cpu().numpy()
            self.depth_gt = self.depth_gt.cpu().numpy()  # [0, 10]
            self.depth_pred = self.depth_pred.cpu().numpy()  # [0, 10]

            valid_mask = np.logical_and(self.depth_gt[0, 0] > self.min_depth, self.depth_gt[0, 0] < self.max_depth)
            eval_mask = np.zeros(valid_mask.shape)
            if self.dataset == 'nyu':
                eval_mask[45:471, 41:601] = 1
            else:
                eval_mask = 1

            valid_mask = np.logical_and(valid_mask, eval_mask)
            errors = self.compute_errors(self.depth_gt[0, 0][valid_mask], self.depth_pred[0, 0][valid_mask])
            self.log.write(f"{input['name'][0]}/ RMSE: {errors['rmse']} \n")
            self.metrics.update(errors)
            self.save_results(input['name'][0], self.test_result_dir)

        self.log.write(f"Test{self.checkpoint_path[:-4]}/ Average:{self.metrics.get_value()}")

    def predict(self, input):
        self.input_im = input['color'].to(self.device_id)
        self.depth_gt = input['depth'].to(self.device_id)
        pred = self.model(self.input_im)

        if self.do_flip:
            input_flip = tf.hflip(self.input_im)
            pred_flip = self.model(input_flip)
            pred = (pred + tf.hflip(pred_flip)) * 0.5

        pred = torch.clamp(pred, min=self.min_depth, max=self.max_depth)
        return pred

    def save_results(self, prename, save_dir):
        sep_folder = False
        utils.save_images(save_dir, self.input_im, suffix=prename+'-color', sep_folder=sep_folder, is_test=True)
        utils.save_images(save_dir, self.depth_gt / 10.0, suffix=prename+'-depth', sep_folder=sep_folder, is_test=True)
        utils.save_images(save_dir, self.depth_pred / 10.0, suffix=prename+'-pred', sep_folder=sep_folder, is_test=True)

    def compute_errors(self, gt, pred):
        thresh = np.maximum((gt / pred), (pred / gt))
        a1 = (thresh < 1.25).mean()
        a2 = (thresh < 1.25 ** 2).mean()
        a3 = (thresh < 1.25 ** 3).mean()

        abs_rel = np.mean(np.abs(gt - pred) / gt)
        sq_rel = np.mean(((gt - pred) ** 2) / gt)

        rmse = (gt - pred) ** 2
        rmse = np.sqrt(rmse.mean())

        rmse_log = (np.log(gt) - np.log(pred)) ** 2
        rmse_log = np.sqrt(rmse_log.mean())

        err = np.log(pred) - np.log(gt)
        silog = np.sqrt(np.mean(err ** 2) - np.mean(err) ** 2) * 100

        log_10 = (np.abs(np.log10(gt) - np.log10(pred))).mean()
        return dict(a1=a1, a2=a2, a3=a3, abs_rel=abs_rel, rmse=rmse, log_10=log_10, rmse_log=rmse_log,
                    silog=silog, sq_rel=sq_rel)

    def load_checkpoint(self):
        print(f"Loading checkpoint from {self.checkpoint_path}")
        cp = torch.load(self.checkpoint_path, map_location=self.device_id)
        if 'model' in cp:
            state_dict = cp['model']
        else:
            state_dict = cp
        state_dict = self.mul_to_single_checkpoint(state_dict)
        self.model.load_state_dict(state_dict, strict=True)  # load model weights

    def mul_to_single_checkpoint(self, state_dict):
        new_state_dict = OrderedDict()

        for i, j in state_dict.items():
            if i.startswith('module.'):
                name = i.replace('module.', '')
            else:
                name = i
            new_state_dict[name] = j
        return new_state_dict
