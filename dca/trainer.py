import os
import time
import glob
from collections import OrderedDict

from .dataloader import DataLoader
from . import network as network
from . import utils
from . import loss
import torch
import torch.nn as nn
from torchviz import make_dot


class Train():
    def __init__(self, cfgs):
        ### trainer ###
        self.device = cfgs.get('device', 'gpu')
        self.multi_gpu = cfgs.get('multi_gpu', True)
        self.device_id = cfgs.get('device_id', [0, 1])
        self.num_epochs = cfgs.get('num_epochs', 30)
        self.run_train = cfgs.get('run_train', False)
        ### dataloader ###
        self.height = cfgs.get('input_height', None)
        self.width = cfgs.get('input_width', None)
        self.batch_size = cfgs.get('batch_size', 16)
        self.num_workers = cfgs.get('num_workers', 8)
        self.do_aug = cfgs.get('do_aug', False)
        ### dataset ###
        self.dataset = cfgs.get('dataset', 'vari')  # training whether in vari or nyu dataset
        self.train_list = cfgs.get('train_list', None)  # see utils.py, which gets paths of lists
        self.val_list = cfgs.get('val_list', None)
        self.test_list = cfgs.get('test_list', None)
        self.min_depth = cfgs.get('min_depth', 0.001)
        self.max_depth = cfgs.get('max_depth', 10)
        ### optimizer & scheduler ###
        self.lr = cfgs.get('lr', 1e-4)
        self.weight_decay = cfgs.get('weight_decay', 0.1)
        self.resume_optimizer = cfgs.get('resume_optimizer', True)
        self.diff_lr = cfgs.get('diff_lr', False)
        self.use_scheduler = cfgs.get('use_scheduler', True)
        self.resume_scheduler = cfgs.get('resume_scheduler', True)
        self.accumulate_grad = cfgs.get('accumulate_grad', True)
        self.accumulate_steps = cfgs.get('accumulate_steps', 4)
        ### checkpoint ###
        self.resume = cfgs.get('resume', True)
        self.checkpoint_dir = os.path.join(cfgs.get('checkpoint_dir', 'checkpoints'), self.dataset)
        self.checkpoint_name = cfgs.get('checkpoint_name', None)
        self.save_checkpoint_freq = cfgs.get('save_checkpoint_freq', 1)
        self.keep_num_checkpoint = cfgs.get('keep_num_checkpoint', 20)  # -1 for keeping all checkpoints
        ### logger ###
        self.use_logger = cfgs.get('use_logger', True)
        self.log_freq = cfgs.get('log_freq', 1000)
        self.visualize_freq = cfgs.get('visualize_freq', 500)
        self.val_dir = os.path.join(cfgs.get('val_dir', ''), self.dataset)

        ######################### Load data ############################
        data_loader = DataLoader(self.dataset, self.height, self.width, self.batch_size, self.num_workers,
                                 self.train_list, self.val_list, self.test_list, self.run_train, self.do_aug)
        self.train_loader, self.val_loader = data_loader.get_train_val_loaders()

        ######################### Load model ############################
        model = network.DCA(cout=1)
        if self.device == 'gpu':
            if self.multi_gpu:
                model = nn.parallel.DataParallel(model, device_ids=self.device_id)
            self.model = model.cuda()
        elif self.device == 'cpu':
            self.model = model

        ######################### Init Optimizer #########################
        if self.diff_lr:
            if self.multi_gpu:
                params = [{"params": self.model.module.encoder.parameters(), "lr": self.lr / 10},
                          {"params": self.model.module.decoder.parameters(), "lr": self.lr}]
            else:
                params = [{"params": self.model.encoder.parameters(), "lr": self.lr / 10},
                          {"params": self.model.decoder.parameters(), "lr": self.lr}]
        else:
            params = [{"params": self.model.parameters(), "lr": self.lr}]
        self.optimizer = torch.optim.AdamW(params, weight_decay=self.weight_decay, eps=1e-3)

        ######################### Init Lr Scheduler #########################
        if self.use_scheduler:
            self.scheduler = torch.optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=0.98)

        ######################### Init Losses ############################
        self.l1loss = nn.L1Loss().cuda()
        self.siloss = loss.SILoss().cuda()

    def train(self):
        start_epoch = 0

        if self.resume:
            start_epoch = self.load_checkpoint(self.resume_optimizer, self.resume_scheduler)

        if self.use_logger:
            # cache one batch for visualization
            self.train_viz = next(iter(self.train_loader))
            # self.val_viz = next(iter(self.val_loader))

        # run epochs
        print(f"Optimizing to {self.num_epochs} epochs")
        total_epochs = 0
        for epoch in range(start_epoch, self.num_epochs):
            self.current_epoch = epoch
            print(f"Current lr is {self.optimizer.state_dict()['param_groups'][0]['lr']}")
            self.run_epoch(self.train_loader, epoch)

            with torch.no_grad():
                self.run_epoch(self.val_loader, epoch, is_validation=True)

            if (epoch+1) % self.save_checkpoint_freq == 0:
                self.save_checkpoint(epoch+1)
            total_epochs = epoch

        print(f"Training completed after {total_epochs+1} epochs.")

    def run_epoch(self, loader, epoch=0, is_validation=False):
        """Run one epoch."""
        self.is_train = not is_validation
        self.iter_per_epoch = len(loader)
        epoch_start_time = time.time()
        if self.is_train:
            print(f"Starting training epoch {epoch+1}")
            self.model.train()
        else:
            print(f"Starting validation epoch {epoch+1}")
            self.model.eval()
        epoch_avg_loss = {}  # compute average loss of each epoch

        for iter, input in enumerate(loader):
            metrics = self.run_iter(input)

            if self.is_train:
                if self.accumulate_grad:
                    self.loss_total /= self.accumulate_steps
                    self.loss_total.backward()
                    if (iter+1) % self.accumulate_steps == 0:
                        self.optimizer.step()
                        self.optimizer.zero_grad()
                else:
                    self.optimizer.zero_grad()
                    self.loss_total.backward()
                    self.optimizer.step()

            if iter % self.log_freq == 0 and (self.is_train or is_validation):
                print(f"{'Train' if self.is_train else 'Val'}{epoch+1:03}/{iter:05}/{metrics}")

            if self.use_logger and self.is_train:
                total_iter = iter + epoch*self.iter_per_epoch
                if total_iter % self.visualize_freq == 0:
                    with torch.no_grad():
                        self.run_iter(self.train_viz)
                        self.visualize(total_iter=total_iter, total_epoch=epoch+1, val_dir=self.val_dir, vis_num=5, is_train=True)
            if self.use_logger and is_validation:
                with torch.no_grad():
                    self.run_iter(input)
                    self.visualize(total_iter=None, total_epoch=epoch+1, val_dir=self.val_dir, vis_num=1, is_train=False)

            for i, j in metrics.items():  # compute average loss of each epoch
                if i in epoch_avg_loss.keys():
                    epoch_avg_loss[i] += j
                else:
                    epoch_avg_loss[i] = j

        for i, j in epoch_avg_loss.items():
            epoch_avg_loss[i] = j / self.iter_per_epoch
        print(f"Avg loss of Epoch{epoch+1:03}: {epoch_avg_loss}")

        if self.is_train and self.use_scheduler:
            self.scheduler.step()
        epoch_time = time.time() - epoch_start_time
        print(f"This epoch cost {epoch_time}.")

    def run_iter(self, input):
        self.input_im = input['color'].cuda()
        self.depth_gt = input['depth'].cuda()
        if 'dense' in input:
            self.depth_dense = input['dense'].cuda()
        else:
            self.depth_dense = self.depth_gt

        self.depth_pred = self.model(self.input_im)
        self.gt_mask = self.depth_gt > self.min_depth
        self.pred_mask = self.depth_pred > self.min_depth
        valid_mask = torch.logical_and(self.gt_mask, self.pred_mask)

        if self.current_epoch >= 20:
            loss_grad = 0.5 * loss.imgrad_loss(self.depth_pred, self.depth_dense)
        else:
            loss_grad = torch.tensor(0.).cuda()

        if self.dataset == 'vari':
            loss_si = self.siloss(self.depth_pred[valid_mask], self.depth_gt[valid_mask])
            self.loss_total = loss_si + loss_grad
            metrics = {'loss': self.loss_total}
            metrics['si_depth'] = loss_si

        elif self.dataset == 'nyu':
            loss_l1 = self.l1loss(self.depth_pred[valid_mask], self.depth_gt[valid_mask])
            loss_si = 0.02 * self.siloss(self.depth_pred[valid_mask], self.depth_gt[valid_mask])
            self.loss_total = loss_l1 + loss_si + loss_grad
            metrics = {'loss': self.loss_total}
            metrics['l1_depth'] = loss_l1
            metrics['si_depth'] = loss_si

        with torch.no_grad():
            rmse_loss = torch.sqrt(torch.pow(self.depth_pred[self.gt_mask].detach()-self.depth_gt[self.gt_mask], 2).mean())
            rmse_loss = rmse_loss.item()

        metrics['grad_depth'] = loss_grad
        metrics['rmse'] = rmse_loss
        return metrics

    def visualize(self, total_iter, total_epoch, val_dir, vis_num=4, is_train=True):
        b, c, h, w = self.input_im.shape
        if is_train:
            self.vis_save_images(dir=val_dir, prename=f'Iter{total_iter:07}', sep_folder=True, save_num=vis_num)
        else:
            self.vis_save_images(dir=val_dir, prename=f'Epoch{total_epoch:03}', sep_folder=True, save_num=vis_num)

    def vis_save_images(self, dir, prename, sep_folder, save_num=None):
        '''num is None means save all this batch'''
        b = self.input_im.shape[0]
        if save_num == None:
            save_num = b
        index = torch.randint(low=0, high=b, size=(save_num,)).cuda()  # random choose some images to show
        input_im = torch.index_select(self.input_im, dim=0, index=index)[:save_num].detach().cpu().numpy()
        utils.save_images(dir, input_im, suffix=prename+'-color', sep_folder=sep_folder)
        depth_gt = torch.index_select(self.depth_gt, dim=0, index=index)[:save_num].detach().cpu().numpy() / 10.0
        utils.save_images(dir, depth_gt, suffix=prename+'-depth', sep_folder=sep_folder)
        depth_pred = torch.index_select(self.depth_pred, dim=0, index=index)[:save_num].detach().cpu().numpy() / 10.0
        utils.save_images(dir, depth_pred, suffix=prename+'-pred', sep_folder=sep_folder)

    def load_checkpoint(self, resume_optimizer, resume_scheduler):
        """Search the specified/latest checkpoint in checkpoint_dir and load the model and optimizer."""
        if self.checkpoint_name is not None:
            checkpoint_path = os.path.join(self.checkpoint_dir, self.checkpoint_name)
        else:
            checkpoints = sorted(glob.glob(os.path.join(self.checkpoint_dir, '*.pth')))
            if len(checkpoints) == 0:
                return 0
            checkpoint_path = checkpoints[-1]
            self.checkpoint_name = os.path.basename(checkpoint_path)
        print(f"Loading checkpoint from {checkpoint_path}")
        cp = torch.load(checkpoint_path)
        if 'model' in cp:
            self.model.load_state_dict(cp['model'], strict=True)  # load model weights
        else:
            self.model.load_state_dict(cp, strict=True)

        if 'epoch' in cp:
            epoch = cp['epoch']
        else:
            epoch = int(self.checkpoint_name[-7:-4])

        if resume_optimizer:
            self.optimizer.load_state_dict(cp['optimizer'])  # load optimizer states
        if self.use_scheduler and resume_scheduler:
            self.scheduler.load_state_dict(cp['scheduler'])  # load scheduler states
        return epoch

    def mul_to_single_checkpoint(self, state_dict):
        new_state_dict = OrderedDict()
        for i, j in state_dict.items():
            name = i[7:]
            new_state_dict[name] = j
        return new_state_dict

    def save_checkpoint(self, epoch, state_dict_only=False):
        """Save model, optimizer, and metrics state to a checkpoint in checkpoint_dir for the specified epoch."""
        utils.xmkdir(self.checkpoint_dir)
        utils.xmkdir(os.path.join(self.checkpoint_dir, 'state_dict'))
        checkpoint_path = os.path.join(self.checkpoint_dir, f'checkpoint{epoch:03}.pth')
        # model_dict_path = os.path.join(self.checkpoint_dir, 'state_dict', f'model{epoch:03}.pth')

        state_dict = {}
        if state_dict_only:
            state_dict = self.model.state_dict()

        else:
            state_dict['model'] = self.model.state_dict()
            state_dict['optimizer'] = self.optimizer.state_dict()
            if self.use_scheduler:
                state_dict['scheduler'] = self.scheduler.state_dict()
            state_dict['epoch'] = epoch

        print(f"Saving checkpoint to {checkpoint_path}")
        torch.save(state_dict, checkpoint_path)
        if self.keep_num_checkpoint > 0:
            utils.clean_checkpoint(self.checkpoint_dir, keep_num=self.keep_num_checkpoint)
