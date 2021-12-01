import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional as tf
import numpy as np
from math import exp

EPS = 1e-5


class OutLoss(nn.Module):
    def __init__(self, min, max):
        super(OutLoss, self).__init__()
        self.min = min
        self.max = max

    def forward(self, recon):
        max_matrix = torch.ones_like(recon) * self.max
        minus_max = recon - max_matrix
        minus_max[minus_max < 0] = 0
        loss_max = torch.mean(minus_max)

        min_matrix = torch.ones_like(recon) * self.min
        minus_min = recon - min_matrix
        minus_min[minus_min > 0] = 0
        loss_min = torch.mean(torch.abs(minus_min))
        loss_out = loss_max + loss_min
        return loss_out


class SILoss(nn.Module):
    def __init__(self, alpha=10.0, lamb=0.85):
        '''Scale invariant loss'''
        super(SILoss, self).__init__()
        self.alpha = alpha
        self.lamb = lamb

    def forward(self, pred, gt, interpolate=False):
        '''Pixel value could not be 0'''
        if interpolate:
            pred = F.interpolate(pred, gt.shape[-2:], mode='bilinear', align_corners=True)
        log_diff = torch.log(pred) - torch.log(gt)
        loss = torch.sqrt(torch.mean(log_diff ** 2) - self.lamb * (torch.mean(log_diff) ** 2)) * self.alpha
        return loss


def imgrad(img):
    img = torch.mean(img, 1, True)
    fx = np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]])
    conv1 = nn.Conv2d(1, 1, kernel_size=3, stride=1, padding=1, bias=False)
    weight = torch.from_numpy(fx).float().unsqueeze(0).unsqueeze(0)
    if img.is_cuda:
        weight = weight.cuda()
    conv1.weight = nn.Parameter(weight)
    grad_x = conv1(img)

    fy = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])
    conv2 = nn.Conv2d(1, 1, kernel_size=3, stride=1, padding=1, bias=False)
    weight = torch.from_numpy(fy).float().unsqueeze(0).unsqueeze(0)

    if img.is_cuda:
        weight = weight.cuda()

    conv2.weight = nn.Parameter(weight)
    grad_y = conv2(img)
    return grad_y, grad_x


def imgrad_loss(pred, gt, mask=None):
    N, C, _, _ = pred.size()
    grad_y, grad_x = imgrad(pred)
    grad_y_gt, grad_x_gt = imgrad(gt)
    grad_y_diff = torch.abs(grad_y - grad_y_gt)
    grad_x_diff = torch.abs(grad_x - grad_x_gt)
    if mask is not None:
        grad_y_diff[~mask] = 0.1*grad_y_diff[~mask]
        grad_x_diff[~mask] = 0.1*grad_x_diff[~mask]
    return (torch.mean(grad_y_diff) + torch.mean(grad_x_diff))


class CrossGradientLoss(nn.Module):
    def __init__(self):
        super(CrossGradientLoss, self).__init__()

    def forward(self, pred, gt, mask=None, interpolate=False):
        if interpolate:
            pred = F.interpolate(pred, gt.shape[-2:], mode='bilinear', align_corners=True)
        if mask is not None:
            valid_mask = (pred > 0.001) & mask
        else:
            valid_mask = (pred > 0.001) & (gt > 0.001)
        N = torch.sum(valid_mask)

        log_pred = torch.log(torch.maximum(pred, torch.as_tensor(EPS)))
        log_gt = torch.log(torch.maximum(gt, torch.as_tensor(EPS)))

        log_d_diff = log_pred - log_gt
        log_d_diff = torch.mul(log_d_diff, valid_mask)

        v_gradient = torch.abs(log_d_diff[0:-2, :] - log_d_diff[2:, :])
        v_mask = torch.mul(valid_mask[0:-2, :], valid_mask[2:, :])
        v_gradient = torch.mul(v_gradient, v_mask)

        h_gradient = torch.abs(log_d_diff[:, 0:-2] - log_d_diff[:, 2:])
        h_mask = torch.mul(valid_mask[:, 0:-2], valid_mask[:, 2:])
        h_gradient = torch.mul(h_gradient, h_mask)

        gradient_loss = torch.sum(h_gradient) + torch.sum(v_gradient)
        gradient_loss = gradient_loss/N

        return gradient_loss


class SobelLayer(nn.Module):

    def __init__(self):
        super(SobelLayer, self).__init__()
        kernel_v = [[0, -1, 0],
                    [0, 0, 0],
                    [0, 1, 0]]
        kernel_h = [[0, 0, 0],
                    [-1, 0, 1],
                    [0, 0, 0]]
        kernel_h = torch.FloatTensor(kernel_h).unsqueeze(0).unsqueeze(0)
        kernel_v = torch.FloatTensor(kernel_v).unsqueeze(0).unsqueeze(0)
        self.weight_h = nn.Parameter(data=kernel_h, requires_grad=False)
        self.weight_v = nn.Parameter(data=kernel_v, requires_grad=False)

    def get_gray(self, x):
        ''' 
        Convert image to its gray one.
        '''
        gray_coeffs = [65.738, 129.057, 25.064]
        convert = x.new_tensor(gray_coeffs).view(1, 3, 1, 1) / 256
        x_gray = x.mul(convert).sum(dim=1)
        return x_gray.unsqueeze(1)

    def forward(self, x):
        if x.shape[1] == 3:
            x = self.get_gray(x)

        x_v = F.conv2d(x, self.weight_v, padding=1)
        x_h = F.conv2d(x, self.weight_h, padding=1)
        x = torch.sqrt(torch.pow(x_v, 2) + torch.pow(x_h, 2) + 1e-6)
        return x


class SobelLoss(nn.Module):

    def __init__(self):
        super(SobelLoss, self).__init__()
        self.loss = nn.L1Loss()
        self.grad_layer = SobelLayer()

    def forward(self, output, gt_img):
        output_grad = self.grad_layer(output)
        gt_grad = self.grad_layer(gt_img)
        return self.loss(output_grad, gt_grad)
