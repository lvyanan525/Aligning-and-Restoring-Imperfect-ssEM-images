import os
import cv2
import torch
import numpy as np
from abc import abstractmethod
from easydict import EasyDict
from losses.pwclite import PWCLite
from register.flow import warp_image_relative, warp_image_two_steps
from register.GPE.WrinkleReg import WrinkleReg
from register.GPE.warp import warp_c
from register.GPE.superpoint import SuperPointReg as SuperPoint
import torch.nn.functional as F
# from time import perf_counter
import subprocess

def img2tensor(img, data_type=np.uint8):
    if data_type == np.uint16 or data_type == "uint16":
        img = torch.from_numpy((img / 65535.).astype(np.float32))
    else:
        img = torch.from_numpy(img) / 255.
    if len(img.shape) == 3:
        img = img.permute([2, 0, 1])
    else:
        img = img.unsqueeze(0)
    return img.unsqueeze(0)


def tensor2img(tensor, data_type=np.uint8):
    if data_type == np.uint16 or data_type == "uint16":
        tensor = (tensor * 65535).cpu().numpy().astype(np.uint16)
    else:
        tensor = (tensor * 255).cpu().numpy().astype(np.uint8)
    if tensor.shape[1] != 1:
        tensor = tensor.transpose([0, 2, 3, 1])
    return tensor.squeeze()

def compose_displacements(disp_a_, disp_b):
    H, W = disp_a_.shape[1], disp_a_.shape[2]
    disp_a_xy = np.stack([disp_a_[1], disp_a_[0]], axis=0)  # [2, H, W] in [x, y] format

    yy, xx = np.meshgrid(np.arange(H, dtype=np.float32), np.arange(W, dtype=np.float32), indexing='ij')
    identity_grid = np.stack([xx, yy], axis=0)  # [2, H, W] in [x, y] format
    
    disp_a = disp_a_xy - identity_grid  # [2, H, W] in [x, y] format
    
    disp_a = torch.from_numpy(disp_a).float().unsqueeze(0)  # [1, 2, H, W]

    if isinstance(disp_a, np.ndarray):
        disp_a = torch.from_numpy(disp_a).float()
    
    if disp_a.dim() == 3:  # [2, H, W]
        disp_a = disp_a.unsqueeze(0)  # [1, 2, H, W]
    elif disp_a.dim() != 4:
        raise ValueError(f"disp_a must have 3 or 4 dimensions, got {disp_a.dim()}")
    
    if disp_b.device != disp_a.device:
        disp_a = disp_a.to(disp_b.device)
    
    B, C, H, W = disp_a.shape
    
    yy, xx = torch.meshgrid(torch.linspace(-1, 1, H, device=disp_a.device), 
                            torch.linspace(-1, 1, W, device=disp_a.device), 
                            indexing='ij')
    identity_grid = torch.stack([xx, yy], dim=-1).unsqueeze(0)  # [1, H, W, 2]
    
    disp_a_norm = disp_a.permute(0, 2, 3, 1).clone()  # [1, H, W, 2]
    disp_a_norm[:, :, :, 0] = disp_a_norm[:, :, :, 0] * 2.0 / (W - 1)  # normalize x
    disp_a_norm[:, :, :, 1] = disp_a_norm[:, :, :, 1] * 2.0 / (H - 1)  # normalize y
    
    sampling_grid = identity_grid + disp_a_norm
    
    resampled_disp_b = F.grid_sample(disp_b, sampling_grid, 
                                     mode='bilinear', 
                                     padding_mode='border', 
                                     align_corners=True)
    
    return disp_a + resampled_disp_b

def gpe_generate(img_fixed_kp, img_moving_kp, img_moving, img_tr=None):
    torch.random.manual_seed(0)
    np.random.seed(0)
    det = SuperPoint(0.015, batch_sz=8, device="cpu")
    det.detectAndCompute(np.random.random([128, 128]))

    kp_fixed, des_fixed = det.detectAndComputeMean(img_fixed_kp, scale=1, mask=None)
    kp_moving, des_moving = det.detectAndComputeMean(img_moving_kp, scale=1, mask=None)
    if img_tr is None:
        img_tr = np.zeros_like(img_moving)
    reg_er = WrinkleReg(k=20, K=3, alpha=1, img_tr=img_tr, match_radius=10000, device="cpu", test=True)
    c_lst = reg_er.plant(kp_moving, kp_fixed, des_moving, des_fixed)
    # t1 = perf_counter()
    field = reg_er.generate_field(c_lst, img_moving)
    # t2 = perf_counter()
    # print("ea time:{:.3f}s".format(t2 - t1))
    return field

class NonLinearAlign:
    def __init__(self):
        pass

    @abstractmethod
    def generate_field(self, img_r, img_m):
        pass

    @abstractmethod
    def warp_with_field(self, img_m, field, mode):
        pass


class FlowAligner(NonLinearAlign):
    def __init__(self, model_path, is_context=True, device="cuda"):
        super(FlowAligner, self).__init__()
        cfg = EasyDict({
            'upsample': True,
            'n_frames': 2,
            'reduce_dense': True,
            'is_context': is_context,
        })
        self.device = device
        self.model = PWCLite(cfg)
        self.model = self.restore_model(self.model, model_path)
        self.model.to(self.device)
        self.model.eval()

    def estimate(self, img_r, img_m, gpe_field):
        """
        params: img_r
        params: img_m
        """
        img_m = warp_c(img_m, gpe_field)
        if not isinstance(img_r, torch.Tensor):
            img_r = img2tensor(img_r)
        if not isinstance(img_m, torch.Tensor):
            img_m = img2tensor(img_m)
        if img_r.shape[1] == 1:
            img_r = img_r.repeat(1, 3, 1, 1)
        if img_m.shape[1] == 1:
            img_m = img_m.repeat(1, 3, 1, 1)
        # imgs = [self.transform(img_r), self.transform(img_m)]
        imgs = [img_r, img_m]
        img_pair = torch.cat(imgs, 1).to(self.device)
        field = self.model(img_pair)['flows_fw'][0].detach().cpu()
        return compose_displacements(gpe_field, field)

    def generate_field(self, img_r, img_m, img_mask):
        """
        params: img_r
        params: img_m
        """
        
        gpe_field = gpe_generate(img_r, img_m, img_m, img_mask)
        # img_r = img2tensor(img_r)
        # img_m = img2tensor(img_m)
        flow = self.estimate(img_r, img_m, gpe_field)
        # flow = compose_displacements(gpe_field, flow)
        return flow

    def warp_with_field(self, img_m, field, mode="bilinear"):
        """
        params: img_m
        params: field
        """
        data_type = img_m.dtype
        img_m = img2tensor(img_m, data_type)
        warped = warp_image_relative(img_m, field, mode)
        warped = tensor2img(warped, data_type)
        return warped

    def load_model(self, model_path):
        self.model.load_model(model_path)

    def load_checkpoint(self, model_path):
        weights = torch.load(model_path)
        epoch = None
        if 'epoch' in weights:
            epoch = weights.pop('epoch')
        if 'state_dict' in weights:
            state_dict = (weights['state_dict'])
        else:
            state_dict = weights
        return epoch, state_dict

    def restore_model(self, model, pretrained_file):
        epoch, weights = self.load_checkpoint(pretrained_file)

        model_keys = set(model.state_dict().keys())
        weight_keys = set(weights.keys())

        # load weights by name
        weights_not_in_model = sorted(list(weight_keys - model_keys))
        model_not_in_weights = sorted(list(model_keys - weight_keys))
        if len(model_not_in_weights):
            print('Warning: There are weights in model but not in pre-trained.')
            for key in (model_not_in_weights):
                print(key)
                weights[key] = model.state_dict()[key]
        if len(weights_not_in_model):
            print('Warning: There are pre-trained weights not in model.')
            for key in (weights_not_in_model):
                print(key)
            from collections import OrderedDict
            new_weights = OrderedDict()
            for key in model_keys:
                new_weights[key] = weights[key]
            weights = new_weights

        model.load_state_dict(weights)
        return model

