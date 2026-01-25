import os
import cv2
import torch
import json
import time
import numpy as np
import torch.nn as nn
from copy import deepcopy
from easydict import EasyDict
from utils.torch_utils import init_seed, restore_model, save_checkpoint
from utils.warp_utils import flow_warp
from losses.pwclite import PWCLite
from Dataset.get_dataset import get_dataloader
from Dataset.format_consis import ensure_cv2_grayscale_format, match_tensor_shape, match_mask_shape
from losses.flow_loss import get_loss
from utils.misc_utils import AverageMeter
from Dataset.transforms.ar_transforms.sp_transfroms import RandomAffineFlow
from Dataset.transforms.ar_transforms.oc_transforms import run_slic_pt, random_crop
from skimage.metrics import structural_similarity as compute_ssim
from register.non_linear_align import gpe_generate, warp_c, img2tensor, tensor2img
from register.pairwise_reg import ext_call_pairwise

def compute_ncc(img1, img2):
    img1 = img1.astype(np.float32)
    img2 = img2.astype(np.float32)
    img1 = (img1 - img1.mean()) / img1.std()
    img2 = (img2 - img2.mean()) / img2.std()
    return np.corrcoef(img1.flatten(), img2.flatten())[0, 1]

def train_one_epoch(cfg, model, loss_func, optimizer, train_loader, device, epoch):
    sp_transform = RandomAffineFlow(cfg.st_cfg, addnoise=cfg.st_cfg.add_noise).to(device)
    # define meters
    am_batch_time = AverageMeter()
    am_data_time = AverageMeter()
    model.train()
    end = time.time()

    key_meter_names = ['Loss', 'l_ph', 'l_sm', 'flow_mean', 'l_atst', 'l_ot']
    key_meters = AverageMeter(i=len(key_meter_names), precision=4)

    for i_step, data in enumerate(train_loader):
        img_r = data['img1']
        img_m_ = data['img2']
        img2_mask_ = data.get('img2_mask', None)
        
        # Convert to numpy format (like cv2.imread with IMREAD_GRAYSCALE)
        img_r_np = tensor2img(img_r)
        img_m_np = tensor2img(img_m_)
        # Ensure they are in cv2 grayscale format: (H, W), uint8, [0, 255]
        img_r_np = ensure_cv2_grayscale_format(img_r_np)
        img_m_np = ensure_cv2_grayscale_format(img_m_np)
        
        if img2_mask_ is None:
            img2_mask_np = np.zeros_like(img_m_np, dtype=np.uint8)
        else:
            img2_mask_np = tensor2img(img2_mask_)
            img2_mask_np = ensure_cv2_grayscale_format(img2_mask_np)
        
        # Generate field and warp using numpy format
        gpe_field = gpe_generate(img_r_np, img_m_np, img_m_np, img2_mask_np)
        img_m_np = warp_c(img_m_np, gpe_field)
        img2_mask_np = warp_c(img2_mask_np, gpe_field)
        
        # Convert back to tensor using img2tensor
        img_m = img2tensor(img_m_np)
        img2_mask = img2tensor(img2_mask_np)
        
        # Match shapes to original tensors
        # img_m should match img_r shape (including channels)
        img_m = match_tensor_shape(img_m, img_r.shape)
        # img2_mask should be [B, 1, H, W] (single channel), only match batch and spatial dims
        img2_mask = match_mask_shape(img2_mask, img_r.shape)
        
        # Verify shapes match before concatenation
        assert img_r.shape == img_m.shape, f"Shape mismatch: img_r {img_r.shape} vs img_m {img_m.shape}"
        # img2_mask should be [B, 1, H, W], matching batch and spatial dims of img_r
        assert img2_mask.shape[0] == img_r.shape[0] and img2_mask.shape[1] == 1 and \
               img2_mask.shape[2] == img_r.shape[2] and img2_mask.shape[3] == img_r.shape[3], \
               f"Mask shape mismatch: img2_mask {img2_mask.shape} vs expected [B={img_r.shape[0]}, 1, H={img_r.shape[2]}, W={img_r.shape[3]}]"
        
        # read data to device
        img1, img2 = img_r.to(device), img_m.to(device)
        img_pair = torch.cat([img1, img2], 1)
        img2_mask = img2_mask.to(device)       


        am_data_time.update(time.time() - end)

        # run 1st pass
        res_dict = model(img_pair, with_bk=True)
        flows_12, flows_21 = res_dict['flows_fw'], res_dict['flows_bw']
        flows = [torch.cat([flo12, flo21], 1) for flo12, flo21 in zip(flows_12, flows_21)]
        loss, l_ph, l_sm, flow_mean = loss_func(flows, img_pair, img2_mask)

        flow_ori = res_dict['flows_fw'][0].detach()

        if cfg.run_atst:
            # true
            img1, img2 = data['img1_ph'].to(device), data['img2_ph'].to(device)

            # construct augment sample
            noc_ori = loss_func.pyramid_occu_mask1[0]  # non-occluded region
            s = {'imgs': [img1, img2], 'flows_f': [flow_ori], 'masks_f': [noc_ori]}
            st_res = sp_transform(deepcopy(s)) if cfg.run_st else deepcopy(s)  # true
            flow_t, noc_t = st_res['flows_f'][0], st_res['masks_f'][0]

            # run 2nd pass
            img_pair = torch.cat(st_res['imgs'], 1)
            flow_t_pred = model(img_pair, with_bk=False)['flows_fw'][0]

            if not cfg.mask_st:
                # False
                noc_t = torch.ones_like(noc_t)
            l_atst = ((flow_t_pred - flow_t).abs() + cfg.ar_eps) ** cfg.ar_q
            l_atst = (l_atst * noc_t).mean() / (noc_t.mean() + 1e-7)

            loss += cfg.w_ar * l_atst
        else:
            l_atst = torch.zeros_like(loss)

        if cfg.run_ot:
            # false
            img1, img2 = data['img1_ph'].to(device), data['img2_ph'].to(device)
            # run 3rd pass
            img_pair = torch.cat([img1, img2], 1)

            # random crop images
            img_pair, flow_t, occ_t = random_crop(img_pair, flow_ori, 1 - noc_ori, cfg.ot_size)

            # slic 200, random select 8~16
            if cfg.ot_slic:
                img2 = img_pair[:, 3:]
                seg_mask = run_slic_pt(img2, n_seg=200,
                                       compact=cfg.ot_compact, rd_select=[8, 16],
                                       fast=cfg.ot_fast).type_as(img2)  # Nx1xHxW
                noise = torch.rand(img2.size()).type_as(img2)
                img2 = img2 * (1 - seg_mask) + noise * seg_mask
                img_pair[:, 3:] = img2

            flow_t_pred = model(img_pair, with_bk=False)['flows_fw'][0]
            noc_t = 1 - occ_t
            l_ot = ((flow_t_pred - flow_t).abs() + cfg.ar_eps) ** cfg.ar_q
            l_ot = (l_ot * noc_t).mean() / (noc_t.mean() + 1e-7)

            loss += cfg.w_ar * l_ot
        else:
            l_ot = torch.zeros_like(loss)

        # update meters
        key_meters.update(
            [loss.item(), l_ph.item(), l_sm.item(), flow_mean.item(),
             l_atst.item(), l_ot.item()],
            img_pair.size(0))

        # compute gradient and do optimization step
        optimizer.zero_grad()
        # loss.backward()

        scaled_loss = 1024. * loss
        scaled_loss.backward()

        for param in [p for p in model.parameters() if p.requires_grad]:
            if param.grad is not None:
                param.grad.data.mul_(1. / 1024)

        # clip gradients
        nn.utils.clip_grad_norm_(model.parameters(), 10)
        optimizer.step()

        # measure elapsed time
        am_batch_time.update(time.time() - end)
        end = time.time()
        if i_step % 10 == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {am_batch_time.val[0]:.3f} ({am_batch_time.avg[0]:.3f})\t'
                  'Data {am_data_time.val[0]:.3f} ({am_data_time.avg[0]:.3f})\t'
                  'Loss {key_meters.val[0]:.4f} ({key_meters.avg[0]:.4f})\t'
                  'l_ph {key_meters.val[1]:.4f} ({key_meters.avg[1]:.4f})\t'
                  'l_sm {key_meters.val[2]:.4f} ({key_meters.avg[2]:.4f})\t'
                  'flow_mean {key_meters.val[3]:.4f} ({key_meters.avg[3]:.4f})\t'
                  'l_atst {key_meters.val[4]:.4f} ({key_meters.avg[4]:.4f})\t'
                  'l_ot {key_meters.val[5]:.4f} ({key_meters.avg[5]:.4f})'.format(
                epoch, i_step, len(train_loader), am_batch_time=am_batch_time,
                am_data_time=am_data_time, key_meters=key_meters))
    print(
        'Epoch: [{0}][{1}/{2}]\t, Time {batch_time.val[0]:.3f} ({batch_time.avg[0]:.3f})\t, Data {data_time.val[0]:.3f} '
        '({data_time.avg[0]:.3f})\t, {key_meter}'.format(epoch, len(train_loader), len(train_loader),
                                                         batch_time=am_batch_time, data_time=am_data_time,
                                                         key_meter=key_meters))
    return key_meters

def valid(cfg, model, valid_loader, epoch):
    with torch.no_grad():
        model.eval()
        ssim_meter = AverageMeter()
        ncc_meter = AverageMeter()
        for i_step, data in enumerate(valid_loader):
            img1, img2 = data['img1'].to(cfg.device), data['img2'].to(cfg.device)
            img_pair = torch.cat([img1, img2], 1)
            flow_t_pred = model(img_pair, with_bk=False)['flows_fw'][0]
            # warp img2 to img1
            img2_warp = flow_warp(img2, flow_t_pred)
            # convert to numpy
            img1, img2, img2_warp = img1[0, 0].cpu().numpy(), img2[0, 0].cpu().numpy(), img2_warp[0, 0].cpu().numpy()
            # compute metrics
            ssim_meter.update(compute_ssim(img1, img2_warp).item())
            ncc_meter.update(compute_ncc(img1, img2_warp).item())
           
    print('Valid Epoch: [{0}][{1}/{2}]\t, SSIM {ssim:.4f}\t, NCC {ncc:.4f}'.format(0, len(valid_loader),
                                                                                   len(valid_loader),
                                                                                   ssim=ssim_meter.avg[0],
                                                                                   ncc=ncc_meter.avg[0]))

def train(cfg, model, loss_func, optimizer, train_loader, valid_loader, test_loader):
    model = model.to(cfg.device)
    # create directory of rst/
    rst_file = "rst/{}_rst.csv".format(cfg.name)
    if not os.path.exists("rst/" + cfg.name):
        os.makedirs("rst/" + cfg.name)
    for epoch in range(cfg.train.epoch_num):
        key_meters = train_one_epoch(cfg.train, model, loss_func, optimizer, train_loader, cfg.device,
                                     epoch)
        
        torch.save(model.state_dict(), 'models/{}_{}.pth'.format(cfg.name, epoch))
        pair_rst = ext_call_pairwise(cfg.name, 'models/{}_{}.pth'.format(cfg.name, epoch), cfg.device)

        with open(rst_file, 'a') as f:
            f.write("{},{},{}\n".format(cfg.name, 'models/{}_{}.pth'.format(cfg.name, epoch), pair_rst))


configs_lst = ["FE_ft_c"]
if __name__ == '__main__':
    import warnings

    warnings.filterwarnings('ignore')
    # create a rst.csv
    for cfg_name in configs_lst:
        cfg_path = 'configs/{}.json'.format(cfg_name)
        print('Loading {}'.format(cfg_path))
        init_seed(0)
        with open(cfg_path) as f:
            cfg = EasyDict(json.load(f))

        train_loader, valid_loader, test_loader = get_dataloader(cfg)

        model = PWCLite(cfg.model)
        model = restore_model(model, cfg.train.pretrained_model)
        loss = get_loss(cfg.loss)
        optimizer = torch.optim.Adam(model.parameters(), cfg.train.lr, betas=(cfg.train.momentum, cfg.train.beta),
                                     eps=1e-7)
                                     
        train(cfg, model, loss, optimizer, train_loader, valid_loader, test_loader)
        torch.save(model.state_dict(), 'models/{}.pth'.format(cfg.name))
