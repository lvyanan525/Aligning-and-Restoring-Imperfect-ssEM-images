
import os
import cv2
import glob
import shutil
import argparse
import numpy as np
from PIL import Image
from register.non_linear_align import FlowAligner

def compute_ncc(img1, img2):
    img1 = img1.astype(np.float32)
    img2 = img2.astype(np.float32)
    img1 = (img1 - img1.mean()) / img1.std()
    img2 = (img2 - img2.mean()) / img2.std()
    return np.corrcoef(img1.flatten(), img2.flatten())[0, 1]

def pairwise_align(img_r, img_m, img_mask, aligner, label=None):
    field = aligner.generate_field(img_r, img_m, img_mask)
    img_w = aligner.warp_with_field(img_m, field.clone())
    if label is not None:
        label = aligner.warp_with_field(label, field.clone(), "nearest")
        return img_w, label
    else:
        return img_w


def batch_align(input_dir, aligner, output_dir):
    ncc_r = []
    path_r_lst = glob.glob(os.path.join(input_dir, '*reference.tif'))
    path_m_lst = glob.glob(os.path.join(input_dir, "*moving.tif"))
    path_mask_lst = glob.glob(os.path.join(input_dir, "*mask.tif"))
   
    path_r_lst.sort()
    path_m_lst.sort()
    path_mask_lst.sort()
    # Create a dictionary for quick mask lookup
    mask_dict = {os.path.basename(path): path for path in path_mask_lst}
   
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir)
    # Iterate based on path_r_lst length, mask is optional
    for path_r, path_m in zip(path_r_lst, path_m_lst):
        img_r = np.array(Image.open(path_r))
        img_m = np.array(Image.open(path_m))
        # Try to find corresponding mask file
        mask_filename = os.path.basename(path_r).replace('reference', 'mask')
        path_mask = mask_dict.get(mask_filename, None)
        if path_mask is not None and os.path.exists(path_mask):
            img_mask = np.array(Image.open(path_mask))
        else:
            img_mask = np.zeros_like(img_m, dtype=np.uint8) * 255

        img_w = pairwise_align(img_r, img_m, img_mask,aligner)
        cv2.imwrite(os.path.join(output_dir, os.path.basename(path_r).replace('reference', 'w')), img_w)
        shutil.copyfile(path_r, os.path.join(output_dir, os.path.basename(path_r).replace('reference', 'r')))
        shutil.copyfile(path_m, os.path.join(output_dir, os.path.basename(path_r).replace('reference', 'm')))
        ncc_r.append(compute_ncc(img_r, img_w))       
    print("ncc between img_r and img_w: {:.3f}".format(np.mean(ncc_r)))    
    return np.mean(ncc_r)


def ext_call_pairwise(name, model_path, device):
    path = os.path.dirname(os.path.dirname(__file__))
    input_dir = "/mnt/Ext001/xint/DATA/opticalData/CREMI_A_valid/"
    output_dir = path + '/rst/{}'.format(name)
    is_c = True if name[-1] == 'c' else False
    aligner = FlowAligner(model_path, is_c, device)
    return batch_align(input_dir, aligner, output_dir)



model_lst = {'ARFlow_c': 'pwclite_ar.tar',
             'ARFlow_ft_c': 'ARFlow_ft_c_9.pth',
             'FE_ft_c': 'FE_pairwise.pth'}
