import os
import cv2
import glob
import shutil
import argparse
import numpy as np
from PIL import Image
from register.non_linear_align import FlowAligner
import math

def pairwise_align(img_r, img_m, img_mask, aligner):
    field = aligner.generate_field(img_r, img_m, img_mask)
    img_w = aligner.warp_with_field(img_m, field)
    img_mask_w = aligner.warp_with_field(img_mask, field, "nearest")
    return img_w, img_mask_w


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference", type=str, default="./DATA/0000.tif")
    parser.add_argument("--moving", type=str, default="./DATA/0001.tif")
    parser.add_argument("--mask", type=str, default="./DATA/0002.tif")
    parser.add_argument("--model_path", type=str, default='./models/FE_final.pth')
    parser.add_argument("--output", type=str, default='./test2_aligned.tif')
    parser.add_argument("--output_mask", type=str, default='./test2_aligned_mask.tif')
    args = parser.parse_args()

    aligner = FlowAligner(args.model_path, True, "cuda")
    img_r_ = np.array(Image.open(args.reference,))
    img_m_ = np.array(Image.open(args.moving))
    img_mask_ = np.array(Image.open(args.mask))

    hw = max(img_r_.shape)
    hw = math.ceil(hw/64)*64
    img_r = np.zeros((hw, hw), dtype=img_r_.dtype)
    img_r[0:img_r_.shape[0], 0:img_r_.shape[1]] = img_r_
    img_m = np.zeros((hw, hw), dtype=img_m_.dtype)
    img_m[0:img_m_.shape[0], 0:img_m_.shape[1]] = img_m_
    img_mask = np.zeros((hw, hw), dtype=img_mask_.dtype)
    img_mask[0:img_mask_.shape[0], 0:img_mask_.shape[1]] = img_mask_
   
    img_w, img_mask_w = pairwise_align(img_r, img_m, img_mask, aligner)
   
    img_w_ = np.zeros(img_r_.shape, dtype=img_r_.dtype)
    img_w_ = img_w[0:img_r_.shape[0], 0:img_r_.shape[1]]
    img_mask_w_ = np.zeros(img_mask_.shape, dtype=img_mask_.dtype)
    img_mask_w_ = img_mask_w[0:img_r_.shape[0], 0:img_r_.shape[1]]
    cv2.imwrite(args.output, img_w_)
    cv2.imwrite(args.output_mask, img_mask_w_)

