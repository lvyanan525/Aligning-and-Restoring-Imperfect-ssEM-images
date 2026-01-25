import torch
import torch.nn as nn
import torch.nn.functional as F

def TernaryLoss(im, im_warp, max_distance=1):
    patch_size = 2 * max_distance + 1

    def _rgb_to_grayscale(image):
        grayscale = image[:, 0, :, :] * 0.2989 + \
                    image[:, 1, :, :] * 0.5870 + \
                    image[:, 2, :, :] * 0.1140
        return grayscale.unsqueeze(1)

    def _ternary_transform(image):
        intensities = _rgb_to_grayscale(image) * 255
        out_channels = patch_size * patch_size
        w = torch.eye(out_channels).view((out_channels, 1, patch_size, patch_size))
        weights = w.type_as(im)
        patches = F.conv2d(intensities, weights, padding=max_distance)
        transf = patches - intensities
        transf_norm = transf / torch.sqrt(0.81 + torch.pow(transf, 2))
        return transf_norm

    def _hamming_distance(t1, t2):
        dist = torch.pow(t1 - t2, 2)
        dist_norm = dist / (0.1 + dist)
        dist_mean = torch.mean(dist_norm, 1, keepdim=True)  # instead of sum
        return dist_mean

    def _valid_mask(t, padding):
        n, _, h, w = t.size()
        inner = torch.ones(n, 1, h - 2 * padding, w - 2 * padding).type_as(t)
        mask = F.pad(inner, [padding] * 4)
        return mask

    t1 = _ternary_transform(im)
    t2 = _ternary_transform(im_warp)
    dist = _hamming_distance(t1, t2)
    mask = _valid_mask(im, max_distance)

    return dist * mask


def SSIM(x, y, md=1):
    patch_size = 2 * md + 1
    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    mu_x = nn.AvgPool2d(patch_size, 1, 0)(x)
    mu_y = nn.AvgPool2d(patch_size, 1, 0)(y)
    mu_x_mu_y = mu_x * mu_y
    mu_x_sq = mu_x.pow(2)
    mu_y_sq = mu_y.pow(2)

    sigma_x = nn.AvgPool2d(patch_size, 1, 0)(x * x) - mu_x_sq
    sigma_y = nn.AvgPool2d(patch_size, 1, 0)(y * y) - mu_y_sq
    sigma_xy = nn.AvgPool2d(patch_size, 1, 0)(x * y) - mu_x_mu_y

    SSIM_n = (2 * mu_x_mu_y + C1) * (2 * sigma_xy + C2)
    SSIM_d = (mu_x_sq + mu_y_sq + C1) * (sigma_x + sigma_y + C2)
    SSIM = SSIM_n / SSIM_d
    dist = torch.clamp((1 - SSIM) / 2, 0, 1)
    return dist


def gradient(data):
    D_dy = data[:, :, 1:] - data[:, :, :-1]
    D_dx = data[:, :, :, 1:] - data[:, :, :, :-1]
    return D_dx, D_dy


def smooth_grad_1st(flo, image, alpha, mask=None):
    img_dx, img_dy = gradient(image)
    weights_x = torch.exp(-torch.mean(torch.abs(img_dx), 1, keepdim=True) * alpha)
    weights_y = torch.exp(-torch.mean(torch.abs(img_dy), 1, keepdim=True) * alpha)

    dx, dy = gradient(flo)

    loss_x = weights_x * dx.abs() / 2.
    loss_y = weights_y * dy.abs() / 2

    # Apply mask: set loss to 0 where mask is 0
    if mask is not None:
        # mask shape: [B, 1, H, W]
        # loss_x shape: [B, C, H, W-1] (after gradient)
        # loss_y shape: [B, C, H-1, W] (after gradient)
        # Compute mask gradients to match loss dimensions
        mask = 1-mask
        # mask_dx, mask_dy = gradient(mask)
        # Average mask over channels and take mean to get [B, 1, H, W-1] and [B, 1, H-1, W]
        mask_x = (mask[:, :, :, :-1] + mask[:, :, :, 1:]) / 2.0  # [B, 1, H, W-1]
        mask_y = (mask[:, :, :-1, :] + mask[:, :, 1:, :]) / 2.0  # [B, 1, H-1, W]
        # Expand to match loss dimensions
        if loss_x.shape[1] > 1:
            mask_x = mask_x.expand(-1, loss_x.shape[1], -1, -1)
        if loss_y.shape[1] > 1:
            mask_y = mask_y.expand(-1, loss_y.shape[1], -1, -1)
        loss_x = loss_x * mask_x
        loss_y = loss_y * mask_y

    return loss_x.mean() / 2. + loss_y.mean() / 2.


def smooth_grad_2nd(flo, image, alpha, mask=None):
    img_dx, img_dy = gradient(image)
    weights_x = torch.exp(-torch.mean(torch.abs(img_dx), 1, keepdim=True) * alpha)
    weights_y = torch.exp(-torch.mean(torch.abs(img_dy), 1, keepdim=True) * alpha)

    dx, dy = gradient(flo)
    dx2, dxdy = gradient(dx)
    dydx, dy2 = gradient(dy)

    loss_x = weights_x[:, :, :, 1:] * dx2.abs()
    loss_y = weights_y[:, :, 1:, :] * dy2.abs()

    # Apply mask: set loss to 0 where mask is 0
    if mask is not None:
        # mask shape: [B, 1, H, W]
        # loss_x shape: [B, C, H, W-2] (after 2nd order gradient)
        # loss_y shape: [B, C, H-2, W] (after 2nd order gradient)
        # Average mask over the gradient directions for 2nd order
        mask = 1-mask
        mask_x = (mask[:, :, :, :-2] + mask[:, :, :, 1:-1] + mask[:, :, :, 2:]) / 3.0  # [B, 1, H, W-2]
        mask_y = (mask[:, :, :-2, :] + mask[:, :, 1:-1, :] + mask[:, :, 2:, :]) / 3.0  # [B, 1, H-2, W]
        # Expand to match loss dimensions
        if loss_x.shape[1] > 1:
            mask_x = mask_x.expand(-1, loss_x.shape[1], -1, -1)
        if loss_y.shape[1] > 1:
            mask_y = mask_y.expand(-1, loss_y.shape[1], -1, -1)
        loss_x = loss_x * mask_x
        loss_y = loss_y * mask_y

    return loss_x.mean() / 2. + loss_y.mean() / 2.

