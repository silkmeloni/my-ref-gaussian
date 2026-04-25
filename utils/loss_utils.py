#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import torch
import torch.nn.functional as F
from torch.autograd import Variable
from math import cos, exp, radians
from kornia.filters import spatial_gradient
from .image_utils import psnr
from utils.image_utils import erode
import numpy as np

def l1_loss(network_output, gt):
    return torch.abs((network_output - gt)).mean()

def l2_loss(network_output, gt):
    return ((network_output - gt) ** 2).mean()

def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
    return gauss / gauss.sum()


def smooth_loss(disp, img):
    grad_disp_x = torch.abs(disp[:,1:-1, :-2] + disp[:,1:-1,2:] - 2 * disp[:,1:-1,1:-1])
    grad_disp_y = torch.abs(disp[:,:-2, 1:-1] + disp[:,2:,1:-1] - 2 * disp[:,1:-1,1:-1])
    grad_img_x = torch.mean(torch.abs(img[:, 1:-1, :-2] - img[:, 1:-1, 2:]), 0, keepdim=True) * 0.5
    grad_img_y = torch.mean(torch.abs(img[:, :-2, 1:-1] - img[:, 2:, 1:-1]), 0, keepdim=True) * 0.5
    grad_disp_x *= torch.exp(-grad_img_x)
    grad_disp_y *= torch.exp(-grad_img_y)
    return grad_disp_x.mean() + grad_disp_y.mean()

def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
    return window

def ssim(img1, img2, window_size=11, size_average=True):
    channel = img1.size(-3)
    window = create_window(window_size, channel)

    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)

    return _ssim(img1, img2, window, window_size, channel, size_average)

def _ssim(img1, img2, window, window_size, channel, size_average=True):
    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)

def first_order_edge_aware_loss(data, img):
    return (spatial_gradient(data[None], order=1)[0].abs() * torch.exp(-spatial_gradient(img[None], order=1)[0].abs())).sum(1).mean()

def _masked_mean(value, mask, fallback):
    if mask.sum().item() == 0:
        return torch.zeros_like(fallback)
    return value[mask].mean()

def _normalize_with_mask(value, mask, min_pixels):
    if mask.sum().item() < min_pixels:
        return None
    valid = value[mask]
    center = valid.median().detach()
    scale = (valid - center).abs().mean().detach().clamp_min(1e-6)
    return (value - center) / scale

def _mono_prior_mask(opacity, opt, gt_alpha_mask=None):
    valid = opacity.detach() > opt.mono_prior_alpha_thr
    if gt_alpha_mask is not None:
        gt_alpha_mask = gt_alpha_mask.to(opacity.device)
        valid = valid & (gt_alpha_mask > opt.mono_prior_alpha_thr)
    return valid

def mono_depth_edge_weights(mono_disp_norm, valid, opt):
    if not getattr(opt, "mono_depth_edge_mask", False):
        return torch.ones_like(mono_disp_norm)

    min_weight = float(getattr(opt, "mono_depth_edge_mask_min_weight", 0.25))
    min_weight = max(0.0, min(1.0, min_weight))
    sigma = float(getattr(opt, "mono_depth_edge_mask_sigma", 1.0))
    sigma = max(1e-6, sigma)

    mono_disp_norm = mono_disp_norm.detach()
    valid = valid.detach()
    edge = torch.zeros_like(mono_disp_norm)
    count = torch.zeros_like(mono_disp_norm)

    dx = (mono_disp_norm[:, :, 1:] - mono_disp_norm[:, :, :-1]).abs()
    mask_dx = valid[:, :, 1:] & valid[:, :, :-1]
    dx = dx * mask_dx.float()
    edge[:, :, 1:] += dx
    edge[:, :, :-1] += dx
    count[:, :, 1:] += mask_dx.float()
    count[:, :, :-1] += mask_dx.float()

    dy = (mono_disp_norm[:, 1:, :] - mono_disp_norm[:, :-1, :]).abs()
    mask_dy = valid[:, 1:, :] & valid[:, :-1, :]
    dy = dy * mask_dy.float()
    edge[:, 1:, :] += dy
    edge[:, :-1, :] += dy
    count[:, 1:, :] += mask_dy.float()
    count[:, :-1, :] += mask_dy.float()

    edge = edge / count.clamp_min(1.0)
    if valid.sum().item() < getattr(opt, "mono_prior_min_pixels", 32):
        return torch.ones_like(mono_disp_norm)

    edge_valid = edge[valid]
    edge_scale = torch.quantile(edge_valid, 0.90).detach().clamp_min(1e-6)
    edge_score = edge / edge_scale
    weights = min_weight + (1.0 - min_weight) * torch.exp(-edge_score / sigma)
    return weights.clamp(min_weight, 1.0)

def _weighted_masked_mean(value, mask, weights, fallback):
    if mask.sum().item() == 0:
        return torch.zeros_like(fallback)
    valid_weights = weights[mask]
    denom = valid_weights.sum().clamp_min(1e-6)
    return (value[mask] * valid_weights).sum() / denom

def mono_normal_angle_valid(cosine, valid, opt):
    if not getattr(opt, "mono_normal_angle_mask", False):
        return valid
    threshold = float(getattr(opt, "mono_normal_angle_thr", 60.0))
    threshold = max(0.0, min(180.0, threshold))
    cos_threshold = cos(radians(threshold))
    return valid & (cosine.detach() >= cos_threshold)

def _mono_depth_loss(rendered_depth, mono_depth, opacity, opt, loss_ref, gt_alpha_mask=None):
    if mono_depth is None:
        return torch.zeros_like(loss_ref), torch.zeros_like(loss_ref)
    mono_depth = mono_depth.to(rendered_depth.device)
    render_disp = 1.0 / rendered_depth.clamp_min(1e-6)
    valid = torch.isfinite(mono_depth) & torch.isfinite(render_disp) & (mono_depth > 0)
    valid = valid & _mono_prior_mask(opacity, opt, gt_alpha_mask)
    render_disp_norm = _normalize_with_mask(render_disp, valid, opt.mono_prior_min_pixels)
    mono_disp_norm = _normalize_with_mask(mono_depth, valid, opt.mono_prior_min_pixels)
    if render_disp_norm is None or mono_disp_norm is None:
        return torch.zeros_like(loss_ref), torch.zeros_like(loss_ref)
    depth_weights = mono_depth_edge_weights(mono_disp_norm, valid, opt)
    depth_loss = _weighted_masked_mean((render_disp_norm - mono_disp_norm).abs(), valid, depth_weights, loss_ref)

    grad_loss = torch.zeros_like(loss_ref)
    if opt.lambda_mono_depth_grad > 0:
        render_dx = render_disp_norm[:, :, 1:] - render_disp_norm[:, :, :-1]
        mono_dx = mono_disp_norm[:, :, 1:] - mono_disp_norm[:, :, :-1]
        mask_dx = valid[:, :, 1:] & valid[:, :, :-1]
        weight_dx = 0.5 * (depth_weights[:, :, 1:] + depth_weights[:, :, :-1])
        render_dy = render_disp_norm[:, 1:, :] - render_disp_norm[:, :-1, :]
        mono_dy = mono_disp_norm[:, 1:, :] - mono_disp_norm[:, :-1, :]
        mask_dy = valid[:, 1:, :] & valid[:, :-1, :]
        weight_dy = 0.5 * (depth_weights[:, 1:, :] + depth_weights[:, :-1, :])
        grad_terms = []
        if mask_dx.sum().item() >= opt.mono_prior_min_pixels:
            grad_terms.append(_weighted_masked_mean((render_dx - mono_dx).abs(), mask_dx, weight_dx, loss_ref))
        if mask_dy.sum().item() >= opt.mono_prior_min_pixels:
            grad_terms.append(_weighted_masked_mean((render_dy - mono_dy).abs(), mask_dy, weight_dy, loss_ref))
        if grad_terms:
            grad_loss = sum(grad_terms) / len(grad_terms)
    return depth_loss, grad_loss

def _mono_normal_loss(rendered_normal, mono_normal, opacity, opt, loss_ref, gt_alpha_mask=None):
    if mono_normal is None:
        return torch.zeros_like(loss_ref)
    mono_normal = mono_normal.to(rendered_normal.device)
    render_normal = F.normalize(rendered_normal, dim=0, eps=1e-6)
    mono_normal = F.normalize(mono_normal, dim=0, eps=1e-6)
    cosine = (render_normal * mono_normal).sum(dim=0, keepdim=True).clamp(-1.0, 1.0)
    valid = torch.isfinite(cosine) & _mono_prior_mask(opacity, opt, gt_alpha_mask)
    valid = mono_normal_angle_valid(cosine, valid, opt)
    return _masked_mean(1.0 - cosine, valid, loss_ref)



def calculate_loss(viewpoint_camera, pc, render_pkg, opt, iteration):
    tb_dict = {
        "num_points": pc.get_xyz.shape[0],
    }
    
    rendered_image = render_pkg["render"]
    rendered_opacity = render_pkg["rend_alpha"]
    rendered_depth = render_pkg["surf_depth"]
    rendered_normal = render_pkg["rend_normal"]
    visibility_filter = render_pkg["visibility_filter"]
    rend_dist = render_pkg["rend_dist"]
    gt_image = viewpoint_camera.original_image.cuda()

    Ll1 = l1_loss(rendered_image, gt_image)
    ssim_val = ssim(rendered_image, gt_image)
    loss0 = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim_val)
    loss = torch.zeros_like(loss0)
    tb_dict["loss_l1"] = Ll1.item()
    tb_dict["psnr"] = psnr(rendered_image, gt_image).mean().item()
    tb_dict["ssim"] = ssim_val.item()
    tb_dict["loss0"] = loss0.item()
    loss += loss0

    if opt.lambda_normal_render_depth > 0 and iteration > opt.normal_loss_start:
        surf_normal = render_pkg['surf_normal']
        loss_normal_render_depth = (1 - (rendered_normal * surf_normal).sum(dim=0))[None]
        loss_normal_render_depth = loss_normal_render_depth.mean()
        tb_dict["loss_normal_render_depth"] = loss_normal_render_depth
        loss = loss + opt.lambda_normal_render_depth * loss_normal_render_depth
    else:
        tb_dict["loss_normal_render_depth"] = torch.zeros_like(loss)

    if opt.lambda_dist > 0 and iteration > opt.dist_loss_start:
        dist_loss = opt.lambda_dist * rend_dist.mean()
        tb_dict["loss_dist"] = dist_loss
        loss += dist_loss
    else:
        tb_dict["loss_dist"] = torch.zeros_like(loss)

    if opt.lambda_normal_smooth > 0 and iteration > opt.normal_smooth_from_iter and iteration < opt.normal_smooth_until_iter:
        loss_normal_smooth = first_order_edge_aware_loss(rendered_normal, gt_image)
        tb_dict["loss_normal_smooth"] = loss_normal_smooth.item()
        lambda_normal_smooth = opt.lambda_normal_smooth
        loss = loss + lambda_normal_smooth * loss_normal_smooth
    else:
        tb_dict["loss_normal_smooth"] = torch.zeros_like(loss)
    
    if opt.lambda_depth_smooth > 0 and iteration > 3000:
        loss_depth_smooth = first_order_edge_aware_loss(rendered_depth, gt_image)
        tb_dict["loss_depth_smooth"] = loss_depth_smooth.item()
        lambda_depth_smooth = opt.lambda_depth_smooth
        loss = loss + lambda_depth_smooth * loss_depth_smooth
    else:
        tb_dict["loss_depth_smooth"] = torch.zeros_like(loss)

    use_mono_depth = (
        opt.lambda_mono_depth > 0
        and opt.mono_depth_from_iter <= iteration < opt.mono_depth_until_iter
        and hasattr(viewpoint_camera, "mono_depth")
    )
    if use_mono_depth:
        loss_mono_depth, loss_mono_depth_grad = _mono_depth_loss(
            rendered_depth, viewpoint_camera.mono_depth, rendered_opacity, opt, loss,
            getattr(viewpoint_camera, "gt_alpha_mask", None)
        )
        loss = loss + opt.lambda_mono_depth * loss_mono_depth
        if opt.lambda_mono_depth_grad > 0:
            loss = loss + opt.lambda_mono_depth_grad * loss_mono_depth_grad
        tb_dict["loss_mono_depth"] = loss_mono_depth
        tb_dict["loss_mono_depth_grad"] = loss_mono_depth_grad
    else:
        tb_dict["loss_mono_depth"] = torch.zeros_like(loss)
        tb_dict["loss_mono_depth_grad"] = torch.zeros_like(loss)

    use_mono_normal = (
        opt.lambda_mono_normal > 0
        and opt.mono_normal_from_iter <= iteration < opt.mono_normal_until_iter
        and hasattr(viewpoint_camera, "mono_normal")
    )
    if use_mono_normal:
        loss_mono_normal = _mono_normal_loss(
            rendered_normal, viewpoint_camera.mono_normal, rendered_opacity, opt, loss,
            getattr(viewpoint_camera, "gt_alpha_mask", None)
        )
        loss = loss + opt.lambda_mono_normal * loss_mono_normal
        tb_dict["loss_mono_normal"] = loss_mono_normal
    else:
        tb_dict["loss_mono_normal"] = torch.zeros_like(loss)

    
    tb_dict["loss"] = loss.item()
    
    return loss, tb_dict
