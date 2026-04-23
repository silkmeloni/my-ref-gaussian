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

from scene.cameras import Camera
import numpy as np
import os, cv2, torch
from utils.general_utils import PILtoTorch
from utils.graphics_utils import fov2focal

WARNED = False

def _resolve_prior_dir(scene_path, prior_dir):
    if not prior_dir:
        return ""
    if os.path.isabs(prior_dir):
        return prior_dir
    scene_prior_dir = os.path.join(scene_path, prior_dir)
    if os.path.exists(scene_prior_dir):
        return scene_prior_dir

    scene_name = os.path.basename(os.path.normpath(scene_path))
    scene_parent = os.path.dirname(os.path.normpath(scene_path))
    parent_prior_dir = os.path.join(scene_parent, prior_dir)
    if os.path.exists(parent_prior_dir):
        return parent_prior_dir

    full_scene_dir = os.path.join(scene_parent, f"{scene_name}_full")
    full_prior_dir = os.path.join(full_scene_dir, prior_dir)
    if os.path.exists(full_prior_dir):
        return full_prior_dir

    return scene_prior_dir

_MONO_DEBUG_STATS = {}

def _mono_debug_log(args, prior_type, image_path, prior_dir, prior_path, candidates):
    if not getattr(args, "mono_debug", False):
        return
    stats = _MONO_DEBUG_STATS.setdefault(prior_type, {"loaded": 0, "missing": 0})
    if prior_path is None:
        stats["missing"] += 1
        if stats["missing"] <= 20:
            print(f"[MonoPrior][missing {prior_type}] image={image_path}")
            print(f"[MonoPrior][missing {prior_type}] dir={prior_dir}")
            print(f"[MonoPrior][missing {prior_type}] tried={', '.join(candidates)}")
    else:
        stats["loaded"] += 1
        if stats["loaded"] <= 20:
            print(f"[MonoPrior][loaded {prior_type}] {prior_path}")

def _prior_name_candidates(scene_path, image_path, image_name):
    candidates = []
    try:
        rel_path = os.path.relpath(image_path, scene_path)
        rel_stem = os.path.splitext(rel_path)[0]
        flat_rel_stem = "_".join(part for part in rel_stem.split(os.sep) if part and part != ".")
        if flat_rel_stem:
            candidates.append(flat_rel_stem)
    except ValueError:
        pass

    basename_stem = os.path.splitext(os.path.basename(image_path))[0]
    candidates.extend([image_name, basename_stem])

    unique_candidates = []
    for candidate in candidates:
        if candidate and candidate not in unique_candidates:
            unique_candidates.append(candidate)
    return unique_candidates

def _find_prior_file(scene_path, prior_dir, image_path, image_name, extensions):
    if not prior_dir:
        return None, []
    tried = []
    for base in _prior_name_candidates(scene_path, image_path, image_name):
        for ext in extensions:
            path = os.path.join(prior_dir, base + ext)
            tried.append(path)
            if os.path.exists(path):
                return path, tried
    return None, tried

def _resize_single_channel(tensor, resolution):
    tensor = tensor[None]
    tensor = torch.nn.functional.interpolate(tensor, size=(resolution[1], resolution[0]), mode="bilinear", align_corners=False)
    return tensor[0]

def _load_mono_depth(args, cam_info, resolution):
    prior_dir = _resolve_prior_dir(args.source_path, args.mono_depth_dir)
    path, tried = _find_prior_file(
        args.source_path,
        prior_dir,
        cam_info.image_path,
        cam_info.image_name,
        (".npy", ".png")
    )
    _mono_debug_log(args, "depth", cam_info.image_path, prior_dir, path, tried)
    if path is None:
        return None
    if path.lower().endswith(".npy"):
        depth = np.load(path).astype(np.float32)
    elif path.lower().endswith(".npz"):
        data = np.load(path)
        key = data.files[0]
        depth = data[key].astype(np.float32)
    else:
        depth = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        if depth is None:
            return None
        depth = depth.astype(np.float32)
        if depth.ndim == 3:
            depth = depth[..., 0]
        if depth.max() > 1.0:
            depth = depth / 255.0 if depth.max() <= 255.0 else depth / 65535.0
    depth = np.nan_to_num(depth, nan=0.0, posinf=0.0, neginf=0.0)
    depth = torch.from_numpy(depth).float()
    if depth.ndim == 2:
        depth = depth[None]
    elif depth.ndim == 3 and depth.shape[-1] == 1:
        depth = depth.permute(2, 0, 1)
    elif depth.ndim == 3 and depth.shape[-1] == 3:
        depth = depth[..., :1].permute(2, 0, 1)
    else:
        depth = depth[:1]
    return _resize_single_channel(depth, resolution)

def _load_mono_normal(args, cam_info, resolution):
    prior_dir = _resolve_prior_dir(args.source_path, args.mono_normal_dir)
    path, tried = _find_prior_file(
        args.source_path,
        prior_dir,
        cam_info.image_path,
        cam_info.image_name,
        (".png", ".npy")
    )
    _mono_debug_log(args, "normal", cam_info.image_path, prior_dir, path, tried)
    if path is None:
        return None
    if path.lower().endswith(".npy"):
        normal = np.load(path).astype(np.float32)
    elif path.lower().endswith(".npz"):
        data = np.load(path)
        key = data.files[0]
        normal = data[key].astype(np.float32)
    else:
        normal = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        if normal is None:
            return None
        normal = cv2.cvtColor(normal, cv2.COLOR_BGR2RGB).astype(np.float32)
        if normal.max() > 1.0:
            normal = normal / 255.0 if normal.max() <= 255.0 else normal / 65535.0
    normal = np.nan_to_num(normal, nan=0.0, posinf=0.0, neginf=0.0)
    normal = torch.from_numpy(normal).float()
    if normal.ndim == 3 and normal.shape[-1] == 3:
        normal = normal.permute(2, 0, 1)
    if normal.ndim != 3 or normal.shape[0] != 3:
        return None
    if normal.min() >= 0.0 and normal.max() <= 1.0:
        normal = normal * 2.0 - 1.0
    normal = torch.nn.functional.interpolate(normal[None], size=(resolution[1], resolution[0]), mode="bilinear", align_corners=False)[0]
    if args.mono_normal_flip_x:
        normal[0] = -normal[0]
    if args.mono_normal_flip_y:
        normal[1] = -normal[1]
    if args.mono_normal_flip_z:
        normal[2] = -normal[2]
    return torch.nn.functional.normalize(normal, dim=0, eps=1e-6)

def loadCam(args, id, cam_info, resolution_scale):
    orig_w, orig_h = cam_info.image.size

    if args.resolution in [1, 2, 4, 8]:
        resolution = round(orig_w/(resolution_scale * args.resolution)), round(orig_h/(resolution_scale * args.resolution))
        scale = float(resolution_scale * args.resolution)
    else:  # should be a type that converts to float
        if args.resolution == -1:
            if orig_w > 1600 and False:
                global WARNED
                if not WARNED:
                    print("[ INFO ] Encountered quite large input images (>1.6K pixels width), rescaling to 1.6K.\n "
                        "If this is not desired, please explicitly specify '--resolution/-r' as 1")
                    WARNED = True
                global_down = orig_w / 1600
            else:
                global_down = 1
        else:
            global_down = orig_w / args.resolution

        scale = float(global_down) * float(resolution_scale)
        resolution = (int(orig_w / scale), int(orig_h / scale))
    HWK = None  # #
    if cam_info.K is not None:
        K = cam_info.K.copy()
        K[:2] = K[:2] / scale
        HWK = (resolution[1], resolution[0], K)

    if len(cam_info.image.split()) > 3:
        import torch
        resized_image_rgb = torch.cat([PILtoTorch(im, resolution) for im in cam_info.image.split()[:3]], dim=0)
        loaded_mask = PILtoTorch(cam_info.image.split()[3], resolution)
        gt_image = resized_image_rgb
    else:
        resized_image_rgb = PILtoTorch(cam_info.image, resolution)
        loaded_mask = None
        gt_image = resized_image_rgb
    # #
    refl_path = os.path.join(
        os.path.dirname(os.path.dirname(cam_info.image_path)), 'image_msk')
    refl_path = os.path.join(refl_path, os.path.basename(cam_info.image_path))
    if not os.path.exists(refl_path):
        refl_path = refl_path.replace('.JPG', '.jpg')
    if os.path.exists(refl_path):
        refl_msk = cv2.imread(refl_path) != 0 # max == 1
        refl_msk = torch.tensor(refl_msk).permute(2,0,1).float()
    else: refl_msk = None
    mono_depth = _load_mono_depth(args, cam_info, resolution) if args.mono_depth_dir else None
    mono_normal = _load_mono_normal(args, cam_info, resolution) if args.mono_normal_dir else None

    return Camera(colmap_id=cam_info.uid, R=cam_info.R, T=cam_info.T, 
                  FoVx=cam_info.FovX, FoVy=cam_info.FovY, 
                  image=gt_image, gt_alpha_mask=loaded_mask,
                  image_name=cam_info.image_name, uid=id, 
                  data_device=args.data_device, HWK=HWK, gt_refl_mask=refl_msk,
                  mono_depth=mono_depth, mono_normal=mono_normal)

def cameraList_from_camInfos(cam_infos, resolution_scale, args):
    camera_list = []

    if getattr(args, "mono_debug", False):
        _MONO_DEBUG_STATS.clear()

    for id, c in enumerate(cam_infos):
        camera_list.append(loadCam(args, id, c, resolution_scale))

    if getattr(args, "mono_debug", False):
        for prior_type, stats in _MONO_DEBUG_STATS.items():
            total = stats["loaded"] + stats["missing"]
            print(
                f"[MonoPrior][summary {prior_type}] "
                f"loaded={stats['loaded']} missing={stats['missing']} total={total}"
            )

    return camera_list

def camera_to_JSON(id, camera : Camera):
    Rt = np.zeros((4, 4))
    Rt[:3, :3] = camera.R.transpose()
    Rt[:3, 3] = camera.T
    Rt[3, 3] = 1.0

    W2C = np.linalg.inv(Rt)
    pos = W2C[:3, 3]
    rot = W2C[:3, :3]
    serializable_array_2d = [x.tolist() for x in rot]
    camera_entry = {
        'id' : id,
        'img_name' : camera.image_name,
        'width' : camera.width,
        'height' : camera.height,
        'position': pos.tolist(),
        'rotation': serializable_array_2d,
        'fy' : fov2focal(camera.FovY, camera.height),
        'fx' : fov2focal(camera.FovX, camera.width)
    }
    return camera_entry
