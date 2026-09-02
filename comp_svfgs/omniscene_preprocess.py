import json
import os
from typing import Dict, List, Tuple

import numpy as np
import torch
from PIL import Image

from scene.dataset_readers import storePly


PREPARED_FORMAT_VERSION = 2


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _tensor_to_uint8_image(img: torch.Tensor) -> np.ndarray:
    """将 [C,H,W] Tensor 转为 uint8 HWC 图像。"""
    img = img.detach().cpu().clamp(0, 1)
    img = (img * 255.0).to(torch.uint8)
    img = img.permute(1, 2, 0).numpy()
    return img


def _save_images(images: torch.Tensor, names: List[str], dst_dir: str) -> None:
    for img, name in zip(images, names):
        img_np = _tensor_to_uint8_image(img)
        Image.fromarray(img_np).save(os.path.join(dst_dir, name))


def _make_frame(file_path: str, c2w: torch.Tensor, ck: torch.Tensor) -> Dict:
    return {
        "file_path": file_path,
        "transform_matrix": c2w.detach().cpu().tolist(),
        "fl_x": float(ck[0, 0].item()),
        "fl_y": float(ck[1, 1].item()),
        "cx": float(ck[0, 2].item()),
        "cy": float(ck[1, 2].item()),
    }


def _save_transforms(frames: List[Dict], out_path: str) -> None:
    data = {
        "no_flip_yz": True,
        "frames": frames,
    }
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def prepared_scene_complete(
    scene_dir: str,
    bin_token: str,
    resolution: Tuple[int, int],
    conf_threshold: float,
) -> bool:
    """校验预处理缓存是否与当前样本及参数完全一致。"""
    images_dir = os.path.join(scene_dir, "images")
    required_paths = [
        os.path.join(scene_dir, "transforms_train.json"),
        os.path.join(scene_dir, "transforms_test.json"),
        os.path.join(scene_dir, "points3D.ply"),
        os.path.join(scene_dir, "meta.json"),
    ]
    if not os.path.isdir(images_dir) or any(not os.path.isfile(path) for path in required_paths):
        return False
    if any(os.path.getsize(path) == 0 for path in required_paths):
        return False

    try:
        with open(os.path.join(scene_dir, "meta.json"), "r", encoding="utf-8") as meta_file:
            meta = json.load(meta_file)
        if meta.get("format_version") != PREPARED_FORMAT_VERSION:
            return False
        if meta.get("scene") != bin_token:
            return False
        if meta.get("resolution") != [int(resolution[0]), int(resolution[1])]:
            return False
        if abs(float(meta.get("conf_threshold")) - float(conf_threshold)) > 1e-12:
            return False

        with open(required_paths[0], "r", encoding="utf-8") as train_file:
            train_frames = json.load(train_file).get("frames", [])
        with open(required_paths[1], "r", encoding="utf-8") as test_file:
            test_frames = json.load(test_file).get("frames", [])
    except (OSError, TypeError, ValueError, json.JSONDecodeError):
        return False

    if len(train_frames) != 6 or len(test_frames) != 18:
        return False
    image_paths = [frame.get("file_path") for frame in train_frames + test_frames]
    if any(not isinstance(path, str) for path in image_paths):
        return False
    return all(os.path.isfile(os.path.join(scene_dir, path)) for path in image_paths)


def _build_point_cloud(
    images: torch.Tensor,
    depths: torch.Tensor,
    confs: torch.Tensor,
    intrinsics: torch.Tensor,
    c2ws: torch.Tensor,
    conf_threshold: float = 0.3,
) -> Tuple[np.ndarray, np.ndarray]:
    """基于绝对深度与置信度生成点云。"""
    points_list = []
    colors_list = []

    for img, depth, conf, ck, c2w in zip(images, depths, confs, intrinsics, c2ws):
        img_np = _tensor_to_uint8_image(img)
        depth_np = depth.detach().cpu().numpy()
        conf_np = conf.detach().cpu().numpy()
        ck_np = ck.detach().cpu().numpy()
        c2w_np = c2w.detach().cpu().numpy()

        mask = conf_np > conf_threshold
        mask &= depth_np > 0
        if not np.any(mask):
            continue

        h, w = depth_np.shape
        u, v = np.meshgrid(np.arange(w), np.arange(h))
        fx, fy = ck_np[0, 0], ck_np[1, 1]
        cx, cy = ck_np[0, 2], ck_np[1, 2]

        z = depth_np
        x = (u - cx) / fx * z
        y = (v - cy) / fy * z

        pts_cam = np.stack([x, y, z, np.ones_like(z)], axis=-1).reshape(-1, 4)
        mask_flat = mask.reshape(-1)
        pts_cam = pts_cam[mask_flat]

        pts_world = (c2w_np @ pts_cam.T).T[:, :3]
        colors = img_np.reshape(-1, 3)[mask_flat]

        points_list.append(pts_world)
        colors_list.append(colors)

    if not points_list:
        return np.empty((0, 3), dtype=np.float32), np.empty((0, 3), dtype=np.uint8)

    points = np.concatenate(points_list, axis=0).astype(np.float32)
    colors = np.concatenate(colors_list, axis=0).astype(np.uint8)
    return points, colors


def preprocess_scene(
    scene_data: dict,
    output_root: str,
    conf_threshold: float = 0.3,
    prefix: str = "01",
) -> str:
    """将单个 bin 预处理为 Octree-GS 可加载的场景结构。"""
    bin_token = scene_data["scene"]
    scene_name = f"{prefix}_{bin_token}"
    scene_dir = os.path.join(output_root, scene_name)
    images_dir = os.path.join(scene_dir, "images")

    transforms_train = os.path.join(scene_dir, "transforms_train.json")
    transforms_test = os.path.join(scene_dir, "transforms_test.json")
    ply_path = os.path.join(scene_dir, "points3D.ply")

    context = scene_data["context"]
    target = scene_data["target"]

    if prepared_scene_complete(
        scene_dir, bin_token, (int(context["image"].shape[2]), int(context["image"].shape[3])), conf_threshold
    ):
        return scene_dir

    _ensure_dir(images_dir)

    # 保存图像
    train_names = [f"train_{i:03d}.jpg" for i in range(context["image"].shape[0])]
    test_names = [f"test_{i:03d}.jpg" for i in range(target["image"].shape[0])]
    _save_images(context["image"], train_names, images_dir)
    _save_images(target["image"], test_names, images_dir)

    # 生成 transforms
    train_frames = [
        _make_frame(os.path.join("images", name), c2w, ck)
        for name, c2w, ck in zip(train_names, context["extrinsics"], context["intrinsics"])
    ]
    test_frames = [
        _make_frame(os.path.join("images", name), c2w, ck)
        for name, c2w, ck in zip(test_names, target["extrinsics"], target["intrinsics"])
    ]
    _save_transforms(train_frames, transforms_train)
    _save_transforms(test_frames, transforms_test)

    # 点云初始化
    depths = context.get("depths")
    confs = context.get("confs")
    if depths is None or confs is None:
        raise ValueError("未加载到绝对深度或置信度，无法生成点云。")

    points, colors = _build_point_cloud(
        context["image"],
        depths,
        confs,
        context["intrinsics"],
        context["extrinsics"],
        conf_threshold=conf_threshold,
    )
    if points.shape[0] == 0:
        raise ValueError("点云为空，可能是置信度阈值过高或深度缺失。")

    storePly(ply_path, points, colors)

    # 元信息
    meta = {
        "format_version": PREPARED_FORMAT_VERSION,
        "scene": bin_token,
        "prefix": prefix,
        "resolution": [int(context["image"].shape[2]), int(context["image"].shape[3])],
        "conf_threshold": float(conf_threshold),
        "context_views": len(train_frames),
        "target_views": len(test_frames),
        "train_images": train_names,
        "test_images": test_names,
    }
    with open(os.path.join(scene_dir, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    return scene_dir
