import os
import os.path as osp
import json
import pickle as pkl
import copy
from typing import List, Tuple, Optional

import numpy as np
import torch
from PIL import Image

# OmniScene 相关常量
DATA_VERSION = "interp_12Hz_trainval"
DATASET_PREFIX = "/datasets/nuScenes"
CAMERA_TYPES = [
    "CAM_FRONT",
    "CAM_FRONT_RIGHT",
    "CAM_FRONT_LEFT",
    "CAM_BACK",
    "CAM_BACK_LEFT",
    "CAM_BACK_RIGHT",
]

bins_dynamic_demo = [
    "scenee7ef871f77f44331aefdebc24ec034b7_bin010",
    "scenee7ef871f77f44331aefdebc24ec034b7_bin200",
    "scene30ae9c1092f6404a9e6aa0589e809780_bin100",
    "scene84e056bd8e994362a37cba45c0f75558_bin100",
    "scene717053dec2ef4baa913ba1e24c09edff_bin000",
    "scene82240fd6d5ba4375815f8a7fa1561361_bin050",
    "scene724957e51f464a9aa64a16458443786d_bin000",
    "scened3c39710e9da42f48b605824ce2a1927_bin050",
    "scene034256c9639044f98da7562ef3de3646_bin000",
    "scenee0b14a8e11994763acba690bbcc3f56a_bin080",
    "scene7e2d9f38f8eb409ea57b3864bb4ed098_bin150",
    "scene50ff554b3ecb4d208849d042b7643715_bin000",
]


def _hwc3(img: np.ndarray) -> np.ndarray:
    """确保为三通道 RGB。"""
    if img.ndim == 2:
        img = img[:, :, None]
    if img.shape[2] == 1:
        img = np.repeat(img, 3, axis=2)
    if img.shape[2] == 4:
        img = img[:, :, :3]
    return img


def load_info(info: dict) -> Tuple[str, np.ndarray, np.ndarray]:
    """读取路径与位姿信息，不做 flip_yz。"""
    img_path = info["data_path"]
    c2w = info["sensor2lidar_transform"]

    lidar2cam_r = np.linalg.inv(info["sensor2lidar_rotation"])
    lidar2cam_t = info["sensor2lidar_translation"] @ lidar2cam_r.T
    w2c = np.eye(4)
    w2c[:3, :3] = lidar2cam_r.T
    w2c[3, :3] = -lidar2cam_t

    return img_path, c2w, w2c


def _maybe_resize(img: Image.Image, tgt_reso: Tuple[int, int], ck: np.ndarray) -> Tuple[np.ndarray, np.ndarray, bool]:
    """按目标分辨率缩放，并同步缩放内参。"""
    resize_flag = False
    if img.height != tgt_reso[0] or img.width != tgt_reso[1]:
        fx, fy, cx, cy = ck[0, 0], ck[1, 1], ck[0, 2], ck[1, 2]
        scale_h, scale_w = tgt_reso[0] / img.height, tgt_reso[1] / img.width
        fx_scaled, fy_scaled = fx * scale_w, fy * scale_h
        cx_scaled, cy_scaled = cx * scale_w, cy * scale_h
        ck = np.array([[fx_scaled, 0, cx_scaled], [0, fy_scaled, cy_scaled], [0, 0, 1]], dtype=np.float32)
        img = img.resize((tgt_reso[1], tgt_reso[0]))
        resize_flag = True
    return np.array(img), ck, resize_flag


def load_conditions(
    img_paths: List[str],
    reso: Tuple[int, int],
    is_input: bool = False,
    load_depth_conf: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
    """加载图像与内参，可选加载绝对深度与置信度。"""
    imgs, cks = [], []
    depths_m, confs_m = [], []

    for img_path in img_paths:
        # 相机内参
        param_path = img_path.replace("samples", "samples_param_small")
        param_path = param_path.replace("sweeps", "sweeps_param_small")
        param_path = param_path.replace(".jpg", ".json")
        param = json.load(open(param_path))
        ck = np.array(param["camera_intrinsic"], dtype=np.float32)

        # 图像路径
        img_path = img_path.replace("samples", "samples_small")
        img_path = img_path.replace("sweeps", "sweeps_small")
        img = Image.open(img_path).convert("RGB")
        img, ck, resize_flag = _maybe_resize(img, reso, ck)
        img = _hwc3(img)
        imgs.append(img)
        cks.append(ck)

        if load_depth_conf:
            depthm_path = img_path.replace("sweeps_small", "sweeps_dptm_small")
            depthm_path = depthm_path.replace("samples_small", "samples_dptm_small")
            depthm_path = depthm_path.replace(".jpg", "_dpt.npy")
            conf_path = depthm_path.replace("_dpt.npy", "_conf.npy")
            dptm = np.load(depthm_path).astype(np.float32)
            conf = np.load(conf_path).astype(np.float32)
            if resize_flag:
                dptm = Image.fromarray(dptm)
                dptm = dptm.resize((reso[1], reso[0]), Image.BILINEAR)
                dptm = np.array(dptm)
                conf = Image.fromarray(conf)
                conf = conf.resize((reso[1], reso[0]), Image.BILINEAR)
                conf = np.array(conf)
            depths_m.append(dptm)
            confs_m.append(conf)

    imgs = torch.from_numpy(np.stack(imgs, axis=0)).permute(0, 3, 1, 2).float() / 255.0
    cks = torch.as_tensor(cks, dtype=torch.float32)
    if load_depth_conf:
        depths_m = torch.from_numpy(np.stack(depths_m, axis=0)).float()
        confs_m = torch.from_numpy(np.stack(confs_m, axis=0)).float()
    else:
        depths_m, confs_m = None, None

    return imgs, cks, depths_m, confs_m


class OmniSceneDataset:
    """仅用于预处理与逐场景优化的数据读取器。"""

    def __init__(
        self,
        data_root: str,
        stage: str = "val",
        reso: Tuple[int, int] = (112, 200),
    ) -> None:
        self.data_root = data_root
        self.stage = stage
        self.reso = reso

        if stage == "train":
            self.bin_tokens = json.load(
                open(osp.join(self.data_root, DATA_VERSION, "bins_train_3.2m.json"))
            )["bins"]
        elif stage == "val":
            self.bin_tokens = json.load(
                open(osp.join(self.data_root, DATA_VERSION, "bins_val_3.2m.json"))
            )["bins"]
            self.bin_tokens = self.bin_tokens[:30000:3000][:10]
        elif stage == "test":
            self.bin_tokens = json.load(
                open(osp.join(self.data_root, DATA_VERSION, "bins_val_3.2m.json"))
            )["bins"]
            self.bin_tokens = self.bin_tokens[0::14][:2048]
        elif stage == "demo":
            self.bin_tokens = bins_dynamic_demo
        else:
            raise ValueError(f"未知阶段: {stage}")

    def __len__(self) -> int:
        return len(self.bin_tokens)

    def __getitem__(self, index: int) -> dict:
        bin_token = self.bin_tokens[index]
        with open(osp.join(self.data_root, DATA_VERSION, "bin_infos_3.2m", bin_token + ".pkl"), "rb") as f:
            bin_info = pkl.load(f)

        sensor_info_center = {
            sensor: bin_info["sensor_info"][sensor][0]
            for sensor in CAMERA_TYPES + ["LIDAR_TOP"]
        }

        # 输入视角（6 张）
        input_img_paths, input_c2ws = [], []
        for cam in CAMERA_TYPES:
            info = copy.deepcopy(sensor_info_center[cam])
            img_path, c2w, _ = load_info(info)
            img_path = img_path.replace(DATASET_PREFIX, self.data_root)
            input_img_paths.append(img_path)
            input_c2ws.append(c2w)
        input_c2ws = torch.as_tensor(input_c2ws, dtype=torch.float32)

        input_imgs, input_cks, input_depths_m, input_confs_m = load_conditions(
            input_img_paths,
            self.reso,
            is_input=True,
            load_depth_conf=True,
        )

        # 输出视角（12 张）
        output_img_paths, output_c2ws = [], []
        frame_num = len(bin_info["sensor_info"]["LIDAR_TOP"])
        if frame_num < 3:
            raise ValueError(f"bin {bin_token} 仅包含 {frame_num} 帧，无法构造输出视角")
        rend_indices = [[1, 2]] * 6
        for cam_id, cam in enumerate(CAMERA_TYPES):
            indices = rend_indices[cam_id]
            for ind in indices:
                info = copy.deepcopy(bin_info["sensor_info"][cam][ind])
                img_path, c2w, _ = load_info(info)
                img_path = img_path.replace(DATASET_PREFIX, self.data_root)
                output_img_paths.append(img_path)
                output_c2ws.append(c2w)
        output_c2ws = torch.as_tensor(output_c2ws, dtype=torch.float32)

        output_imgs, output_cks, _, _ = load_conditions(
            output_img_paths,
            self.reso,
            is_input=False,
            load_depth_conf=False,
        )

        # 输出拼接输入（总 18 张）
        output_imgs = torch.cat([output_imgs, input_imgs], dim=0)
        output_c2ws = torch.cat([output_c2ws, input_c2ws], dim=0)
        output_cks = torch.cat([output_cks, input_cks], dim=0)

        return {
            "scene": bin_token,
            "context": {
                "image": input_imgs,
                "intrinsics": input_cks,
                "extrinsics": input_c2ws,
                "depths": input_depths_m,
                "confs": input_confs_m,
            },
            "target": {
                "image": output_imgs,
                "intrinsics": output_cks,
                "extrinsics": output_c2ws,
            },
        }
