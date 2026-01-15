import argparse
import os
import subprocess
import sys
import time

from comp_svfgs.dataset_omniscene import OmniSceneDataset
from comp_svfgs.omniscene_preprocess import preprocess_scene

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def _parse_reso(value: str):
    if "x" in value:
        h, w = value.lower().split("x")
    else:
        parts = value.split(",")
        if len(parts) != 2:
            raise ValueError("分辨率格式应为 HxW 或 H,W")
        h, w = parts
    return int(h), int(w)


def _run_train(scene_dir: str, output_root: str, gpu: str, extra_args: list):
    scene_name = os.path.basename(scene_dir)
    timestamp = time.strftime("%Y-%m-%d_%H-%M-%S")
    model_path = os.path.join(output_root, scene_name, "omniscene", timestamp)
    os.makedirs(model_path, exist_ok=True)

    cmd = [
        sys.executable,
        os.path.join(BASE_DIR, "train.py"),
        "--eval",
        "-s",
        scene_dir,
        "-m",
        model_path,
        "-r",
        "1",
    ]
    if gpu is not None:
        cmd += ["--gpu", gpu]
    cmd += extra_args

    print(f"[INFO] 开始训练: {scene_name}")
    result = subprocess.run(cmd, check=False, cwd=BASE_DIR)
    if result.returncode != 0:
        raise RuntimeError(f"训练失败: {scene_name}")


def main():
    parser = argparse.ArgumentParser(description="OmniScene 单阶段预处理+训练+评估")
    parser.add_argument("--data_root", default="data/omniscene", help="OmniScene 数据根目录")
    parser.add_argument("--output_root", default="output/omniscene", help="预处理输出根目录")
    parser.add_argument("--result_root", default="output/omniscene_results", help="训练输出根目录")
    parser.add_argument("--stage", default="val", choices=["train", "val", "test", "demo"], help="数据划分")
    parser.add_argument("--reso", default="112x200", help="图像分辨率，格式 HxW 或 H,W")
    parser.add_argument("--conf_threshold", type=float, default=0.3, help="点云置信度阈值")
    parser.add_argument("--gpu", default=None, help="指定 GPU id，默认使用最空闲 GPU")
    parser.add_argument("--extra_train_args", nargs=argparse.REMAINDER, default=[], help="透传给 train.py 的额外参数")
    parser.add_argument("--iterations", type=int, default=10000, help="训练迭代次数，默认 10000")

    args = parser.parse_args()
    reso = _parse_reso(args.reso)
    output_root = args.output_root
    result_root = args.result_root
    if not os.path.isabs(output_root):
        output_root = os.path.join(BASE_DIR, output_root)
    if not os.path.isabs(result_root):
        result_root = os.path.join(BASE_DIR, result_root)

    dataset = OmniSceneDataset(data_root=args.data_root, stage=args.stage, reso=reso)

    iterations_arg = ["--iterations", str(args.iterations)]
    for idx in range(len(dataset)):
        scene_data = dataset[idx]
        scene_dir = preprocess_scene(
            scene_data=scene_data,
            output_root=output_root,
            conf_threshold=args.conf_threshold,
        )
        _run_train(
            scene_dir=scene_dir,
            output_root=result_root,
            gpu=args.gpu,
            extra_args=iterations_arg + args.extra_train_args,
        )


if __name__ == "__main__":
    main()
