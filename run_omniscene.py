import argparse
import json
import math
import os
import shlex
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from typing import Dict, List, Optional, Sequence, Tuple

from comp_svfgs.dataset_omniscene import CENTER150_SAMPLE_COUNT, OmniSceneDataset
from comp_svfgs.omniscene_preprocess import preprocess_scene, prepared_scene_complete


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_ITERATIONS = 10000
DEFAULT_EVAL_ITERATIONS = (1000, 5000, 10000)
METRIC_NAMES = ("PSNR", "SSIM", "LPIPS")
TRAINING_TIME_KEY = "training_time_seconds"
PROTOCOL_VERSION = 1
RESERVED_EXTRA_ARGS = {
    "-s", "--source_path", "-m", "--model_path", "--gpu", "--iterations",
    "--test_iterations", "--save_iterations", "--checkpoint_iterations",
    "--start_checkpoint", "--full_eval_metrics",
}


def _parse_reso(value: str) -> Tuple[int, int]:
    try:
        if "x" in value.lower():
            h, w = value.lower().split("x")
        else:
            h, w = value.split(",")
        resolution = int(h), int(w)
    except (TypeError, ValueError) as exc:
        raise argparse.ArgumentTypeError("分辨率格式应为 HxW 或 H,W") from exc
    if resolution[0] <= 0 or resolution[1] <= 0:
        raise argparse.ArgumentTypeError("分辨率必须为正整数")
    return resolution


def _absolute_path(path: str) -> str:
    if os.path.isabs(path):
        return path
    return os.path.join(BASE_DIR, path)


def _atomic_write_json(path: str, payload: Dict) -> None:
    temporary_path = path + ".tmp"
    with open(temporary_path, "w", encoding="utf-8") as output_file:
        json.dump(payload, output_file, indent=2, ensure_ascii=False)
        output_file.write("\n")
    os.replace(temporary_path, path)


def _atomic_write_text(path: str, content: str) -> None:
    temporary_path = path + ".tmp"
    with open(temporary_path, "w", encoding="utf-8") as output_file:
        output_file.write(content)
    os.replace(temporary_path, path)


def _parse_metrics(path: str) -> Dict[str, float]:
    values = {}
    with open(path, "r", encoding="utf-8") as metrics_file:
        for line in metrics_file:
            name, separator, value = line.partition(":")
            name = name.strip()
            if separator and name in METRIC_NAMES:
                values[name] = float(value.strip())
    if set(values) != set(METRIC_NAMES):
        raise ValueError("指标文件不完整：{}".format(path))
    if not all(math.isfinite(value) for value in values.values()):
        raise ValueError("指标文件包含非有限值：{}".format(path))
    return values


def _parse_training_time(path: str) -> float:
    with open(path, "r", encoding="utf-8") as timing_file:
        line = timing_file.read().strip()
    name, separator, value = line.partition(":")
    if not separator or name.strip() != "TRAINING_TIME_SECONDS":
        raise ValueError("训练耗时文件格式错误：{}".format(path))
    training_time = float(value.strip())
    if not math.isfinite(training_time) or training_time < 0.0:
        raise ValueError("训练耗时无效：{}".format(path))
    return training_time


def _expected_test_image_names(scene_dir: str) -> set:
    transforms_path = os.path.join(scene_dir, "transforms_test.json")
    with open(transforms_path, "r", encoding="utf-8") as transforms_file:
        frames = json.load(transforms_file).get("frames", [])
    names = {
        os.path.splitext(os.path.basename(frame["file_path"]))[0] + ".png"
        for frame in frames
    }
    if len(frames) != 18 or len(names) != 18:
        raise ValueError("测试视角必须恰好为 18 个：{}".format(transforms_path))
    return names


def _artifact_exists(path: str) -> bool:
    return os.path.isfile(path) and os.path.getsize(path) > 0


def iteration_complete(model_path: str, scene_dir: str, iteration: int) -> bool:
    try:
        _parse_metrics(os.path.join(model_path, "metrics_{}.txt".format(iteration)))
        _parse_training_time(os.path.join(model_path, "training_time_{}.txt".format(iteration)))
        expected_names = _expected_test_image_names(scene_dir)
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError):
        return False

    iteration_dir = os.path.join(model_path, "test", "ours_{}".format(iteration))
    render_dir = os.path.join(iteration_dir, "renders")
    gt_dir = os.path.join(iteration_dir, "gt")
    if not os.path.isdir(render_dir) or not os.path.isdir(gt_dir):
        return False
    render_paths = [os.path.join(render_dir, name) for name in os.listdir(render_dir) if name.endswith(".png")]
    gt_paths = [os.path.join(gt_dir, name) for name in os.listdir(gt_dir) if name.endswith(".png")]
    if {os.path.basename(path) for path in render_paths} != expected_names:
        return False
    if {os.path.basename(path) for path in gt_paths} != expected_names:
        return False
    if not all(_artifact_exists(path) for path in render_paths + gt_paths):
        return False

    return True


def scene_complete(
    model_path: str,
    scene_dir: str,
    bin_token: str,
    resolution: Tuple[int, int],
    conf_threshold: float,
    iterations: int,
    eval_iterations: Sequence[int],
) -> bool:
    if not prepared_scene_complete(scene_dir, bin_token, resolution, conf_threshold):
        return False
    if not all(iteration_complete(model_path, scene_dir, iteration) for iteration in eval_iterations):
        return False
    final_point_cloud = os.path.join(
        model_path, "point_cloud", "iteration_{}".format(iterations), "point_cloud.ply"
    )
    if not _artifact_exists(final_point_cloud):
        return False
    try:
        training_times = [
            _parse_training_time(os.path.join(model_path, "training_time_{}.txt".format(iteration)))
            for iteration in eval_iterations
        ]
    except (OSError, ValueError):
        return False
    return training_times == sorted(training_times)


def _validate_extra_train_args(parser: argparse.ArgumentParser, extra_args: Sequence[str]) -> None:
    for value in extra_args:
        option = value.split("=", 1)[0]
        if option in RESERVED_EXTRA_ARGS:
            parser.error(
                "{} 由 run_omniscene.py 的顶层参数管理，不能放入 --extra_train_args。".format(option)
            )


def _build_train_command(
    scene_dir: str,
    model_path: str,
    gpu: Optional[str],
    iterations: int,
    eval_iterations: Sequence[int],
    extra_args: Sequence[str],
    full_eval_metrics: bool,
) -> List[str]:
    command = [
        sys.executable, os.path.join(BASE_DIR, "train.py"), "--eval", "-s", scene_dir,
        "-m", model_path, "-r", "1", "--appearance_dim", "0", "--iterations", str(iterations),
        "--test_iterations",
    ]
    command.extend(str(iteration) for iteration in eval_iterations)
    command.append("--save_iterations")
    command.append(str(iterations))
    if full_eval_metrics:
        command.append("--full_eval_metrics")
    if gpu is not None:
        command.extend(["--gpu", gpu])
    command.extend(extra_args)
    return command


def _run_command(command: Sequence[str]) -> None:
    print("[RUN] {}".format(" ".join(shlex.quote(value) for value in command)), flush=True)
    environment = os.environ.copy()
    if "--gpu" in command:
        environment["CUDA_VISIBLE_DEVICES"] = command[command.index("--gpu") + 1]
    subprocess.run(command, check=True, cwd=BASE_DIR, env=environment)


def _write_scene_completion(
    model_path: str,
    scene_name: str,
    bin_token: str,
    resolution: Tuple[int, int],
    iterations: int,
    eval_iterations: Sequence[int],
) -> None:
    metrics = {}
    for iteration in eval_iterations:
        values = _parse_metrics(os.path.join(model_path, "metrics_{}.txt".format(iteration)))
        metrics[str(iteration)] = {
            "psnr": values["PSNR"],
            "ssim": values["SSIM"],
            "lpips": values["LPIPS"],
            TRAINING_TIME_KEY: _parse_training_time(
                os.path.join(model_path, "training_time_{}.txt".format(iteration))
            ),
        }
    _atomic_write_json(
        os.path.join(model_path, "center150_complete.json"),
        {
            "protocol_version": PROTOCOL_VERSION,
            "split": "center150",
            "scene_name": scene_name,
            "bin_token": bin_token,
            "resolution": list(resolution),
            "iterations": iterations,
            "eval_iterations": list(eval_iterations),
            "metrics": metrics,
            "completed_at": datetime.now(timezone.utc).isoformat(),
        },
    )


def _run_center150_scene(
    scene_name: str,
    bin_token: str,
    scene_dir: str,
    model_path: str,
    resolution: Tuple[int, int],
    conf_threshold: float,
    gpu: Optional[str],
    iterations: int,
    eval_iterations: Sequence[int],
    extra_args: Sequence[str],
) -> None:
    if scene_complete(
        model_path, scene_dir, bin_token, resolution, conf_threshold, iterations, eval_iterations
    ):
        _write_scene_completion(
            model_path, scene_name, bin_token, resolution, iterations, eval_iterations
        )
        print("[SKIP] 已完成：{}".format(scene_name), flush=True)
        return

    if os.path.isdir(model_path):
        print("[RESTART] 样本不完整，从头训练：{}".format(scene_name), flush=True)
        shutil.rmtree(model_path)
    else:
        print("[START] 从头训练：{}".format(scene_name), flush=True)

    os.makedirs(model_path, exist_ok=True)
    command = _build_train_command(
        scene_dir, model_path, gpu, iterations, eval_iterations, extra_args,
        full_eval_metrics=True
    )
    _run_command(command)

    if not scene_complete(
        model_path, scene_dir, bin_token, resolution, conf_threshold, iterations, eval_iterations
    ):
        raise RuntimeError("Center150 样本产物不完整：{}".format(scene_name))
    _write_scene_completion(
        model_path, scene_name, bin_token, resolution, iterations, eval_iterations
    )
    print("[DONE] {}".format(scene_name), flush=True)


def _run_standard_scene(
    scene_dir: str,
    experiment_root: str,
    gpu: Optional[str],
    iterations: int,
    extra_args: Sequence[str],
) -> None:
    scene_name = os.path.basename(scene_dir)
    timestamp = time.strftime("%Y-%m-%d_%H-%M-%S")
    model_path = os.path.join(experiment_root, scene_name, "omniscene", timestamp)
    os.makedirs(model_path, exist_ok=True)
    command = _build_train_command(
        scene_dir, model_path, gpu, iterations, [iterations], extra_args,
        full_eval_metrics=False
    )
    _run_command(command)


def _aggregate_center150_results(
    experiment_root: str,
    scene_records: Sequence[Tuple[str, str, str, str]],
    resolution: Tuple[int, int],
    conf_threshold: float,
    iterations: int,
    eval_iterations: Sequence[int],
) -> Dict:
    if len(scene_records) != CENTER150_SAMPLE_COUNT:
        raise RuntimeError(
            "Center150 汇总需要 {} 个样本，实际为 {} 个。".format(
                CENTER150_SAMPLE_COUNT, len(scene_records)
            )
        )

    accumulators = {
        iteration: {name: [] for name in METRIC_NAMES + (TRAINING_TIME_KEY,)}
        for iteration in eval_iterations
    }
    samples = []
    for scene_name, bin_token, scene_dir, model_path in scene_records:
        if not scene_complete(
            model_path, scene_dir, bin_token, resolution, conf_threshold, iterations, eval_iterations
        ):
            raise RuntimeError("无法汇总未完成样本：{}".format(scene_name))
        sample_metrics = {}
        for iteration in eval_iterations:
            metrics = _parse_metrics(os.path.join(model_path, "metrics_{}.txt".format(iteration)))
            training_time = _parse_training_time(
                os.path.join(model_path, "training_time_{}.txt".format(iteration))
            )
            sample_metrics[str(iteration)] = {
                "psnr": metrics["PSNR"], "ssim": metrics["SSIM"],
                "lpips": metrics["LPIPS"], TRAINING_TIME_KEY: training_time,
            }
            for name in METRIC_NAMES:
                accumulators[iteration][name].append(metrics[name])
            accumulators[iteration][TRAINING_TIME_KEY].append(training_time)
        samples.append({"scene_name": scene_name, "bin_token": bin_token, "metrics": sample_metrics})

    averages = {}
    for iteration in eval_iterations:
        averages[str(iteration)] = {
            "num_samples": CENTER150_SAMPLE_COUNT,
            "psnr": sum(accumulators[iteration]["PSNR"]) / CENTER150_SAMPLE_COUNT,
            "ssim": sum(accumulators[iteration]["SSIM"]) / CENTER150_SAMPLE_COUNT,
            "lpips": sum(accumulators[iteration]["LPIPS"]) / CENTER150_SAMPLE_COUNT,
            TRAINING_TIME_KEY: (
                sum(accumulators[iteration][TRAINING_TIME_KEY]) / CENTER150_SAMPLE_COUNT
            ),
        }

    summary = {
        "protocol_version": PROTOCOL_VERSION,
        "split": "center150",
        "sample_count": CENTER150_SAMPLE_COUNT,
        "resolution": list(resolution),
        "conf_threshold": conf_threshold,
        "iterations": iterations,
        "eval_iterations": list(eval_iterations),
        "averages": averages,
        "samples": samples,
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }
    _atomic_write_json(os.path.join(experiment_root, "center150_metrics_summary.json"), summary)

    lines = ["Center150 samples: {}".format(CENTER150_SAMPLE_COUNT)]
    for iteration in eval_iterations:
        result = averages[str(iteration)]
        lines.extend([
            "Iteration {}".format(iteration),
            "PSNR : {:.7f}".format(result["psnr"]),
            "SSIM : {:.7f}".format(result["ssim"]),
            "LPIPS : {:.7f}".format(result["lpips"]),
            "TRAINING_TIME_SECONDS : {:.7f}".format(result[TRAINING_TIME_KEY]),
        ])
    _atomic_write_text(
        os.path.join(experiment_root, "center150_metrics_summary.txt"), "\n".join(lines) + "\n"
    )

    print("[SUMMARY] Center150 共 {} 个样本".format(CENTER150_SAMPLE_COUNT), flush=True)
    for iteration in eval_iterations:
        result = averages[str(iteration)]
        print(
            "  {}: PSNR={:.7f}, SSIM={:.7f}, LPIPS={:.7f}, TRAIN_TIME={:.7f}s".format(
                iteration, result["psnr"], result["ssim"], result["lpips"],
                result[TRAINING_TIME_KEY]
            ),
            flush=True,
        )
    return summary


def _ensure_protocol(path: str, protocol: Dict) -> None:
    if os.path.isfile(path):
        with open(path, "r", encoding="utf-8") as protocol_file:
            existing = json.load(protocol_file)
        if existing != protocol:
            raise RuntimeError(
                "实验目录已有不同协议：{}。请恢复原参数，或通过 --result_root 使用新目录。".format(path)
            )
        return
    _atomic_write_json(path, protocol)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="OmniScene 逐场景预处理、优化、阶段评估与汇总",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--data_root", default="data/omniscene", help="OmniScene 数据根目录")
    parser.add_argument("--output_root", default="output/omniscene", help="预处理缓存根目录")
    parser.add_argument("--result_root", default="output/omniscene_results", help="实验结果根目录")
    parser.add_argument(
        "--stage", default="center150",
        choices=["train", "val", "center150", "test", "demo"], help="数据划分"
    )
    parser.add_argument("--reso", type=_parse_reso, default=(112, 200), help="图像分辨率 HxW 或 H,W")
    parser.add_argument("--conf_threshold", type=float, default=0.3, help="初始化点云置信度阈值")
    parser.add_argument("--gpu", default="0", help="指定 GPU id；本机默认使用 GPU 0")
    parser.add_argument("--iterations", type=int, default=DEFAULT_ITERATIONS, help="总优化迭代数")
    parser.add_argument(
        "--eval_iterations", nargs="+", type=int, default=list(DEFAULT_EVAL_ITERATIONS),
        help="Center150 的评估迭代点"
    )
    parser.add_argument(
        "--extra_train_args", nargs=argparse.REMAINDER, default=[],
        help="透传给 train.py 的额外模型或优化参数"
    )
    args = parser.parse_args()

    if args.iterations <= 0:
        parser.error("--iterations 必须为正整数")
    if not 0.0 <= args.conf_threshold <= 1.0:
        parser.error("--conf_threshold 必须位于 [0, 1]")
    eval_iterations = tuple(args.eval_iterations)
    if args.stage == "center150":
        if not eval_iterations or any(iteration <= 0 for iteration in eval_iterations):
            parser.error("--eval_iterations 必须包含正整数")
        if tuple(sorted(set(eval_iterations))) != eval_iterations:
            parser.error("--eval_iterations 必须严格递增且不能重复")
        if eval_iterations[-1] > args.iterations:
            parser.error("--eval_iterations 不能超过 --iterations")
    else:
        eval_iterations = (args.iterations,)
    _validate_extra_train_args(parser, args.extra_train_args)

    data_root = _absolute_path(args.data_root)
    dataset = OmniSceneDataset(data_root=data_root, stage=args.stage, reso=args.reso)
    resolution_tag = "{}x{}".format(args.reso[0], args.reso[1])
    prepared_root = os.path.join(_absolute_path(args.output_root), "{}_{}".format(args.stage, resolution_tag))
    experiment_name = "{}_{}".format(args.stage, resolution_tag)
    if args.stage == "center150" and (
        args.iterations != DEFAULT_ITERATIONS or eval_iterations != DEFAULT_EVAL_ITERATIONS
    ):
        experiment_name += "_iter{}_eval{}".format(
            args.iterations, "-".join(str(iteration) for iteration in eval_iterations)
        )
    experiment_root = os.path.join(_absolute_path(args.result_root), experiment_name)
    os.makedirs(prepared_root, exist_ok=True)
    os.makedirs(experiment_root, exist_ok=True)

    if args.stage == "center150":
        protocol = {
            "protocol_version": PROTOCOL_VERSION,
            "split": args.stage,
            "data_root": os.path.realpath(data_root),
            "resolution": list(args.reso),
            "conf_threshold": args.conf_threshold,
            "iterations": args.iterations,
            "eval_iterations": list(eval_iterations),
            "default_appearance_dim": 0,
            "extra_train_args": list(args.extra_train_args),
        }
        _ensure_protocol(os.path.join(experiment_root, "center150_protocol.json"), protocol)

    prefix_width = 3 if args.stage == "center150" else max(2, len(str(len(dataset))))
    scene_records = []

    for index, bin_token in enumerate(dataset.bin_tokens):
        prefix = str(index + 1).zfill(prefix_width)
        scene_name = "{}_{}".format(prefix, bin_token)
        scene_dir = os.path.join(prepared_root, scene_name)
        model_path = os.path.join(experiment_root, scene_name)
        scene_records.append((scene_name, bin_token, scene_dir, model_path))

        if args.stage == "center150" and scene_complete(
            model_path, scene_dir, bin_token, args.reso, args.conf_threshold,
            args.iterations, eval_iterations
        ):
            _write_scene_completion(
                model_path, scene_name, bin_token, args.reso, args.iterations, eval_iterations
            )
            print("[SKIP] 已完成：{}".format(scene_name), flush=True)
            continue

        scene_data = dataset[index]
        scene_dir = preprocess_scene(
            scene_data=scene_data,
            output_root=prepared_root,
            conf_threshold=args.conf_threshold,
            prefix=prefix,
        )
        if args.stage == "center150":
            _run_center150_scene(
                scene_name, bin_token, scene_dir, model_path, args.reso,
                args.conf_threshold, args.gpu, args.iterations, eval_iterations,
                args.extra_train_args
            )
        else:
            _run_standard_scene(
                scene_dir, experiment_root, args.gpu, args.iterations, args.extra_train_args
            )

    if args.stage == "center150":
        _aggregate_center150_results(
            experiment_root, scene_records, args.reso, args.conf_threshold,
            args.iterations, eval_iterations
        )


if __name__ == "__main__":
    main()
