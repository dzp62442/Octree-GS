import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from comp_svfgs.dataset_omniscene import OmniSceneDataset
from comp_svfgs.omniscene_preprocess import PREPARED_FORMAT_VERSION
import run_omniscene


class OmniSceneCenter150Test(unittest.TestCase):
    @staticmethod
    def _tokens():
        return ["scene{:032x}_bin000".format(index) for index in range(150)]

    @classmethod
    def _write_split(cls, root: Path, tokens=None):
        version_root = root / "interp_12Hz_trainval"
        info_root = version_root / "bin_infos_3.2m"
        info_root.mkdir(parents=True)
        tokens = tokens or cls._tokens()
        (version_root / "bins_center150_v1.json").write_text(
            json.dumps({"bins": tokens}), encoding="utf-8"
        )
        for token in tokens:
            (info_root / (token + ".pkl")).write_bytes(b"pkl")
        return tokens

    @staticmethod
    def _write_prepared_scene(scene_dir: Path, token: str):
        images_dir = scene_dir / "images"
        images_dir.mkdir(parents=True)
        train_frames = []
        test_frames = []
        for index in range(6):
            name = "train_{:03d}.jpg".format(index)
            (images_dir / name).write_bytes(b"image")
            train_frames.append({"file_path": "images/" + name})
        for index in range(18):
            name = "test_{:03d}.jpg".format(index)
            (images_dir / name).write_bytes(b"image")
            test_frames.append({"file_path": "images/" + name})
        (scene_dir / "transforms_train.json").write_text(
            json.dumps({"frames": train_frames}), encoding="utf-8"
        )
        (scene_dir / "transforms_test.json").write_text(
            json.dumps({"frames": test_frames}), encoding="utf-8"
        )
        (scene_dir / "points3D.ply").write_bytes(b"ply")
        (scene_dir / "meta.json").write_text(
            json.dumps({
                "format_version": PREPARED_FORMAT_VERSION,
                "scene": token,
                "resolution": [112, 200],
                "conf_threshold": 0.3,
            }),
            encoding="utf-8",
        )

    @staticmethod
    def _write_iteration(model_dir: Path, iteration: int, offset: float = 0.0):
        (model_dir / "metrics_{}.txt".format(iteration)).write_text(
            "PSNR : {}\nSSIM : {}\nLPIPS : {}\n".format(
                20.0 + offset + iteration / 1000.0,
                0.7 + offset / 100.0,
                0.2 - offset / 100.0,
            ),
            encoding="utf-8",
        )
        (model_dir / "training_time_{}.txt".format(iteration)).write_text(
            "TRAINING_TIME_SECONDS : {}\n".format(iteration / 100.0 + offset),
            encoding="utf-8",
        )
        method_dir = model_dir / "test" / "ours_{}".format(iteration)
        render_dir = method_dir / "renders"
        gt_dir = method_dir / "gt"
        render_dir.mkdir(parents=True)
        gt_dir.mkdir(parents=True)
        for index in range(18):
            name = "test_{:03d}.png".format(index)
            (render_dir / name).write_bytes(b"render")
            (gt_dir / name).write_bytes(b"gt")

    def test_center150_loader_preserves_validated_order(self):
        with tempfile.TemporaryDirectory() as temporary_dir:
            root = Path(temporary_dir)
            tokens = self._write_split(root)
            dataset = OmniSceneDataset(str(root), stage="center150")
            self.assertEqual(dataset.bin_tokens, tokens)

    def test_center150_loader_rejects_duplicate_scene(self):
        with tempfile.TemporaryDirectory() as temporary_dir:
            root = Path(temporary_dir)
            tokens = self._tokens()
            tokens[-1] = "scene{:032x}_bin001".format(0)
            self._write_split(root, tokens)
            with self.assertRaisesRegex(ValueError, "不同场景"):
                OmniSceneDataset(str(root), stage="center150")

    def test_train_command_contains_center150_milestones(self):
        command = run_omniscene._build_train_command(
            "/scene", "/model", "2", 10000, (1000, 5000, 10000),
            ["--base_layer", "4"], full_eval_metrics=True
        )
        self.assertIn("--full_eval_metrics", command)
        self.assertNotIn("--checkpoint_iterations", command)
        self.assertNotIn("--start_checkpoint", command)
        self.assertEqual(command[command.index("--appearance_dim") + 1], "0")
        test_index = command.index("--test_iterations") + 1
        self.assertEqual(command[test_index:test_index + 3], ["1000", "5000", "10000"])
        save_index = command.index("--save_iterations") + 1
        self.assertEqual(command[save_index], "10000")
        self.assertEqual(command[-2:], ["--base_layer", "4"])

    def test_run_command_exports_requested_gpu_before_import(self):
        with mock.patch.object(run_omniscene.subprocess, "run") as subprocess_run:
            run_omniscene._run_command(["python", "train.py", "--gpu", "2"])
        environment = subprocess_run.call_args[1]["env"]
        self.assertEqual(environment["CUDA_VISIBLE_DEVICES"], "2")

    def test_incomplete_scene_is_cleared_and_restarted_without_checkpoint(self):
        with tempfile.TemporaryDirectory() as temporary_dir:
            model_dir = Path(temporary_dir) / "model"
            model_dir.mkdir()
            stale_artifact = model_dir / "metrics_5000.txt"
            stale_artifact.write_text("stale", encoding="utf-8")
            with mock.patch.object(
                run_omniscene, "scene_complete", side_effect=[False, True]
            ), mock.patch.object(
                run_omniscene, "_run_command"
            ) as run_command, mock.patch.object(
                run_omniscene, "_write_scene_completion"
            ):
                run_omniscene._run_center150_scene(
                    "001_scene", "scene_bin", "/prepared", str(model_dir),
                    (112, 200), 0.3, "0", 10000, (1000, 5000, 10000), []
                )
            self.assertFalse(stale_artifact.exists())
            command = run_command.call_args.args[0]
            self.assertNotIn("--checkpoint_iterations", command)
            self.assertNotIn("--start_checkpoint", command)

    def test_completion_and_summary_require_all_artifacts(self):
        with tempfile.TemporaryDirectory() as temporary_dir:
            root = Path(temporary_dir)
            records = []
            for scene_index in range(2):
                token = "scene{:032x}_bin000".format(scene_index)
                scene_name = "{:03d}_{}".format(scene_index + 1, token)
                scene_dir = root / "prepared" / scene_name
                model_dir = root / "results" / scene_name
                self._write_prepared_scene(scene_dir, token)
                model_dir.mkdir(parents=True)
                for iteration in run_omniscene.DEFAULT_EVAL_ITERATIONS:
                    self._write_iteration(model_dir, iteration, float(scene_index * 2))
                final_point_cloud = model_dir / "point_cloud" / "iteration_10000"
                final_point_cloud.mkdir(parents=True)
                (final_point_cloud / "point_cloud.ply").write_bytes(b"ply")
                self.assertTrue(
                    run_omniscene.scene_complete(
                        str(model_dir), str(scene_dir), token, (112, 200), 0.3,
                        10000, run_omniscene.DEFAULT_EVAL_ITERATIONS
                    )
                )
                records.append((scene_name, token, str(scene_dir), str(model_dir)))

            experiment_root = root / "results"
            with mock.patch.object(run_omniscene, "CENTER150_SAMPLE_COUNT", 2):
                summary = run_omniscene._aggregate_center150_results(
                    str(experiment_root), records, (112, 200), 0.3, 10000,
                    run_omniscene.DEFAULT_EVAL_ITERATIONS
                )
            self.assertAlmostEqual(summary["averages"]["1000"]["psnr"], 22.0)
            self.assertAlmostEqual(
                summary["averages"]["10000"]["training_time_seconds"], 101.0
            )

            (Path(records[0][3]) / "metrics_5000.txt").unlink()
            self.assertFalse(
                run_omniscene.scene_complete(
                    records[0][3], records[0][2], records[0][1], (112, 200), 0.3,
                    10000, run_omniscene.DEFAULT_EVAL_ITERATIONS
                )
            )


if __name__ == "__main__":
    unittest.main()
