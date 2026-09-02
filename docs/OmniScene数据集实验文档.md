# OmniScene 数据集实验文档（Octree-GS / comp_svfgs）

## 1. 实验边界

本分支把原本面向 Mip-NeRF 360 等数据集的逐场景优化流程接入 OmniScene，用于和 SVF-GS 等前馈方法进行同数据、同视角、同优化预算对比。

Center150 的样本划分由 `~/Projects/SVF-GS` 唯一负责生成。本项目：

- 只读取 `interp_12Hz_trainval/bins_center150_v1.json`；
- 不包含 Center150 生成、重排或修改逻辑；
- 加载前检查清单包含 150 个唯一 bin、覆盖 150 个不同场景、token 格式合法，并确认对应 `bin_infos_3.2m/*.pkl` 存在；
- 保持清单中的原始顺序，场景名前缀固定为 `001` 至 `150`。

## 2. 默认 Center150 协议

项目根目录直接执行：

```bash
python run_omniscene.py
```

默认配置为：

- 数据划分：`center150`；
- 分辨率：`112x200`；
- 每个 bin 使用 6 张输入图像优化高斯；
- 在 18 张目标图像上评估，其中前 12 张为新视角，最后 6 张为输入视角；
- 总优化次数：10000；
- 评估里程碑：1000、5000、10000；
- GPU：0；
- appearance embedding：关闭（`--appearance_dim 0`）；
- 初始化点云：由 6 个输入视角的 Metric3D 绝对深度和置信度构造，阈值为 0.3。

本机只有一张卡，因此默认命令已经使用 GPU 0；如需显式指定，仍可执行：

```bash
python run_omniscene.py --gpu 0
```

## 3. 参数覆写

顶层协议参数直接在启动器中覆写：

```bash
python run_omniscene.py \
  --gpu 0 \
  --reso 224x400 \
  --iterations 12000 \
  --eval_iterations 1000 5000 10000 12000
```

Octree-GS 自身的模型或优化参数放在 `--extra_train_args` 之后：

```bash
python run_omniscene.py --gpu 0 \
  --extra_train_args --base_layer 4 --visible_threshold 0.01
```

数据路径、模型路径、总迭代数和评估/保存里程碑由启动器统一管理，不能放入 `--extra_train_args`，避免产物协议与实际命令不一致。Center150 启动器明确不启用 checkpoint，`--checkpoint_iterations` 与 `--start_checkpoint` 同样不能透传。使用不同额外参数进行独立实验时，应通过 `--result_root` 指定不同结果根目录。

如需复现旧的 10-bin val 实验，可执行：

```bash
python run_omniscene.py --stage val --gpu 0
```

## 4. 数据与坐标约定

预处理输出采用 Octree-GS 已支持的 NeRF/Blender 风格目录：

```text
output/omniscene/center150_112x200/
└── 001_<bin_token>/
    ├── images/
    ├── transforms_train.json
    ├── transforms_test.json
    ├── points3D.ply
    └── meta.json
```

每个 frame 分别保存自己的 `fl_x/fl_y/cx/cy`，因此本项目对 OmniScene 使用逐图像内参，而不是把一个全局内参强行共享给所有视角。

OmniScene 的 `sensor2lidar_transform` 在本适配中作为 OpenCV 相机坐标系下的 c2w 使用。`transforms_*.json` 写入 `no_flip_yz=true`，Octree-GS reader 不再执行 Blender/OpenGL 的 Y/Z 轴翻转；初始化点云也用同一 c2w 直接把 OpenCV 相机点变换到世界坐标系。该链路与 SVF-GS/DropGaussian 对照校验后保持不变。

预处理缓存会检查版本、bin token、分辨率、置信度阈值、6/18 视角数量、图像、transform 和 PLY；不匹配时重新预处理。

## 5. 里程碑评估与计时

在每个指定里程碑，`train.py` 会：

1. 用当前高斯在全部 18 个 test 视角渲染；
2. 计算并落盘 PSNR、SSIM、LPIPS；
3. 记录从第 1 次迭代累计到当前里程碑的训练耗时；
4. 仅在最终迭代（默认 10k）保存 PLY。

训练耗时通过 CUDA 同步后的 wall time 统计，并排除里程碑评估、渲染图像写盘和最终 PLY 保存耗时。每个样本在单次进程中从 0 训练到 10k，因此 1k/5k/10k 均为从第 1 次迭代开始的累计训练时间。Center150 不保存任何样本内 checkpoint。

单个样本的核心产物为：

```text
<model_path>/
├── metrics_1000.txt
├── metrics_5000.txt
├── metrics_10000.txt
├── training_time_1000.txt
├── training_time_5000.txt
├── training_time_10000.txt
├── point_cloud/iteration_10000/point_cloud.ply
├── test/ours_{1000,5000,10000}/{renders,gt}/
└── center150_complete.json
```

## 6. 断点续跑与快速跳过

Center150 使用固定模型目录，不使用时间戳。命令可以安全重复执行：

- 完整样本必须同时具备三个里程碑的有限指标、非负且单调的累计训练时间、各 18 张 render/GT，以及最终 10k PLY；完整后直接跳过，不再读取该样本的 RGB/深度；
- 不完整样本不做样本内续跑。启动器会删除该样本的未完成结果目录，再从 0 训练到 10k，避免新旧里程碑产物混合；预处理缓存不会被删除；
- Center150 不生成 `chkpnt*.pth`，断点粒度就是完整样本；
- 实验根目录保存 `center150_protocol.json`。同一目录若检测到分辨率、迭代里程碑、置信度阈值或额外训练参数发生变化，会拒绝混用结果。

## 7. 自动汇总

只有 150 个样本全部通过完整性检查后，启动器才生成：

```text
output/omniscene_results/center150_112x200/
├── center150_metrics_summary.json
└── center150_metrics_summary.txt
```

汇总文件包含：

- 每个样本在 1k/5k/10k 的 PSNR、SSIM、LPIPS 和累计训练耗时；
- 150 个样本在每个里程碑的平均 PSNR、SSIM、LPIPS 和平均累计训练耗时；
- 分辨率、总迭代数、评估里程碑和生成时间。

如果任一样本失败或产物不完整，流程会报错退出而不会生成一个看似完整的 150 样本平均值；修复问题后重复同一命令即可续跑。
