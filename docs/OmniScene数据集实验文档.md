# OmniScene 数据集实验文档（Octree-GS / comp_svfgs 分支）

## 1. 背景与目标
- 当前分支为 `comp_svfgs`，目标是在本项目（逐场景优化的高斯重建）上复现 OmniScene（魔改 nuScenes）实验，用于与自研前馈方法对比。
- 参考项目：
  - `depthsplat`：前馈式高斯重建，已有 OmniScene 配置、加载与调用实现。
  - `SVF-GS`：完整支持 OmniScene 的前馈项目，包含绝对深度与置信度加载逻辑，可用于生成初始化点云。

## 2. Octree-GS 数据加载的约束
- 本项目支持三类数据入口：
  1) **Colmap**：`sparse/0` + `points3D.(bin/txt/ply)`。
  2) **Blender/NeRF 风格**：存在 `transforms_train.json` 时走该分支，可读取 `transforms_train.json`/`transforms_test.json`。
  3) **City**：自定义 JSON + PLY/LAS。
- 对于 **Colmap/City**，点云是必须的；对 **Blender**，若缺少 PLY 会随机初始化点云。
- 结论：**OmniScene 必须在预处理阶段生成点云（PLY），禁止随机初始化。若点云生成失败则直接报错退出。**

## 3. 数据预处理规划（输出到 output）
### 3.1 目录结构
- 在项目根目录创建 `output/`，统一存放预处理结果与训练输出。
- 每个 bin 作为一个“场景”，命名规则：`01_<bin_token>`。
- 预处理场景目录建议结构如下：
```
output/omniscene/
├── 01_<bin_token>/
│   ├── images/                 # 6 张输入 + 18 张输出
│   ├── transforms_train.json   # 6 张输入视角
│   ├── transforms_test.json    # 18 张输出视角
│   ├── points3D.ply            # 由绝对深度生成的初始化点云
│   └── meta.json               # 记录分辨率、bin_token、相机列表等
```

### 3.2 预处理步骤
1) 读取 OmniScene 的 bin 信息（默认 `val` 模式前 10 个 bin）。
2) 对每个 bin：
   - 选取 **6 张输入视角**（固定 key-frame）。
   - 选取 **18 张输出视角**（每个相机取 index `[1,2]`，共 12 张，再拼入 6 张输入）。
   - 统一 resize 到目标分辨率（默认 `112x200`，可选 `224x400`）。
   - 写入 `images/` 与 `transforms_train.json` / `transforms_test.json`。
3) 生成初始化点云（详见第 5 节），必须成功生成 PLY。
4) 保存 `meta.json`（方便排查与复现实验）。
5) 若场景目录已存在且完整，则跳过预处理。

## 4. comp_svfgs 数据加载模块设计
### 4.1 新增文件
- `comp_svfgs/dataset_omniscene.py`：参考 `depthsplat/src/dataset/dataset_omniscene.py`。

### 4.2 核心功能
- **加载 context / target**：仅保留 RGB + 相机内外参（不加载动态掩码/相对深度）。
- **支持模式**：`train / val / test / demo`，默认 `val`。
- **val 模式**：固定 10 个 bin，每个 bin 视为一个场景，适配逐场景优化方式。
- **load_conditions**：保持 `depthsplat` 版本的路径替换与 resize 逻辑（包括 `samples_small` / `sweeps_small` 和内参缩放方式），确保与 OmniScene 数据目录一致。

## 5. 初始化点云生成（绝对尺度深度）
### 5.1 参考实现
- 参考 `SVF-GS`：
  - `data/dataloader.py`
  - `data/transforms/loading.py`
- 使用 **Metric3D 深度**与**置信度**：
  - 深度路径：`samples_dptm_small` / `sweeps_dptm_small`
  - 置信度路径：`*_conf.npy`

### 5.2 处理流程
1) 读取输入视角的绝对深度与置信度，resize 到目标分辨率。
2) 置信度过滤：`conf > 0.3` 视为有效。
3) 通过内参将深度反投影到相机坐标系，再用 `c2w` 转到世界坐标系。
4) 合并 6 张输入视角点云，必要时按体素/采样率下采样。
5) 保存为 `points3D.ply`（用于 Octree-GS 初始化）。

### 5.3 坐标系注意事项
- OmniScene 在本项目中 **不需要 flip_yz**，与 `depthsplat` 的处理一致。
- 若沿用 `readCamerasFromTransforms`（其内部会做 `c2w[:3,1:3] *= -1`），需要在自定义 loader 中显式禁用该翻转，或改为使用自定义相机读取逻辑，确保 **全流程不做 flip_yz**。
- 仍建议用少量场景渲染做坐标一致性校验，避免方向错误导致重建退化。

## 6. 主流程整合（单阶段完成）
### 6.1 新增运行脚本
- 在项目根目录新增 `run_omniscene.py`（或同级脚本）。
- 该脚本完成：
  1) **预处理**（若未完成则生成）
  2) **训练**（逐场景优化）
  3) **渲染与评估**（调用 `train.py` 内置流程）

### 6.2 执行逻辑（伪流程）
```
for bin in val_bins:
  scene_dir = output/omniscene/01_<bin>
  if not preprocessed(scene_dir):
    preprocess(scene_dir)
  run_train(scene_dir, eval=True)
```

### 6.3 与现有 train.py 的对接要点
- `train.py` 自带训练+渲染+评估流程：
  - `--eval` 为 `True` 时会渲染 test set 并计算指标。
- 每个场景独立产出 `output/omniscene_results/<scene_name>/<exp>/<time>/`。

## 7. 参数约定
- **分辨率**：默认 `112x200`，可选 `224x400`。
- **-r 参数**：固定为 `1`（因为已经在预处理阶段手动 resize）。
- **输入视角**：固定 6 张（训练用）。
- **输出视角**：固定 18 张（评估用）。
- **迭代次数**：沿用默认值（后续可配置）。

## 8. 与 depthsplat 的关键差异
1) **训练范式**：
   - depthsplat：前馈式训练，batch 为多个 bin。
   - Octree-GS：逐场景优化，每个 bin 单独训练、渲染与评估。
2) **数据组织**：
   - depthsplat：直接读取原始 OmniScene 目录。
   - Octree-GS：需先预处理成 `transforms_train/test + points3D.ply` 形式。
3) **点云初始化**：
   - depthsplat：不需要点云。
   - Octree-GS：必须使用绝对深度生成点云，不允许随机点云。
4) **评估方式**：
   - depthsplat：由模型前向输出。
   - Octree-GS：由 `render.py`/`train.py` 渲染 test 视角并计算 PSNR/SSIM/LPIPS。

## 9. 待确认事项
- OmniScene 数据中 `Metric3D` 绝对深度与置信度文件是否完整存在。
- 坐标系一致性是否通过小场景验证（强调“不翻转”前提下的正确性）。
- 是否需要在评估阶段导出额外指标（如 PCC）。

## 10. 下一步实现清单（审阅通过后执行）
- 新建 `comp_svfgs/` 与 `output/` 目录。
- 实现 `comp_svfgs/dataset_omniscene.py` 与预处理逻辑。
- 新增 `run_omniscene.py` 并串联预处理 + 训练 + 渲染评估。
- 小规模 sanity check（1~2 个 bin）后再批量运行。
