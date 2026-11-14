# Structure-conditioned counting RL (FLUX.2 Klein Base 9B)

同一个工程、两份配置、两套 Trainer。推理时都只保留 Student LoRA：普通数量 prompt + 50-step ODE，不再给结构提示，也不做 best-of-G。

| | **Version A 纯蒸馏** | **Version B GRPO-Fast + KL** |
|---|---|---|
| 配置 | `configs/distill.yaml` | `configs/fast_grpo_distill.yaml` |
| 入口 | `--method distill` | `--method fast_grpo` |
| Student | 1 条普通 prompt ODE | **4** 条普通 prompt SDE（group size=4） |
| Teacher | 至多 1 条结构残差 ODE | 至多 1 条结构残差 SDE（配对一个错误 student 的噪声） |
| 分支点 | 完成 **2** 个 ODE step 之后的下一次转移 | 默认配置同左；全量交付配置为 **3** |
| SDE / GRPO | 关 | 开，单步随机转移 |
| Reward | 只筛选教师 | Group Advantage **加上** 教师筛选 |
| 损失 | \(w\,\mathrm{MSE}(v_\phi,v_T)+0.1\mathcal L_{\mathrm{keep}}\) | \(\mathcal L_{\mathrm{GRPO}}+\lambda_D w\,\mathrm{MSE}_{3:15}+\beta\mathcal L_{\mathrm{ref}}\)（全量交付配置） |
| 学习率 | \(1\times10^{-5}\) | \(5\times10^{-6}\) |
| 预算单位 | 240 个案例 | 96 个 group |
| 终点图上限 | 480（1+1） | 480（4+1） |

Teacher 目标（两版共用）：

\[
v_T = v_{\mathrm{old}} + \gamma(v_0^{S}-v_0^{P}),\quad \gamma=1
\]

门控：

\[
R=\frac{1}{1+|\hat N-N|}+0.3\cdot\mathbf 1[\hat N=N],\quad
w=\mathrm{clip}((R_T-R_P)/0.5,\,0,1)
\]

Version B 的 KL 是同方差高斯、按维平均：

\[
\mathrm{KL}_{\mathrm{dim}}=\frac{\|\mu_T-\mu_\phi\|_2^2}{2d\sigma^2}
\]

教师动作**不进入** GRPO group。四个 student 奖励完全相同则跳过该组 GRPO。默认 `half_inexact`：组内至少一半 inexact 才跑教师，并从错误成员里按 seed 抽一个做锚点。CountGD 检测失败（`count=-1`）的组整组作废，不进 GRPO、不调教师。

Version B 的单步随机核是系数保持 DDIM 均值加保守噪声 \(s=\eta t'\sqrt{1-t}\)（\(\eta=0.3\)）。`log_prob` 仍按 Flow-GRPO 对 latent 维做均值，不是全维联合密度。ODE 后缀保持 Euler。旧 smoke 结果与新核不可直接比较。

## 运行

在 `flux2-count` 目录下，使用已有 FLUX.2 Klein 的环境（需 `torch`、`diffusers`、`pyyaml`）：

```bash
cd /home/lsj/flux2-count
python -m count_rl --selftest
python -m count_rl --compare
python -m count_rl --method distill --dry-run
python -m count_rl --method fast_grpo --dry-run

# 正式训练（会加载 9B）
CUDA_VISIBLE_DEVICES=0 python -m count_rl --method distill
CUDA_VISIBLE_DEVICES=0 python -m count_rl --method fast_grpo
CUDA_VISIBLE_DEVICES=0 python -m count_rl --method fast_grpo_b50
```

CountGD 路径写在 `configs/common.yaml` 的 `reward` 段，与现有 V5.1 管线一致（阈值 0.26）。

## 结构化 prompt

沿用已跑过的 `structured_exact_counting_5x6`。教师只多一句布局，类别短语与学生相同：

| N | 布局 |
|---|---|
| 5 | 一行五个 |
| 6 | 后 3 前 3 |
| 7 | 后 4 前 3 |
| 8 | 后 4 前 4 |
| 9 | 3×3 |
| 10 | 后 5 前 5 |

学生（N=10 苹果）不含布局句；教师在同一场景句后插入 two rows of five。mug 在该 spec 里的短语是 `cups`，不是 1120 的 `ceramic mugs`，这是为了让教师/学生只差布局句。

## LoRA

覆盖全部 8 个 dual-stream + 24 个 single-stream block 的选定投影：dual 的 Q/K/V/O 与 text-side `add_*`；single 的融合 `to_qkv_mlp_proj` 与 `to_out`。rank 16 / alpha 32。CFG 无条件分支始终走冻结 Base。

## 评测时盯的数

- 分数量 exact accuracy，尤其 8 / 9 / 10
- Teacher gate 通过率（按 target）
- Version B：group reward std、零 Advantage 比例
- 实际终点图数量和 GPU 时间（同样 480 张图，B 的 Transformer 调用仍更多）
- 布局是否塌成模板；不能把准确率上升直接说成抽象计数能力


## Full 960-group delivery run (branch 3, eta 2.0)

The handoff configuration is `configs/fast_grpo_full960_branch3_eta2.yaml`. It trains 960 groups across 10 categories and targets 5-10 (160 groups per target), uses shared branch index 3, eta=2.0, and reward-gated multi-step ODE velocity distillation over transition indices [3, 15). Mug/bottle phrases are overridden only for this config (`ceramic mugs`, `water bottles`); legacy configs remain unchanged.

Run from `/home/lsj/flux2-count` with the project environment activated:

```bash
python -m count_rl --selftest
python -m count_rl --config count_rl/configs/fast_grpo_full960_branch3_eta2.yaml --dry-run
# One-group smoke run; use only after selecting an available GPU.
CUDA_VISIBLE_DEVICES=<free_gpu_id> python -m count_rl --config count_rl/configs/fast_grpo_full960_branch3_eta2.yaml --limit 1
# Evaluate final LoRA on the fixed 960-case plain-prompt ODE set.
python -m count_rl --config count_rl/configs/fast_grpo_full960_branch3_eta2.yaml --eval-lora /path/to/lora_final.pt
```

The trainer logs per-group collection time/peak CUDA memory and per-update time/peak CUDA memory. Do not start the smoke run on a GPU that is already occupied.

The eta=1.0 configuration (`configs/fast_grpo_full960_branch3_eta1.yaml`) remains unchanged for the already running mini15 test, reproducibility, and comparison. The eta=2.0 configuration is for NEW training launches; do not relabel the running mini15 results.
