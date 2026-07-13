# Gemma 3 270M SVG LoRA - Part B

本仓库包含“详细视觉提示词 -> SVG 徽标”课程作业的完整可复现实验：程序化 reward、Gemma 3 270M LoRA 训练、基座/微调配对评测、逐样本结果和中文报告。

## 数据

- `train.jsonl`：219 条原始训练记录，其中两条冲突的 `placeholder` prompt 在训练加载时过滤，源文件不修改；
- `valid.jsonl`：17 条最终验证配对，不参与训练或模型选择；
- 数据来源：[roboticcam/logo-detailed-prompt](https://github.com/roboticcam/logo-detailed-prompt)。

每条记录均为 `system/user/assistant` chat 格式。训练时 system/user/padding 标签设为 `-100`，只对 assistant 的 SVG token 计算交叉熵。

## 环境与复现

基座模型应位于 `gemma3-270m-it/`，该目录不会提交到 GitHub。

```bash
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python -r requirements.txt

.venv/bin/python student_kit/audit_data.py
.venv/bin/python -m unittest discover -s tests -v
.venv/bin/python student_kit/train_peft.py --config train_config.yaml
.venv/bin/python student_kit/eval_self.py \
  --model gemma3-270m-it --adapter adapter --valid valid.jsonl --output results.json
.venv/bin/python student_kit/analyze_results.py results.json
.venv/bin/python student_kit/render_svg.py results.json
.venv/bin/python student_kit/verify_artifacts.py
.venv/bin/python student_kit/build_report.py
```

Apple Silicon 训练需要 PyTorch MPS 可用。训练不使用 4-bit/8-bit 量化。实验矩阵见 `experiment_matrix.yaml`，可运行：

```bash
.venv/bin/python student_kit/run_experiments.py
```

长序列 MPS 训练使用安全内存水位：

```bash
PYTORCH_MPS_HIGH_WATERMARK_RATIO=1.1 \
PYTORCH_MPS_LOW_WATERMARK_RATIO=0.9 \
.venv/bin/python student_kit/train_peft.py --config train_config.yaml
```

## Reward

`student_kit/reward.py` 返回：

- `total`：五个分项的加权总分；
- `validity`：语法/安全和几何有效性的组合；
- `fidelity`：颜色、可验证图元词和构图代理；
- `components`、`violations`、`metadata`：用于解释每个分数。

Reward v2 要求正确 SVG namespace，检查画布内前景、重复图元、巨型背景和折叠路径。致命 XML/安全错误与背景退化都有总分上限。该 reward 不能替代视觉评审；具体 Goodhart 案例见 `report.md`。

## 最终配置的边界

最终 adapter 使用 rank 4、学习率 `2e-4`、长度 1024、2 epochs，并把每个训练目标派生为“前三个可见图元”的完整 SVG。这提高了格式稳定性，却会损失细节。最终验证表明 fatal rate 显著下降，但多数可解析输出仍是单色背景退化；本仓库不把它表述为高质量视觉生成成功。

## 主要提交物

- `adapter/`：最终 PEFT LoRA adapter；
- `reward.py`：提交要求的顶层 reward 副本；
- `train_config.yaml`：最终超参数；
- `results.json`：17 条验证集的基座/LoRA 逐样本结果；
- `report.md` 与 `output/pdf/partB_svg_lora_report.pdf`：分析报告。
