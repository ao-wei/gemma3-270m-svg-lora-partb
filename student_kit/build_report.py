#!/usr/bin/env python3
"""Build the final Chinese Markdown and PDF report from audited artifacts."""

from __future__ import annotations

import argparse
import json
import shutil
from datetime import date
from pathlib import Path

import yaml


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def f(value, digits=4):
    return f"{value:.{digits}f}"


def experiment_rows():
    names = [
        ("E0", "e00_default", "default r=8", "screening"),
        ("E1", "e01_rank4", "rank 4", "screening"),
        ("E2", "e02_rank16", "rank 16", "screening"),
        ("E3", "e03_lr1e4", "lr=1e-4", "screening"),
        ("E4", "e04_lr5e4", "lr=5e-4", "screening"),
        ("E5", "e05_length3072", "length 3072", "screening"),
        ("E6", "e06_full_early_final", "195 rows, 4 epochs", "refinement"),
        ("E7", "e07_seed123_final", "seed 123", "refinement"),
        ("E8", "e08_short_curriculum", "shortest 110", "refinement"),
        ("E9", "e09_simplified_targets", "6 elements", "refinement"),
        ("E10", "e10_stage1", "3 elements", "refinement"),
        ("E11", "e11_fullsvg_lr5e5", "full SVG, lr=5e-5", "final"),
        ("E12", "e12_fullsvg_lr1e4", "full SVG, lr=1e-4", "final"),
        ("E13", "e13_full217_seed42", "full SVG, seed 42", "final"),
        ("E14", "e14_full217_seed123", "full SVG, seed 123", "final"),
    ]
    rows = []
    for name, slug, target, role in names:
        data = read_json(f"artifacts/experiment_summaries/{slug}.json")
        peak_bytes = data.get("peak_memory", {}).get("mps_driver_allocated_bytes")
        peak = None if peak_bytes is None else peak_bytes / (1024 ** 3)
        rows.append({
            "name": name, "target": target, "role": role,
            "rows": data["train_rows"], "dev": data["dev_rows"],
            "lr": data["config"]["learning_rate"], "length": data["config"]["max_length"],
            "eval_loss": data.get("eval_metrics", {}).get("eval_loss"),
            "seconds": None if name in {"E6", "E7"} else data["elapsed_seconds"],
            "timing_note": "checkpoint reload" if name in {"E6", "E7"} else "measured training",
            "peak_gib": peak,
            "truncated": data.get("sequence_lengths", {}).get("truncated_rows"),
            "epochs": data["config"]["num_train_epochs"],
            "rank": data["config"]["lora_r"],
        })
    return rows


CASE_IDS = (0, 4, 6, 7)

CASE_NOTES = {
    0: "微调前的输出无法作为单一 SVG 解析；两个最终模型虽然生成了完整 XML，但图元集中在负坐标并反复复制。语法改善是真实的，画面仍为空白。",
    4: "两个最终模型都只留下一个铺满画布的圆，主体水果和叶片消失。该例说明颜色或标签命中不能代替完整构图。",
    6: "seed 42 的程序化总分是 0.5837，为验证集中最高值之一，但实际渲染为空白。无效 polygon 属性和画布外坐标造成了最明显的代理指标高估。",
    7: "seed 42 生成了橙色圆盘，保真度代理分达到 0.4062，但提示中的深色徽章与中央火焰人物均未出现；seed 123 同一条样本又发生 XML 解析错误，显示训练结果对随机种子敏感。",
}


def build_markdown(primary, secondary, stage1, selection, manual, experiments, name, student_id):
    p0, p1 = primary["summary"]["base"], primary["summary"]["tuned"]
    s1 = secondary["summary"]["tuned"]
    warm = stage1["summary"]["tuned"]
    comp = primary["comparison"]
    verify = primary["verification"]
    yaml_text = Path("train_config.yaml").read_text(encoding="utf-8").rstrip()
    lines = [
        "# Gemma 3 270M 的 SVG 徽标 LoRA 微调实验",
        "", f"- 姓名：{name}", f"- 学号：{student_id}",
        "- 作业仓库：https://github.com/ao-wei/gemma3-270m-svg-lora-partb",
        "- 数据来源：https://github.com/roboticcam/logo-detailed-prompt",
        "", "## 摘要", "",
        "我在 Apple M4 上使用 Transformers 和 PEFT 对 Gemma 3 270M 进行 LoRA 微调。原始训练集有 219 条记录，其中两条只含冲突的 placeholder，因此仅在加载时过滤，不改动源文件。17 条验证数据一直保留到最后，没有用于选择学习率、epoch、随机种子或 checkpoint。前期的三图元模型作为预热；最终模型从该权重出发，用全部 217 条完整 SVG 继续训练一轮。",
        "",
        f"主模型将 fatal rate 从 {p0['fatal_rate']:.1%} 降到 {p1['fatal_rate']:.1%}，总 reward 从 {f(p0['total']['mean'])} 提高到 {f(p1['total']['mean'])}。但 fidelity 只从 {f(p0['fidelity']['mean'])} 变为 {f(p1['fidelity']['mean'])}，配对置信区间跨过 0；quality pass rate 也仍为 {p1['quality_pass_rate']:.1%}。结合全部渲染图，我把结论限定为“SVG 格式稳定性有提升，提示保真和视觉质量没有得到可靠改善”。",
        "", "## 1. 数据边界与训练方法", "",
        "训练加载器使用模型自带 chat template；system、user 与 padding token 的标签全部为 -100，仅 assistant SVG token 参与交叉熵。完整 chat 最大长度为 3535，最终 max_length=3584，所有最终训练记录零截断。LoRA rank=4、alpha=16、dropout=0.05，仅作用于 q_proj/k_proj/v_proj/o_proj；batch size=1、梯度累积=8、梯度检查点开启。训练是监督微调，reward 只用于内部选择和分析，并非 RL。",
        "",
        "为了在 16GB MPS 上保留完整 SVG，训练使用精确的分块词表投影损失：按 token 块计算 lm_head 与交叉熵，再按有效 assistant token 总数归一化。数值检查与原始全 logits loss 的绝对差为 0。precision:auto 仅对明确 BF16 算子不兼容错误重启 FP32；OOM 不会伪装成兼容问题。",
        "", "## 2. Reward v3 与 pass 定义", "",
        "总分公式为 0.30×语法安全 + 0.20×几何 + 0.15×结构样式 + 0.25×提示保真 + 0.10×反退化。提示保真内部为颜色 45%、图形 25%、空间 20%、构图 10%；不存在可靠空间要求时对其他分项重新归一化。空间关系只在元素可唯一匹配时评分，歧义标记 unscorable。stroke-width 必须有限、非负且不超过 viewBox 短边 10%。",
        "",
        "权重的考虑如下：",
        "",
        "- 语法与安全占 30%，因为无法解析或含脚本的 SVG 根本不能作为徽标使用，应首先失去大部分分数。",
        "- 几何占 20%，用来区分“XML 可解析”和“画布上真有内容”；负坐标、非有限数和异常粗的线条都会使可见结果失效。",
        "- 结构与样式占 15%，奖励可见图元和合理调色，但不让难以程序化判断的“美观”主导总分。",
        "- 提示保真占 25%，它直接对应作业目标，但颜色和图形关键词只是语义的近似，因此权重低于语法与几何之和。",
        "- 反退化占 10%，专门阻止截断、重复片段和模板泄漏等容易“刷分”的行为。",
        "",
        "- valid pass：非 fatal 且 validity ≥ 0.8。",
        "- quality pass：valid pass、total ≥ 0.5、fidelity ≥ 0.3，且无背景/空白退化。",
        "- 致命 XML/安全错误有总分上限；背景、画布外与重复退化进入 violations。",
        "", "## 3. E0–E14 实验过程", "",
        "| 实验 | 变量/目标 | r | 训练/开发 | lr | 长度 | epoch | eval loss | 秒 |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in experiments:
        loss = "-" if row["eval_loss"] is None else f(row["eval_loss"])
        seconds = "-" if row["seconds"] is None else f"{row['seconds']:.1f}"
        lines.append(f"| {row['name']} | {row['target']} | {row['rank']} | {row['rows']}/{row['dev']} | {row['lr']:.1e} | {row['length']} | {row['epochs']} | {loss} | {seconds} |")
    lines += [
        "",
        "E0–E5 分别比较了 rank、学习率和长度上限。rank 4 与 rank 8/16 的开发 loss 差别很小，因此后续使用参数更少的 rank 4。E4 在早期筛选中 loss 最低，但后续生成检查显示，仅靠 loss 不能保证 SVG 完整。",
        "",
        "E6–E10 尝试了更多样本、更多 epoch、按长度排序以及六图元/三图元派生目标。E6/E7 的摘要是从 checkpoint 重载后记录的，不把其十余秒耗时当作完整训练时间。E10 的三图元模型成为最终完整 SVG 训练的初始权重，但不是最终提交模型。",
        "",
        f"E11/E12 只在固定 22 条内部开发集比较。E12 的 fatal rate 为 {selection['candidates']['E12']['fatal_rate']:.1%}，且 fidelity 没有改善，因此按事先设定的门槛选择 E11 的 lr={selection['selected_learning_rate']:.1e}。E13/E14 都从 E10 权重出发，用全部 217 条完整 SVG 训练；seed 42 是主结果，seed 123 只用于检查稳健性。",
        "",
        "| 最终阶段 | 峰值 MPS driver memory | 最大序列 | 截断 |",
        "|---|---:|---:|---:|",
    ]
    for row in experiments:
        if row["role"] == "final":
            peak = "-" if row["peak_gib"] is None else f"{row['peak_gib']:.2f} GiB"
            trunc = "-" if row["truncated"] is None else row["truncated"]
            lines.append(f"| {row['name']} | {peak} | 3535 / {row['length']} | {trunc} |")
    lines += [
        "", "## 4. 最终 17 条验证结果", "",
        "| 模型 | total | validity | fidelity | fatal rate | valid pass | quality pass |",
        "|---|---:|---:|---:|---:|---:|---:|",
        f"| Base | {f(p0['total']['mean'])} | {f(p0['validity']['mean'])} | {f(p0['fidelity']['mean'])} | {p0['fatal_rate']:.1%} | {p0['valid_pass_rate']:.1%} | {p0['quality_pass_rate']:.1%} |",
        f"| Stage1（三图元） | {f(warm['total']['mean'])} | {f(warm['validity']['mean'])} | {f(warm['fidelity']['mean'])} | {warm['fatal_rate']:.1%} | {warm['valid_pass_rate']:.1%} | {warm['quality_pass_rate']:.1%} |",
        f"| E13 seed42 | {f(p1['total']['mean'])} | {f(p1['validity']['mean'])} | {f(p1['fidelity']['mean'])} | {p1['fatal_rate']:.1%} | {p1['valid_pass_rate']:.1%} | {p1['quality_pass_rate']:.1%} |",
        f"| E14 seed123 | {f(s1['total']['mean'])} | {f(s1['validity']['mean'])} | {f(s1['fidelity']['mean'])} | {s1['fatal_rate']:.1%} | {s1['valid_pass_rate']:.1%} | {s1['quality_pass_rate']:.1%} |",
        "", "seed42 相对 base 的配对结果：",
        "", "| 指标 | 均值差 | bootstrap 95% CI | 改善/持平/退化 |", "|---|---:|---:|---:|",
    ]
    for key, label in (("total", "total"), ("validity", "validity"), ("fidelity", "fidelity")):
        item = comp[key]; ci = item["bootstrap_95_ci"]
        lines.append(f"| {label} | {f(item['paired_mean_delta'])} | [{f(ci[0])}, {f(ci[1])}] | {item['improved']}/{item['unchanged']}/{item['worsened']} |")
    lines += [
        "", "## 5. 典型样本与 Goodhart", "",
        f"我逐条检查了 17 组 reference/base/seed42/seed123 图像。seed42 视觉有效 {manual['summary']['seed42_visually_valid']}/17、背景单图元 {manual['summary']['seed42_background_only']}/17；seed123 视觉有效 {manual['summary']['seed123_visually_valid']}/17、背景单图元 {manual['summary']['seed123_background_only']}/17。下面四例分别展示格式改善但视觉仍空白、单背景退化、代理分高估和随机种子不稳定。",
        "",
    ]
    secondary_by_id = {sample["id"]: sample for sample in secondary["samples"]}
    primary_by_id = {sample["id"]: sample for sample in primary["samples"]}
    for sample_id in CASE_IDS:
        left, right = primary_by_id[sample_id], secondary_by_id[sample_id]
        lines += [
            f"### 样本 {sample_id}", "",
            "| 参考 | 基座 | seed 42 | seed 123 |",
            "|---|---|---|---|",
            f"| ![reference](output/cases/sample_{sample_id:02d}_reference.png) | ![base](output/cases/sample_{sample_id:02d}_base.png) | ![seed42](output/cases/sample_{sample_id:02d}_seed42.png) | ![seed123](output/cases/sample_{sample_id:02d}_seed123.png) |",
            "",
            f"base / seed42 / seed123 的 total 为 {f(left['base']['reward']['total'])} / {f(left['tuned']['reward']['total'])} / {f(right['tuned']['reward']['total'])}；fidelity 为 {f(left['base']['reward']['fidelity'])} / {f(left['tuned']['reward']['fidelity'])} / {f(right['tuned']['reward']['fidelity'])}。{CASE_NOTES[sample_id]}",
            "",
        ]
    lines += ["## 6. 全部 17 条视觉审计", ""]
    for page in range(1, 5):
        lines += [f"![全部样本视觉审计 {page}](output/audit/contact_sheet_{page}.png)", ""]
    lines += [
        "## 7. 结论与局限", "",
        "完整 SVG 继续训练确实把 seed42 的 fatal rate 降至 0，并且 34 个独立复验输出全部一致；但模型学习到的是高度重复、坐标为负的模板，几乎没有前景落在画布内。fidelity 的配对置信区间跨 0，两个 seed 的 quality pass 都为 0。270M 容量、长 SVG token 序列与纯文本交叉熵共同限制了语义构图。后续应采用更强基座、语法约束解码、结构化 SVG 表示和基于栅格图像的感知损失/评测。",
        "", "## 8. 复现与验收", "",
        "```bash", "uv venv --python 3.12 .venv", "uv pip install --python .venv/bin/python -r requirements.txt",
        ".venv/bin/python -m unittest discover -s tests -v",
        ".venv/bin/python student_kit/train_peft.py --config train_config.yaml",
        ".venv/bin/python student_kit/eval_self.py --adapter adapter --output results.json",
        ".venv/bin/python student_kit/analyze_results.py results.json",
        ".venv/bin/python student_kit/build_visual_audit.py",
        ".venv/bin/python student_kit/verify_artifacts.py --results results.json --repro runs/repro_full.json --adapter adapter",
        "```", "",
        f"验收：测试 {verify['unit_test_count']}/{verify['unit_test_count']}；新进程 adapter 加载={verify['fresh_process_adapter_load']}；schema v2={verify['results_schema_v2_valid']}；确定性复验 base {verify['deterministic_base_matches']}/17、tuned {verify['deterministic_tuned_matches']}/17。",
        "", "## 附录 A：最终 train_config.yaml", "", "```yaml", yaml_text, "```", "",
        "## 附录 B：哈希与结果 schema", "",
        f"- 主 adapter SHA-256：`{verify['adapter_sha256']['adapter_model.safetensors']}`",
        f"- train.jsonl SHA-256：`{verify['data_sha256']['train.jsonl']}`",
        f"- valid.jsonl SHA-256：`{verify['data_sha256']['valid.jsonl']}`",
        "- results schema v2：每条样本含 raw_text、svg、reward、passes.valid、passes.quality；汇总含 pass_rate、valid_pass_rate、quality_pass_rate。",
        "- 关键产物：`results.json`、`results_seed123.json`、`results_stage1.json`、`render_manifest.json`、`manual_review.json`。",
    ]
    return "\n".join(lines) + "\n"


def build_pdf(primary, secondary, stage1, selection, manual, experiments, name, student_id, output):
    from reportlab.lib import colors
    from reportlab.lib.enums import TA_CENTER
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
    from reportlab.lib.units import mm
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.ttfonts import TTFont
    from reportlab.platypus import Image, KeepTogether, PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle, XPreformatted

    font = "/System/Library/Fonts/Supplemental/Arial Unicode.ttf"
    pdfmetrics.registerFont(TTFont("CJK", font))
    styles = getSampleStyleSheet()
    for style in styles.byName.values(): style.fontName = "CJK"
    styles.add(ParagraphStyle(name="TitleCJK", parent=styles["Title"], fontName="CJK", fontSize=22, leading=31, alignment=TA_CENTER, textColor=colors.HexColor("#17324D")))
    styles.add(ParagraphStyle(name="H2CJK", parent=styles["Heading2"], fontName="CJK", fontSize=14, leading=20, spaceBefore=10, spaceAfter=6, textColor=colors.HexColor("#176B87")))
    styles.add(ParagraphStyle(name="BodyCJK", parent=styles["BodyText"], fontName="CJK", fontSize=9, leading=14, spaceAfter=5))
    styles.add(ParagraphStyle(name="SmallCJK", parent=styles["BodyText"], fontName="CJK", fontSize=7, leading=9))
    styles.add(ParagraphStyle(name="CodeCJK", parent=styles["Code"], fontName="CJK", fontSize=5.8, leading=7.4, backColor=colors.HexColor("#F4F6F7"), borderPadding=5))

    def footer(canvas, doc):
        canvas.setAuthor(name); canvas.setCreator(name)
        canvas.setTitle("Gemma 3 270M 的 SVG 徽标 LoRA 微调实验")
        canvas.setSubject("LoRA 训练、奖励函数与基座对比分析")
        canvas.saveState(); canvas.setFont("CJK", 8); canvas.setFillColor(colors.HexColor("#607080"))
        canvas.drawString(18*mm, 11*mm, "Gemma 3 270M SVG 徽标 LoRA 实验")
        canvas.drawRightString(192*mm, 11*mm, str(doc.page)); canvas.restoreState()

    def table(rows, widths, size=7.3):
        t = Table(rows, colWidths=widths, repeatRows=1)
        t.setStyle(TableStyle([("FONTNAME", (0,0), (-1,-1), "CJK"), ("FONTSIZE", (0,0), (-1,-1), size), ("BACKGROUND", (0,0), (-1,0), colors.HexColor("#176B87")), ("TEXTCOLOR", (0,0), (-1,0), colors.white), ("ROWBACKGROUNDS", (0,1), (-1,-1), [colors.white, colors.HexColor("#F3F7F8")]), ("GRID", (0,0), (-1,-1), .3, colors.HexColor("#A8B6C2")), ("VALIGN", (0,0), (-1,-1), "MIDDLE"), ("ALIGN", (1,1), (-1,-1), "CENTER"), ("PADDING", (0,0), (-1,-1), 3)]))
        return t

    doc = SimpleDocTemplate(str(output), pagesize=A4, leftMargin=17*mm, rightMargin=17*mm, topMargin=16*mm, bottomMargin=18*mm, title="Gemma 3 270M 的 SVG 徽标 LoRA 微调实验")
    story = [Spacer(1, 30*mm), Paragraph("Gemma 3 270M 的 SVG 徽标<br/>LoRA 微调实验", styles["TitleCJK"]), Spacer(1, 16*mm), table([["姓名", name], ["学号", student_id], ["日期", date.today().isoformat()]], [30*mm, 85*mm], 10), PageBreak()]
    p0, p1 = primary["summary"]["base"], primary["summary"]["tuned"]
    s1, warm, comp = secondary["summary"]["tuned"], stage1["summary"]["tuned"], primary["comparison"]
    story += [Paragraph("摘要", styles["H2CJK"]), Paragraph(f"我先比较了 rank、学习率、序列长度、数据量和派生目标，再从三图元预热权重出发，对全部 217 条完整 SVG 训练。主模型将 fatal rate 从 {p0['fatal_rate']:.1%} 降到 {p1['fatal_rate']:.1%}，但 fidelity 只从 {f(p0['fidelity']['mean'])} 变为 {f(p1['fidelity']['mean'])}，quality pass 仍为 {p1['quality_pass_rate']:.1%}。因此结论是格式稳定性提升，而不是视觉质量改善。", styles["BodyCJK"]),
              Paragraph("1. 数据、训练与评测边界", styles["H2CJK"]), Paragraph("原始 train.jsonl 有 219 条，加载时过滤两条冲突 placeholder，不改动原文件。17 条最终验证数据不用于调参。训练使用模型自带的 chat template，仅 assistant SVG token 参与交叉熵。LoRA r=4、alpha=16，仅修改 q/k/v/o 投影层。最终训练使用 BF16、batch 1、梯度累积 8 和分块词表损失，完整样本最长 3535 token，没有截断。", styles["BodyCJK"]),
              Paragraph("2. Reward v3 与 pass 定义", styles["H2CJK"])]
    reward_rows = [["分项", "权重", "检查"], ["语法/安全", "30%", "可解析、唯一根节点、无脚本或外链"], ["几何", "20%", "viewBox、有限数、stroke、画布内前景"], ["结构/样式", "15%", "可见图元、调色、非空白"], ["提示保真", "25%", "颜色45%、图形25%、空间20%、构图10%"], ["反退化", "10%", "截断、重复、模板泄漏、异常长度"]]
    rationale = "语法与安全的权重最高，因为这类错误会让 SVG 根本无法使用。几何分用来防止“XML 可解析但画布是空的”。提示保真直接对应任务，但颜色和图形关键词只是语义近似，所以不让它压过语法与几何。反退化分则专门防止重复或截断输出利用代理指标获利。"
    story += [table(reward_rows, [30*mm, 17*mm, 125*mm], 7.7), Paragraph(rationale, styles["BodyCJK"]), Paragraph("valid pass 要求非 fatal 且 validity≥0.8；quality pass 还要求 total≥0.5、fidelity≥0.3，且无背景/空白退化。", styles["BodyCJK"]), PageBreak(), Paragraph("3. E0–E14 实验过程", styles["H2CJK"])]

    def experiment_table(role):
        rows = [["实验", "变量/目标", "r", "lr", "len", "训/开发", "epoch", "eval loss", "秒"]]
        for r in experiments:
            if r["role"] == role:
                rows.append([r["name"], r["target"], r["rank"], f"{r['lr']:.1e}", r["length"], f"{r['rows']}/{r['dev']}", r["epochs"], "-" if r["eval_loss"] is None else f(r["eval_loss"]), "-" if r["seconds"] is None else f"{r['seconds']:.0f}"])
        return table(rows, [15*mm, 37*mm, 10*mm, 19*mm, 16*mm, 20*mm, 16*mm, 22*mm, 17*mm], 6.4)

    story += [Paragraph("初步筛选（E0–E5）", styles["BodyCJK"]), experiment_table("screening"), Paragraph("rank 4 与 rank 8/16 的开发 loss 接近，后续因此选用参数更少的 rank 4。E4 的 loss 较低，但生成检查表明 loss 不能单独预测 SVG 完整性。", styles["BodyCJK"]), Paragraph("数据与目标调整（E6–E10）", styles["BodyCJK"]), experiment_table("refinement"), Paragraph("E6/E7 的摘要来自 checkpoint 重载，因此不报十余秒的重载耗时。E10 是三图元预热模型，不是最终提交模型。", styles["BodyCJK"]), Paragraph("完整 SVG 阶段（E11–E14）", styles["BodyCJK"]), experiment_table("final"), Paragraph(f"E12 的 fatal rate 为 {selection['candidates']['E12']['fatal_rate']:.1%} 且 fidelity 未改善，因此按事先设定的规则选择 E11 的 lr={selection['selected_learning_rate']:.1e}。E13/E14 用全部 217 条完整 SVG 训练，两次都零截断，峰值 MPS driver memory 分别为 3.84 GiB 和 3.82 GiB。", styles["BodyCJK"]), PageBreak(), Paragraph("4. 最终定量结果", styles["H2CJK"])]
    result_rows = [["模型", "total", "validity", "fidelity", "fatal", "valid pass", "quality pass"]]
    for label, x in (("Base", p0), ("Stage1", warm), ("E13 seed42", p1), ("E14 seed123", s1)):
        result_rows.append([label, f(x["total"]["mean"]), f(x["validity"]["mean"]), f(x["fidelity"]["mean"]), f"{x['fatal_rate']:.1%}", f"{x['valid_pass_rate']:.1%}", f"{x['quality_pass_rate']:.1%}"])
    story += [table(result_rows, [31*mm, 23*mm, 23*mm, 23*mm, 21*mm, 25*mm, 27*mm], 7.2)]
    ci = comp["fidelity"]["bootstrap_95_ci"]
    story += [Paragraph(f"seed42 total 均值差={f(comp['total']['paired_mean_delta'])}，95% CI=[{f(comp['total']['bootstrap_95_ci'][0])}, {f(comp['total']['bootstrap_95_ci'][1])}]。fidelity 均值差={f(comp['fidelity']['paired_mean_delta'])}，95% CI=[{f(ci[0])}, {f(ci[1])}]，跨 0；因此不能宣称提示保真度显著改善。", styles["BodyCJK"]), PageBreak()]

    primary_by_id = {sample["id"]: sample for sample in primary["samples"]}
    secondary_by_id = {sample["id"]: sample for sample in secondary["samples"]}
    story += [Paragraph("5. 典型样本对比", styles["H2CJK"])]
    for index, sample_id in enumerate(CASE_IDS, 1):
        left, right = primary_by_id[sample_id], secondary_by_id[sample_id]
        images = [Image(f"output/cases/sample_{sample_id:02d}_{variant}.png", width=38*mm, height=38*mm) for variant in ("reference", "base", "seed42", "seed123")]
        labels = [Paragraph(label, styles["SmallCJK"]) for label in ("参考", "基座", "seed 42", "seed 123")]
        grid = Table([images, labels], colWidths=[42*mm]*4)
        grid.setStyle(TableStyle([("ALIGN", (0,0), (-1,-1), "CENTER"), ("VALIGN", (0,0), (-1,-1), "MIDDLE"), ("GRID", (0,0), (-1,0), .3, colors.HexColor("#C7D0D6")), ("BACKGROUND", (0,0), (-1,0), colors.HexColor("#FAFAFA")), ("PADDING", (0,0), (-1,-1), 2)]))
        note = Paragraph(f"样本 {sample_id}：base / seed42 / seed123 的 total 为 {f(left['base']['reward']['total'])} / {f(left['tuned']['reward']['total'])} / {f(right['tuned']['reward']['total'])}，fidelity 为 {f(left['base']['reward']['fidelity'])} / {f(left['tuned']['reward']['fidelity'])} / {f(right['tuned']['reward']['fidelity'])}。{CASE_NOTES[sample_id]}", styles["BodyCJK"])
        story += [KeepTogether([grid, note]), Spacer(1, 3*mm)]
        if index == 2:
            story += [PageBreak(), Paragraph("5. 典型样本对比（续）", styles["H2CJK"])]
    story += [PageBreak()]
    for page in range(1, 5):
        story += [Paragraph(f"6. 全部 17 条视觉审计（{page}/4）", styles["H2CJK"]), Image(f"output/audit/contact_sheet_{page}.png", width=174*mm, height=(174*mm)*(1316/920 if page < 4 else 580/920)), PageBreak()]
    story += [Paragraph("7. Goodhart 分析、结论与局限", styles["H2CJK"]), Paragraph(f"逐条检查后，seed42 视觉有效 {manual['summary']['seed42_visually_valid']}/17，背景单图元 {manual['summary']['seed42_background_only']}/17；seed123 视觉有效 {manual['summary']['seed123_visually_valid']}/17，背景单图元 {manual['summary']['seed123_background_only']}/17。许多输出的 XML 完整，但图元位于画布外或只剩巨型圆。代理分因语法改善而上升，视觉质量却没有跟上，这正是本次实验中最明显的 Goodhart 现象。两个 seed 的 quality pass 都为 0，所以我不把总分上升解释为语义生成成功。", styles["BodyCJK"]), Paragraph("270M 模型容量、长 SVG token 序列和纯文本交叉熵是主要限制。更有希望的后续方向包括语法约束解码、结构化 SVG 表示，以及基于栅格图像的感知评测。", styles["BodyCJK"]), Paragraph("8. 复现与验收", styles["H2CJK"])]
    verify = primary["verification"]
    checks = [["验收项", "结果"], ["单元测试", f"{verify['unit_test_count']}/{verify['unit_test_count']}"], ["adapter 新进程加载", str(verify["fresh_process_adapter_load"])], ["results schema v2", str(verify["results_schema_v2_valid"])], ["确定性复验", f"base {verify['deterministic_base_matches']}/17; tuned {verify['deterministic_tuned_matches']}/17"], ["train/valid SHA-256", verify["data_sha256"]["train.jsonl"][:16]+"… / "+verify["data_sha256"]["valid.jsonl"][:16]+"…"], ["adapter SHA-256", verify["adapter_sha256"]["adapter_model.safetensors"]]]
    story += [table(checks, [55*mm, 118*mm], 7.5), Paragraph("关键产物：results.json、results_seed123.json、results_stage1.json、render_manifest.json、manual_review.json。作业仓库不包含基座模型、.venv、缓存或 checkpoint。", styles["BodyCJK"]), PageBreak(), Paragraph("附录 A：最终 train_config.yaml", styles["H2CJK"]), XPreformatted(Path("train_config.yaml").read_text(encoding="utf-8"), styles["CodeCJK"]), Paragraph("附录 B：Reward、schema 与复现命令", styles["H2CJK"]), Paragraph("结果 schema v2 每条样本保存 prompt、reference_svg、base/tuned raw_text、清理后 svg、reward 分项、violations、passes.valid 和 passes.quality；汇总保存 pass_rate、valid_pass_rate 与 quality_pass_rate。", styles["BodyCJK"])]
    commands = """uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python -r requirements.txt
.venv/bin/python -m unittest discover -s tests -v
.venv/bin/python student_kit/train_peft.py --config train_config.yaml
.venv/bin/python student_kit/eval_self.py --adapter adapter --output results.json
.venv/bin/python student_kit/analyze_results.py results.json
.venv/bin/python student_kit/build_visual_audit.py
.venv/bin/python student_kit/verify_artifacts.py --results results.json --repro runs/repro_full.json --adapter adapter"""
    story += [XPreformatted(commands, styles["CodeCJK"])]
    doc.build(story, onFirstPage=footer, onLaterPages=footer)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", default="敖炜")
    parser.add_argument("--student-id", default="202521080810")
    args = parser.parse_args()
    primary, secondary, stage1 = read_json("results.json"), read_json("results_seed123.json"), read_json("results_stage1.json")
    selection, manual, experiments = read_json("artifacts/model_selection.json"), read_json("manual_review.json"), experiment_rows()
    cases_dir = Path("output/cases")
    cases_dir.mkdir(parents=True, exist_ok=True)
    for sample_id in CASE_IDS:
        for variant in ("reference", "base", "seed42", "seed123"):
            source = Path("output/rendered_final") / f"sample_{sample_id:02d}_{variant}.png"
            shutil.copy2(source, cases_dir / source.name)
    Path("report.md").write_text(build_markdown(primary, secondary, stage1, selection, manual, experiments, args.name, args.student_id), encoding="utf-8")
    output = Path("output/pdf/partB_svg_lora_report.pdf"); output.parent.mkdir(parents=True, exist_ok=True)
    build_pdf(primary, secondary, stage1, selection, manual, experiments, args.name, args.student_id, output)
    print(f"wrote report.md and {output}")


if __name__ == "__main__": main()
