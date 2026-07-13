#!/usr/bin/env python3
"""Build the Chinese Markdown and PDF assignment reports from real results."""

from __future__ import annotations

import argparse
import json
import shutil
from datetime import date
from pathlib import Path


def fmt(value, digits=4):
    return f"{value:.{digits}f}"


def load_experiments(runs_dir: Path):
    rows = []
    names = {
        "e00_default": "E0 default", "e01_rank4": "E1 r=4", "e02_rank16": "E2 r=16",
        "e03_lr1e4": "E3 lr=1e-4", "e04_lr5e4": "E4 lr=5e-4",
        "e05_length3072": "E5 len=3072", "e06_full_early_final": "E6 full-4ep",
        "e07_seed123_final": "E7 seed=123", "e08_short_curriculum": "E8 shortest-110",
        "e09_simplified_targets": "E9 six-elements", "e10_three_elements": "E10 final-3elem",
    }
    for path in sorted(runs_dir.glob("*/training_summary.json")):
        if path.parent.name not in names:
            continue
        data = json.loads(path.read_text(encoding="utf-8"))
        config = data["config"]
        rows.append({
            "name": names[path.parent.name],
            "r": config["lora_r"],
            "lr": config["learning_rate"],
            "length": config["max_length"],
            "samples": data["train_rows"],
            "epochs": config["num_train_epochs"],
            "eval_loss": data.get("eval_metrics", {}).get("eval_loss"),
            "seconds": None if path.parent.name in {"e06_full_early_final", "e07_seed123_final"} else data["elapsed_seconds"],
        })
    return rows


def choose_cases(samples):
    total_delta = lambda sample: sample["tuned"]["reward"]["total"] - sample["base"]["reward"]["total"]
    fidelity_delta = lambda sample: sample["tuned"]["reward"]["fidelity"] - sample["base"]["reward"]["fidelity"]
    improved = max(samples, key=lambda sample: (total_delta(sample), fidelity_delta(sample)))
    positives = [sample for sample in samples if total_delta(sample) > 0]
    slight = min(positives, key=lambda sample: (total_delta(sample), abs(fidelity_delta(sample))))
    unchanged = min(samples, key=lambda sample: abs(total_delta(sample)))
    degraded = min(samples, key=fidelity_delta)
    candidates = [improved, slight, unchanged, degraded]
    ranked = sorted(samples, key=total_delta, reverse=True)
    seen = set()
    result = []
    for sample in candidates + ranked:
        if sample["id"] not in seen:
            seen.add(sample["id"])
            result.append(sample)
        if len(result) == 4:
            break
    return result


def case_note(sample):
    notes = {
        3: "总分明显提高，但图形落在画布外；改善主要是正确 namespace 与闭合结构，并非视觉成功。",
        0: "总分轻微提高，但仅生成重复的浅色圆，提示保真度反而下降。",
        5: "基座与 LoRA 都未生成唯一完整 SVG，是明确的无变化失败。",
        14: "LoRA 输出可解析却是空白画布，fidelity 明显退化，构成典型 Goodhart 反例。",
    }
    return notes.get(sample["id"], "需结合渲染图判断结构有效性与视觉语义是否一致。")


def build_markdown(data, experiments, name, student_id):
    base, tuned, comparison = data["summary"]["base"], data["summary"]["tuned"], data["comparison"]
    lines = [
        "# Gemma 3 270M 基于 LoRA 的 SVG 徽标生成",
        "",
        f"- 姓名：{name}", f"- 学号：{student_id}",
        "",
        "## 摘要",
        "",
        f"原始训练文件含 219 行，其中两行仅为冲突的 placeholder，训练时不修改源文件而过滤它们。其余 217 条固定拆为 195 条训练和 22 条开发数据；17 条最终验证集完全隔离。LoRA 将总 reward 从 {fmt(base['total']['mean'])} 提高到 {fmt(tuned['total']['mean'])}，但人工渲染显示多数可解析输出仍退化为单色背景，因此本报告把提升限定为格式稳定性，而不宣称高质量图形生成。",
        "",
        "## 1. 数据与方法",
        "",
        "原始数据包含 219 条训练记录和 17 条最终验证样本；两条 user prompt 仅为 placeholder 且目标冲突，训练时程序化过滤，源 JSONL 保持不变。其余数据内部固定划分 22 条开发样本用于超参数筛选和早停；公开验证集不参与训练或模型选择。所有输入使用基座模型自带 chat template，损失只计算 assistant 的 SVG token。",
        "",
        "采用 PEFT LoRA，仅作用于 q_proj、k_proj、v_proj、o_proj。训练设备为 Apple M4 MPS，不使用量化；batch size 为 1，并使用梯度累积和梯度检查点。最终课程式目标保留每个源 SVG 的前三个可见图元，以降低 270M 模型的序列难度；这是训练时派生视图，原始 JSONL 未修改。训练优化的是监督交叉熵，reward 只用于选择和分析，因此本项目不声称进行了强化学习。",
        "",
        "## 2. Reward 设计",
        "",
        "| 分项 | 权重 | 程序化检查 |",
        "|---|---:|---|",
        "| 语法与安全 | 30% | 单一 SVG、XML 可解析、xmlns、无脚本/外链/事件处理器 |",
        "| 几何有效性 | 20% | viewBox、有限数值、主体坐标与尺寸合理 |",
        "| 结构与样式 | 15% | 可见图元、元素数量、显式颜色和调色板规模 |",
        "| 提示词保真度 | 25% | 提示颜色、可验证图元词和基本构图覆盖 |",
        "| 反退化 | 10% | 截断、模板泄漏、异常长度和重复结构 |",
        "",
        "解析失败或包含主动内容的 SVG 被设置总分上限，避免用关键词堆砌绕过有效性检查。该 reward 无法可靠判断抽象图标语义和整体美感，因此最终仍需人工检查渲染图。",
        "",
        "## 3. 实验",
        "",
        "| 实验 | rank | 学习率 | 长度 | 样本 | epoch | 内部 eval loss | 耗时(s) |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in experiments:
        loss = "-" if row["eval_loss"] is None else fmt(row["eval_loss"])
        seconds = "-" if row["seconds"] is None else f"{row['seconds']:.1f}"
        lines.append(f"| {row['name']} | {row['r']} | {row['lr']:.1e} | {row['length']} | {row['samples']} | {row['epochs']} | {loss} | {seconds} |")
    lines += [
        "",
        "## 4. 最终验证结果",
        "",
        "| 指标 | 基座 | LoRA | 配对差值 | 95% bootstrap CI | 改善/持平/退化 |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for metric, label in (("total", "总 reward"), ("validity", "有效性"), ("fidelity", "保真度")):
        comp = comparison[metric]
        ci = comp["bootstrap_95_ci"]
        lines.append(f"| {label} | {fmt(base[metric]['mean'])} | {fmt(tuned[metric]['mean'])} | {fmt(comp['paired_mean_delta'])} | [{fmt(ci[0])}, {fmt(ci[1])}] | {comp['improved']}/{comp['unchanged']}/{comp['worsened']} |")
    lines += [
        "",
        f"基座 fatal rate 为 {base['fatal_rate']:.1%}，LoRA fatal rate 为 {tuned['fatal_rate']:.1%}。由于验证集仅 17 条，置信区间和逐例方向比单一均值更重要。",
        "",
        "## 5. 案例与 Goodhart 检查",
        "",
        "PDF 版本展示四组 reference/base/LoRA 渲染对比：指标明显改善、轻微改善、无变化以及 fidelity 退化。人工检查发现，LoRA 多数输出虽使用正确 namespace 并完整闭合，却退化为重复单色圆或画布外图元。Reward v2 因此新增正确 namespace、画布内前景、巨型背景与重复图元检查并给退化输出设置 0.35 上限。即便如此，分数提升主要仍来自 validity。",
        "",
        "| 样本 | total: base→LoRA | fidelity: base→LoRA | 人工结论 |",
        "|---:|---:|---:|---|",
        "",
    ]
    for sample in choose_cases(data["samples"]):
        lines.append(f"| {sample['id']} | {fmt(sample['base']['reward']['total'])}→{fmt(sample['tuned']['reward']['total'])} | {fmt(sample['base']['reward']['fidelity'])}→{fmt(sample['tuned']['reward']['fidelity'])} | {case_note(sample)} |")
    for sample in choose_cases(data["samples"]):
        sample_id = sample["id"]
        lines += [
            "",
            f"### 样本 {sample_id}",
            "",
            "| 参考目标 | 基座 | LoRA |",
            "|---|---|---|",
            f"| ![reference](output/examples/sample_{sample_id:02d}_reference.png) | ![base](output/examples/sample_{sample_id:02d}_base.png) | ![LoRA](output/examples/sample_{sample_id:02d}_tuned.png) |",
            "",
            case_note(sample),
        ]
    lines += [
        "",
        "## 6. 局限与结论",
        "",
        "270M 模型容量极小，且前三图元课程牺牲了细节覆盖。LoRA 将 fatal rate 从 100% 降到 17.6%，证明格式学习有效；但 14 个非致命输出中仍有 13 个触发背景/画布退化上限，不能视为语义生成成功。后续应使用更强模型、结构化 SVG tokenization、分阶段增加图元，并加入栅格感知或视觉模型评测。",
        "",
        "## 7. 复现",
        "",
        "```bash",
        "uv venv --python 3.12 .venv",
        "uv pip install --python .venv/bin/python -r requirements.txt",
        ".venv/bin/python student_kit/audit_data.py",
        ".venv/bin/python student_kit/train_peft.py --config train_config.yaml",
        ".venv/bin/python student_kit/eval_self.py --model gemma3-270m-it --adapter adapter",
        ".venv/bin/python student_kit/analyze_results.py results.json",
        "```",
        "",
        "完整逐样本输出、reward 分项、环境版本和固定解码参数见 `results.json`。",
    ]
    return "\n".join(lines) + "\n"


def build_pdf(data, experiments, cases, name, student_id, output):
    from reportlab.lib import colors
    from reportlab.lib.enums import TA_CENTER
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
    from reportlab.lib.units import mm
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.ttfonts import TTFont
    from reportlab.platypus import Image, KeepTogether, PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

    font_path = "/System/Library/Fonts/Supplemental/Arial Unicode.ttf"
    pdfmetrics.registerFont(TTFont("CJK", font_path))
    styles = getSampleStyleSheet()
    for style in styles.byName.values():
        style.fontName = "CJK"
    styles.add(ParagraphStyle(name="ChineseTitle", parent=styles["Title"], fontName="CJK", fontSize=23, leading=32, alignment=TA_CENTER, textColor=colors.HexColor("#17324D")))
    styles.add(ParagraphStyle(name="ChineseHeading", parent=styles["Heading2"], fontName="CJK", fontSize=15, leading=22, spaceBefore=12, spaceAfter=7, textColor=colors.HexColor("#176B87")))
    styles.add(ParagraphStyle(name="ChineseBody", parent=styles["BodyText"], fontName="CJK", fontSize=9.5, leading=16, spaceAfter=6))
    styles.add(ParagraphStyle(name="SmallCJK", parent=styles["BodyText"], fontName="CJK", fontSize=7.3, leading=10))

    def footer(canvas, doc):
        canvas.saveState()
        canvas.setFont("CJK", 8)
        canvas.setFillColor(colors.HexColor("#607080"))
        canvas.drawString(20 * mm, 12 * mm, "Gemma 3 270M SVG LoRA - Part B")
        canvas.drawRightString(190 * mm, 12 * mm, str(doc.page))
        canvas.restoreState()

    doc = SimpleDocTemplate(str(output), pagesize=A4, leftMargin=18 * mm, rightMargin=18 * mm, topMargin=18 * mm, bottomMargin=20 * mm, title="Gemma 3 270M SVG LoRA")
    story = [Spacer(1, 28 * mm), Paragraph("Gemma 3 270M 基于 LoRA 的<br/>SVG 徽标生成", styles["ChineseTitle"]), Spacer(1, 18 * mm)]
    identity = [["姓名", name], ["学号", student_id], ["日期", date.today().isoformat()]]
    table = Table(identity, colWidths=[32 * mm, 80 * mm], hAlign="CENTER")
    table.setStyle(TableStyle([("FONTNAME", (0, 0), (-1, -1), "CJK"), ("FONTSIZE", (0, 0), (-1, -1), 10), ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#A8B6C2")), ("BACKGROUND", (0, 0), (0, -1), colors.HexColor("#EAF3F6")), ("PADDING", (0, 0), (-1, -1), 7)]))
    story += [table, PageBreak()]

    base, tuned, comp = data["summary"]["base"], data["summary"]["tuned"], data["comparison"]
    story += [Paragraph("摘要", styles["ChineseHeading"]), Paragraph(f"将 217 条可用记录固定分为 195 条训练和 22 条开发数据，对 Gemma 3 270M 进行 LoRA 监督微调；17 条最终验证样本完全隔离。总 reward 从 {fmt(base['total']['mean'])} 提高到 {fmt(tuned['total']['mean'])}，但人工渲染显示多数非致命输出仍退化为单色背景。因此结论是“格式稳定性明显改善，视觉语义生成仍未成功”。", styles["ChineseBody"])]
    story += [Paragraph("1. 数据、训练与评测设计", styles["ChineseHeading"]), Paragraph("原始 219 条中的两条 placeholder 目标相互冲突，仅在加载时过滤，源 JSONL 不修改。模型使用自带 chat template，system/user/padding token 的标签均为 -100，只有 assistant SVG token 计算损失。最终训练使用每个目标的前三个可见图元作为课程式派生目标，以降低 270M 模型的长序列难度。LoRA 仅插入 q/k/v/o 投影层；MPS 上使用 batch 1、梯度累积和梯度检查点。固定评测为 greedy、单 beam、最多 2048 个新 token。", styles["ChineseBody"])]
    story += [Paragraph("2. Reward 设计", styles["ChineseHeading"])]
    reward_rows = [["分项", "权重", "关键检查"], ["语法与安全", "30%", "单一可解析 SVG；无脚本、外链和事件"], ["几何", "20%", "viewBox、有限数值、主体坐标"], ["结构与样式", "15%", "图元、颜色、调色板、非空白"], ["提示词保真", "25%", "颜色、可验证图元词、基本构图"], ["反退化", "10%", "截断、异常长度、模板泄漏、重复"]]
    reward_table = Table(reward_rows, colWidths=[28 * mm, 16 * mm, 125 * mm], repeatRows=1)
    reward_table.setStyle(TableStyle([("FONTNAME", (0, 0), (-1, -1), "CJK"), ("FONTSIZE", (0, 0), (-1, -1), 8), ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#176B87")), ("TEXTCOLOR", (0, 0), (-1, 0), colors.white), ("GRID", (0, 0), (-1, -1), 0.35, colors.HexColor("#A8B6C2")), ("VALIGN", (0, 0), (-1, -1), "MIDDLE"), ("PADDING", (0, 0), (-1, -1), 4)]))
    story += [reward_table, Paragraph("Reward v2 要求正确 SVG namespace，并检查画布内前景、重复图元、巨型背景与折叠路径。背景/空白退化的 total 上限为 0.35；致命 XML/安全错误上限为 0.10。仍需渲染图进行 Goodhart 检查。", styles["ChineseBody"])]

    story += [Paragraph("3. 受控实验", styles["ChineseHeading"])]
    exp_rows = [["实验", "r", "lr", "长度", "样本", "epoch", "eval loss", "秒"]]
    for row in experiments:
        exp_rows.append([row["name"], row["r"], f"{row['lr']:.1e}", row["length"], row["samples"], row["epochs"], "-" if row["eval_loss"] is None else fmt(row["eval_loss"]), "-" if row["seconds"] is None else f"{row['seconds']:.0f}"])
    exp_table = Table(exp_rows, colWidths=[37 * mm, 10 * mm, 20 * mm, 17 * mm, 17 * mm, 16 * mm, 25 * mm, 16 * mm], repeatRows=1)
    exp_table.setStyle(TableStyle([("FONTNAME", (0, 0), (-1, -1), "CJK"), ("FONTSIZE", (0, 0), (-1, -1), 6.8), ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#17324D")), ("TEXTCOLOR", (0, 0), (-1, 0), colors.white), ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F3F7F8")]), ("GRID", (0, 0), (-1, -1), 0.3, colors.HexColor("#B6C0C8")), ("ALIGN", (1, 1), (-1, -1), "CENTER"), ("PADDING", (0, 0), (-1, -1), 3)]))
    story += [exp_table, Paragraph("E6/E7 从已有 checkpoint 独立重载评测，故不把重载耗时冒充训练耗时。完整目标虽有更低 dev loss，但生成经常截断；E10 的三图元课程在内部生成检查上 fatal rate 为 0，因此被冻结为最终 adapter。", styles["ChineseBody"])]

    story += [Paragraph("4. 最终验证结果", styles["ChineseHeading"])]
    result_rows = [["指标", "基座", "LoRA", "差值", "95% CI", "改善/平/退"]]
    for metric, label in (("total", "总分"), ("validity", "有效性"), ("fidelity", "保真度")):
        c = comp[metric]; ci = c["bootstrap_95_ci"]
        result_rows.append([label, fmt(base[metric]["mean"]), fmt(tuned[metric]["mean"]), fmt(c["paired_mean_delta"]), f"[{fmt(ci[0])}, {fmt(ci[1])}]", f"{c['improved']}/{c['unchanged']}/{c['worsened']}"])
    result_table = Table(result_rows, colWidths=[26 * mm, 24 * mm, 24 * mm, 24 * mm, 48 * mm, 25 * mm])
    result_table.setStyle(TableStyle([("FONTNAME", (0, 0), (-1, -1), "CJK"), ("FONTSIZE", (0, 0), (-1, -1), 8), ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#176B87")), ("TEXTCOLOR", (0, 0), (-1, 0), colors.white), ("GRID", (0, 0), (-1, -1), 0.35, colors.HexColor("#A8B6C2")), ("ALIGN", (1, 1), (-1, -1), "CENTER"), ("PADDING", (0, 0), (-1, -1), 4)]))
    story += [result_table, Paragraph(f"基座 fatal rate：{base['fatal_rate']:.1%}；LoRA fatal rate：{tuned['fatal_rate']:.1%}。验证集仅 17 条，因此同时报告配对 bootstrap 区间和逐例方向。", styles["ChineseBody"])]

    story += [Paragraph("5. 前后对比与 Goodhart 检查", styles["ChineseHeading"])]
    rendered = Path("output/rendered")
    for number, sample in enumerate(cases, 1):
        delta = sample["tuned"]["reward"]["total"] - sample["base"]["reward"]["total"]
        images = []
        labels = []
        for variant, label in (("reference", "参考目标"), ("base", "基座"), ("tuned", "LoRA")):
            path = rendered / f"sample_{sample['id']:02d}_{variant}.png"
            if path.exists():
                images.append(Image(str(path), width=48 * mm, height=48 * mm))
                labels.append(Paragraph(label, styles["SmallCJK"]))
        if len(images) == 3:
            grid = Table([images, labels], colWidths=[55 * mm] * 3)
            grid.setStyle(TableStyle([("ALIGN", (0, 0), (-1, -1), "CENTER"), ("VALIGN", (0, 0), (-1, -1), "MIDDLE"), ("GRID", (0, 0), (-1, 0), 0.35, colors.HexColor("#C7D0D6")), ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#FAFAFA"))]))
            caption = Paragraph(f"案例 {number}（样本 {sample['id']}）：base={fmt(sample['base']['reward']['total'])}，LoRA={fmt(sample['tuned']['reward']['total'])}，Δ={fmt(delta)}。{case_note(sample)}", styles["ChineseBody"])
            story.extend([KeepTogether([grid]), caption, Spacer(1, 3 * mm)])
    story += [Paragraph("程序化分数仍不会理解叶片、动物或抽象品牌含义。图中的 INVALID SVG 卡片表示安全检查失败，不是伪造的渲染图。", styles["ChineseBody"])]
    story += [Paragraph("6. 结论与局限", styles["ChineseHeading"]), Paragraph(f"LoRA 将 fatal rate 从 {base['fatal_rate']:.1%} 降至 {tuned['fatal_rate']:.1%}，显示 270M 模型学会了正确 namespace 和完整闭合。但 14 个非致命输出中有 13 个触发背景/画布退化上限，因此不能宣称已解决视觉语义生成。更合理的后续方向是更强模型、逐步增加图元的课程、结构化 SVG tokenization，以及栅格/视觉模型评测。", styles["ChineseBody"])]
    story += [Paragraph("7. 复现与验收", styles["ChineseHeading"]), Paragraph(f"环境：{data['environment']['platform']}；PyTorch {data['environment']['torch']}；Transformers {data['environment']['transformers']}；PEFT {data['environment']['peft']}。固定解码 seed={data['decoding']['seed']}，EOS token IDs={data['decoding'].get('eos_token_ids', [])}。单元测试、adapter 独立加载、数据哈希和逐样本输出分别见 tests/、adapter/、data_audit.json 和 results.json。", styles["ChineseBody"])]
    doc.build(story, onFirstPage=footer, onLaterPages=footer)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", default="results.json")
    parser.add_argument("--runs", default="runs")
    parser.add_argument("--name", default="待填写")
    parser.add_argument("--student-id", default="待填写")
    args = parser.parse_args()
    data = json.loads(Path(args.results).read_text(encoding="utf-8"))
    experiments = load_experiments(Path(args.runs))
    cases = choose_cases(data["samples"])
    examples = Path("output/examples")
    examples.mkdir(parents=True, exist_ok=True)
    for sample in cases:
        for variant in ("reference", "base", "tuned"):
            filename = f"sample_{sample['id']:02d}_{variant}.png"
            shutil.copy2(Path("output/rendered") / filename, examples / filename)
    Path("report.md").write_text(build_markdown(data, experiments, args.name, args.student_id), encoding="utf-8")
    output = Path("output/pdf/partB_svg_lora_report.pdf")
    output.parent.mkdir(parents=True, exist_ok=True)
    build_pdf(data, experiments, cases, args.name, args.student_id, output)
    print(f"wrote report.md and {output}")


if __name__ == "__main__":
    main()
