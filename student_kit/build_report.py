#!/usr/bin/env python3
"""Build the final Chinese Markdown and PDF report from audited artifacts."""

from __future__ import annotations

import argparse
import json
from datetime import date
from pathlib import Path

import yaml


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def f(value, digits=4):
    return f"{value:.{digits}f}"


def experiment_rows():
    names = [
        ("E10 stage1", "artifacts/experiment_summaries/e10_stage1.json", "3 visible elements", "internal warm start"),
        ("E11", "artifacts/experiment_summaries/e11_fullsvg_lr5e5.json", "full SVG, lr=5e-5", "dev candidate"),
        ("E12", "artifacts/experiment_summaries/e12_fullsvg_lr1e4.json", "full SVG, lr=1e-4", "dev candidate"),
        ("E13", "artifacts/experiment_summaries/e13_full217_seed42.json", "full SVG, seed=42", "primary"),
        ("E14", "artifacts/experiment_summaries/e14_full217_seed123.json", "full SVG, seed=123", "robustness"),
    ]
    rows = []
    for name, path, target, role in names:
        data = read_json(path)
        peak_bytes = data.get("peak_memory", {}).get("mps_driver_allocated_bytes")
        peak = None if peak_bytes is None else peak_bytes / (1024 ** 3)
        rows.append({
            "name": name, "target": target, "role": role,
            "rows": data["train_rows"], "dev": data["dev_rows"],
            "lr": data["config"]["learning_rate"], "length": data["config"]["max_length"],
            "eval_loss": data.get("eval_metrics", {}).get("eval_loss"),
            "seconds": data["elapsed_seconds"], "peak_gib": peak,
            "truncated": data.get("sequence_lengths", {}).get("truncated_rows"),
        })
    return rows


def build_markdown(primary, secondary, stage1, selection, manual, experiments, name, student_id):
    p0, p1 = primary["summary"]["base"], primary["summary"]["tuned"]
    s1 = secondary["summary"]["tuned"]
    warm = stage1["summary"]["tuned"]
    comp = primary["comparison"]
    verify = primary["verification"]
    yaml_text = Path("train_config.yaml").read_text(encoding="utf-8").rstrip()
    lines = [
        "# Gemma 3 270M 完整 SVG LoRA 微调与审计",
        "", f"- 姓名：{name}", f"- 学号：{student_id}",
        "- 作业仓库：https://github.com/ao-wei/gemma3-270m-svg-lora-partb",
        "- 数据来源：https://github.com/roboticcam/logo-detailed-prompt",
        "", "## 摘要", "",
        "本项目从零补齐 student kit，并在 Apple M4 MPS 上用 PEFT LoRA 训练 Gemma 3 270M。原始 219 条训练记录中两条 placeholder 在加载时过滤，源文件不修改；17 条验证集不参与学习率、epoch、seed 或 checkpoint 选择。三图元模型仅作为预热阶段，最终主 adapter 从该权重出发，用全部 217 条原始完整 SVG、3584 token 上限、seed 42 再训练一轮；seed 123 仅用于稳健性复跑。",
        "",
        f"最终主模型把 fatal rate 从 {p0['fatal_rate']:.1%} 降到 {p1['fatal_rate']:.1%}，总 reward 从 {f(p0['total']['mean'])} 提高到 {f(p1['total']['mean'])}；但 quality pass rate 为 {p1['quality_pass_rate']:.1%}，人工审计也判定视觉有效输出为 0/17。因此可支持的结论仅是 SVG 外层格式更稳定，不能宣称模型完成了提示驱动的徽标生成。",
        "", "## 1. 数据边界与训练方法", "",
        "训练加载器使用模型自带 chat template；system、user 与 padding token 的标签全部为 -100，仅 assistant SVG token 参与交叉熵。完整 chat 最大长度为 3535，最终 max_length=3584，所有最终训练记录零截断。LoRA rank=4、alpha=16、dropout=0.05，仅作用于 q_proj/k_proj/v_proj/o_proj；batch size=1、梯度累积=8、梯度检查点开启。训练是监督微调，reward 只用于内部选择和分析，并非 RL。",
        "",
        "为了在 16GB MPS 上保留完整 SVG，训练使用精确的分块词表投影损失：按 token 块计算 lm_head 与交叉熵，再按有效 assistant token 总数归一化。数值检查与原始全 logits loss 的绝对差为 0。precision:auto 仅对明确 BF16 算子不兼容错误重启 FP32；OOM 不会伪装成兼容问题。",
        "", "## 2. Reward v3 与 pass 定义", "",
        "总分公式为 0.30×语法安全 + 0.20×几何 + 0.15×结构样式 + 0.25×提示保真 + 0.10×反退化。提示保真内部为颜色 45%、图形 25%、空间 20%、构图 10%；不存在可靠空间要求时对其他分项重新归一化。空间关系只在元素可唯一匹配时评分，歧义标记 unscorable。stroke-width 必须有限、非负且不超过 viewBox 短边 10%。",
        "",
        "- valid pass：非 fatal 且 validity ≥ 0.8。",
        "- quality pass：valid pass、total ≥ 0.5、fidelity ≥ 0.3，且无背景/空白退化。",
        "- 致命 XML/安全错误有总分上限；背景、画布外与重复退化进入 violations。",
        "", "## 3. 补充实验与内部选择", "",
        "| 实验 | 目标/seed | 训练/开发 | lr | 长度 | eval loss | 峰值 driver GiB | 秒 | 截断 |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in experiments:
        loss = "-" if row["eval_loss"] is None else f(row["eval_loss"])
        peak = "-" if row["peak_gib"] is None else f"{row['peak_gib']:.2f}"
        trunc = "-" if row["truncated"] is None else row["truncated"]
        lines.append(f"| {row['name']} | {row['target']} | {row['rows']}/{row['dev']} | {row['lr']:.1e} | {row['length']} | {loss} | {peak} | {row['seconds']:.1f} | {trunc} |")
    lines += [
        "",
        f"E11/E12 只在固定 22 条内部开发集比较。E12 fatal rate={selection['candidates']['E12']['fatal_rate']:.1%} 且 fidelity 未改善，触发预声明排除门槛；选择 E11 的 lr={selection['selected_learning_rate']:.1e}。E13/E14 均从 stage1 初始权重重新训练全部 217 条；seed 42 预先指定为主模型。",
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
        "", "## 5. 全量视觉审计与 Goodhart", "",
        f"全部 17 条均人工检查 reference/base/seed42/seed123。seed42 视觉有效 {manual['summary']['seed42_visually_valid']}/17、背景单图元 {manual['summary']['seed42_background_only']}/17；seed123 视觉有效 {manual['summary']['seed123_visually_valid']}/17、背景单图元 {manual['summary']['seed123_background_only']}/17。程序化 reward 对可解析但空白/画布外输出仍给出约 0.35，总分因语法分被抬高，属于明确 Goodhart 反例。逐样本判断见 `manual_review.json`。",
        "",
    ]
    for page in range(1, 5):
        lines += [f"![全部样本视觉审计 {page}](output/audit/contact_sheet_{page}.png)", ""]
    lines += [
        "## 6. 结论与局限", "",
        "完整 SVG 继续训练确实把 seed42 的 fatal rate 降至 0，并且 34 个独立复验输出全部一致；但模型学习到的是高度重复、坐标为负的模板，几乎没有前景落在画布内。fidelity 的配对置信区间跨 0，两个 seed 的 quality pass 都为 0。270M 容量、长 SVG token 序列与纯文本交叉熵共同限制了语义构图。后续应采用更强基座、语法约束解码、结构化 SVG 表示和基于栅格图像的感知损失/评测。",
        "", "## 7. 复现与验收", "",
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
    from reportlab.platypus import Image, PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle, XPreformatted

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
        canvas.saveState(); canvas.setFont("CJK", 8); canvas.setFillColor(colors.HexColor("#607080"))
        canvas.drawString(18*mm, 11*mm, "Gemma 3 270M 完整 SVG LoRA 实验报告")
        canvas.drawRightString(192*mm, 11*mm, str(doc.page)); canvas.restoreState()

    def table(rows, widths, size=7.3):
        t = Table(rows, colWidths=widths, repeatRows=1)
        t.setStyle(TableStyle([("FONTNAME", (0,0), (-1,-1), "CJK"), ("FONTSIZE", (0,0), (-1,-1), size), ("BACKGROUND", (0,0), (-1,0), colors.HexColor("#176B87")), ("TEXTCOLOR", (0,0), (-1,0), colors.white), ("ROWBACKGROUNDS", (0,1), (-1,-1), [colors.white, colors.HexColor("#F3F7F8")]), ("GRID", (0,0), (-1,-1), .3, colors.HexColor("#A8B6C2")), ("VALIGN", (0,0), (-1,-1), "MIDDLE"), ("ALIGN", (1,1), (-1,-1), "CENTER"), ("PADDING", (0,0), (-1,-1), 3)]))
        return t

    doc = SimpleDocTemplate(str(output), pagesize=A4, leftMargin=17*mm, rightMargin=17*mm, topMargin=16*mm, bottomMargin=18*mm, title="Gemma 3 270M 完整 SVG LoRA")
    story = [Spacer(1, 30*mm), Paragraph("Gemma 3 270M 完整 SVG<br/>LoRA 微调与审计", styles["TitleCJK"]), Spacer(1, 16*mm), table([["姓名", name], ["学号", student_id], ["日期", date.today().isoformat()]], [30*mm, 85*mm], 10), PageBreak()]
    p0, p1 = primary["summary"]["base"], primary["summary"]["tuned"]
    s1, warm, comp = secondary["summary"]["tuned"], stage1["summary"]["tuned"], primary["comparison"]
    story += [Paragraph("摘要", styles["H2CJK"]), Paragraph(f"三图元模型仅作预热阶段；最终 seed42 adapter 从该权重出发，对全部 217 条原始完整 SVG 继续训练，最大序列 3535 token，零截断。fatal rate 从 {p0['fatal_rate']:.1%} 降到 {p1['fatal_rate']:.1%}，但 quality pass 为 {p1['quality_pass_rate']:.1%}，人工视觉有效为 0/17。结论仅限于格式稳定性改善。", styles["BodyCJK"]),
              Paragraph("1. 数据、训练与评测边界", styles["H2CJK"]), Paragraph("原始 train.jsonl 219 条，加载时过滤两条冲突 placeholder，原文件不修改。E11/E12 只用固定 195/22 内部划分选学习率；17 条最终验证集不参与学习率、epoch、seed 或 checkpoint 选择。chat template 来自本地模型，只对 assistant SVG token 计算交叉熵。LoRA r=4、alpha=16，目标 q/k/v/o。MPS 上 BF16、batch 1、累积 8，通过精确分块 lm_head 损失避免物化全长度词表 logits。", styles["BodyCJK"]),
              Paragraph("2. Reward v3 与 pass 定义", styles["H2CJK"])]
    reward_rows = [["分项", "权重", "检查"], ["语法/安全", "30%", "单一可解析 SVG；无脚本、外链"], ["几何", "20%", "viewBox、有限数、stroke、画布内前景"], ["结构/样式", "15%", "可见图元、颜色、非空白"], ["提示保真", "25%", "颜色45%、图形25%、空间20%、构图10%"], ["反退化", "10%", "截断、重复、模板泄漏、异常长度"]]
    story += [table(reward_rows, [30*mm, 17*mm, 125*mm], 8), Paragraph("valid pass = 非 fatal 且 validity≥0.8。quality pass = valid、total≥0.5、fidelity≥0.3，且无背景/空白退化。空间关系仅在图形/颜色可唯一匹配时评分。", styles["BodyCJK"]), Paragraph("3. E10–E14 实验", styles["H2CJK"])]
    exp_rows = [["实验", "目标", "训/开发", "lr", "len", "eval loss", "峰值GiB", "秒"]]
    for r in experiments: exp_rows.append([r["name"], r["target"], f"{r['rows']}/{r['dev']}", f"{r['lr']:.1e}", r["length"], "-" if r["eval_loss"] is None else f(r["eval_loss"]), "-" if r["peak_gib"] is None else f"{r['peak_gib']:.2f}", f"{r['seconds']:.0f}"])
    story += [table(exp_rows, [21*mm, 42*mm, 20*mm, 18*mm, 15*mm, 23*mm, 20*mm, 15*mm], 6.8), Paragraph(f"E12 fatal rate={selection['candidates']['E12']['fatal_rate']:.1%} 且 fidelity 未改善，按预注册门槛排除；选择 E11 lr={selection['selected_learning_rate']:.1e}。E13/E14 从同一 stage1 权重分别训练全部 217 条。", styles["BodyCJK"]), Paragraph("4. 最终定量结果", styles["H2CJK"])]
    result_rows = [["模型", "total", "validity", "fidelity", "fatal", "valid pass", "quality pass"]]
    for label, x in (("Base", p0), ("Stage1", warm), ("E13 seed42", p1), ("E14 seed123", s1)):
        result_rows.append([label, f(x["total"]["mean"]), f(x["validity"]["mean"]), f(x["fidelity"]["mean"]), f"{x['fatal_rate']:.1%}", f"{x['valid_pass_rate']:.1%}", f"{x['quality_pass_rate']:.1%}"])
    story += [table(result_rows, [31*mm, 23*mm, 23*mm, 23*mm, 21*mm, 25*mm, 27*mm], 7.2)]
    ci = comp["fidelity"]["bootstrap_95_ci"]
    story += [Paragraph(f"seed42 total 均值差={f(comp['total']['paired_mean_delta'])}，95% CI=[{f(comp['total']['bootstrap_95_ci'][0])}, {f(comp['total']['bootstrap_95_ci'][1])}]。fidelity 均值差={f(comp['fidelity']['paired_mean_delta'])}，95% CI=[{f(ci[0])}, {f(ci[1])}]，跨 0；因此不能宣称提示保真度显著改善。", styles["BodyCJK"]), PageBreak()]

    for page in range(1, 5):
        story += [Paragraph(f"5. 全量视觉审计（{page}/4）", styles["H2CJK"]), Image(f"output/audit/contact_sheet_{page}.png", width=174*mm, height=(174*mm)*(1316/920 if page < 4 else 580/920)), PageBreak()]
    story += [Paragraph("6. Goodhart 分析、结论与局限", styles["H2CJK"]), Paragraph(f"人工审计：seed42 视觉有效 {manual['summary']['seed42_visually_valid']}/17，背景单图元 {manual['summary']['seed42_background_only']}/17；seed123 视觉有效 {manual['summary']['seed123_visually_valid']}/17，背景单图元 {manual['summary']['seed123_background_only']}/17。许多输出 XML 可解析但图元均在负坐标或仅有巨型圆，Reward 仍因语法分给出约 0.35，是明确的代理指标高估。两个 seed 的 quality pass 均为 0，所以本项目只证明格式稳定性改善，没有解决视觉语义生成。", styles["BodyCJK"]), Paragraph("270M 容量、长 SVG token 序列和纯文本损失是主要限制。后续应考虑更强基座、语法约束解码、结构化 SVG 表示，以及栅格/视觉模型感知损失。", styles["BodyCJK"]), Paragraph("7. 复现与验收", styles["H2CJK"])]
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
    story += [XPreformatted(commands, styles["CodeCJK"]), Paragraph("身份信息已校验：仅包含姓名与学号，无其他身份字段或占位符。", styles["BodyCJK"])]
    doc.build(story, onFirstPage=footer, onLaterPages=footer)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", default="敖炜")
    parser.add_argument("--student-id", default="202521080810")
    args = parser.parse_args()
    primary, secondary, stage1 = read_json("results.json"), read_json("results_seed123.json"), read_json("results_stage1.json")
    selection, manual, experiments = read_json("artifacts/model_selection.json"), read_json("manual_review.json"), experiment_rows()
    Path("report.md").write_text(build_markdown(primary, secondary, stage1, selection, manual, experiments, args.name, args.student_id), encoding="utf-8")
    output = Path("output/pdf/partB_svg_lora_report.pdf"); output.parent.mkdir(parents=True, exist_ok=True)
    build_pdf(primary, secondary, stage1, selection, manual, experiments, args.name, args.student_id, output)
    print(f"wrote report.md and {output}")


if __name__ == "__main__": main()
