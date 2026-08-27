# src/compare.py
"""
Unified cross-model comparison and reporting module for PlantVillage disease classifier.

Usage:
    python -m src.compare
    python -m src.compare --include-efficientnet
    python -m src.compare --output-dir reports/comparison
"""

from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import os
from typing import Any, Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# ----------------------------------------------------------------------
# Model Registry Definition
# ----------------------------------------------------------------------
MODEL_REGISTRY = {
    "mlp": {
        "display_name": "MLP",
        "family": "Dense / Non-Spatial",
        "training_approach": "From scratch",
        "pretrained": False,
        "is_official": True,
        "param_count": 77238799,
        "result_dir": "reports/mlp",
    },
    "custom_cnn": {
        "display_name": "Custom CNN",
        "family": "Convolutional Neural Network",
        "training_approach": "From scratch",
        "pretrained": False,
        "is_official": True,
        "param_count": 392751,
        "result_dir": "reports/cnn_custom",
    },
    "resnet18": {
        "display_name": "ResNet-18",
        "family": "Residual Convolutional Neural Network",
        "training_approach": "ImageNet transfer learning",
        "pretrained": True,
        "is_official": True,
        "param_count": 11184207,
        "result_dir": "reports_archive/reports_20260126_143813_resnet18",
    },
    "vit_b16": {
        "display_name": "ViT-B/16",
        "family": "Vision Transformer / Self-Attention",
        "training_approach": "ImageNet transfer learning",
        "pretrained": True,
        "is_official": True,
        "param_count": 85810191,
        "result_dir": "reports/vit_b16",
    },
    "efficientnet_b0": {
        "display_name": "EfficientNet-B0",
        "family": "Compound-Scaled Convolutional Neural Network",
        "training_approach": "ImageNet transfer learning",
        "pretrained": True,
        "is_official": False,
        "param_count": 4026799,
        "result_dir": "reports_archive/reports_20260126_151938_efficientnet",
    },
}

MODEL_COLORS = {
    "MLP": "#e74c3c",             # Red
    "Custom CNN": "#f39c12",      # Orange/Amber
    "ResNet-18": "#2980b9",       # Blue
    "ViT-B/16": "#8e44ad",        # Purple
    "EfficientNet-B0": "#27ae60", # Green
}


def compute_sha256(filepath: str) -> str:
    """Compute sha256 checksum of a file."""
    if not os.path.exists(filepath):
        return "FILE_NOT_FOUND"
    sha = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(8192):
            sha.update(chunk)
    return sha.hexdigest()


def normalize_metrics(model_key: str, info: Dict[str, Any]) -> Tuple[Dict[str, Any], List[Dict[str, Any]], Dict[str, Any]]:
    """
    Extract and normalize metrics from model results directory.
    Returns:
      - metrics_dict: dict of normalized metric values
      - sources_list: list of metric source lineage records
      - validation_info: dict of validation checks
    """
    rdir = info["result_dir"]
    metrics: Dict[str, Any] = {
        "model_key": model_key,
        "display_name": info["display_name"],
        "family": info["family"],
        "training_approach": info["training_approach"],
        "pretrained": info["pretrained"],
        "is_official": info["is_official"],
        "param_count": info.get("param_count", None),
        "val_accuracy": None,
        "val_macro_f1": None,
        "test_loss": None,
        "test_accuracy": None,
        "test_macro_precision": None,
        "test_macro_recall": None,
        "test_macro_f1": None,
        "test_top3_accuracy": None,
        "test_sample_count": None,
        "training_time_seconds": None,
        "execution_device": None,
        "best_epoch": None,
        "best_phase": None,
        "result_dir": rdir,
        "status": "AVAILABLE" if os.path.exists(rdir) else "NOT_RECORDED",
    }

    sources: List[Dict[str, Any]] = []
    val_info: Dict[str, Any] = {
        "result_dir": rdir,
        "exists": os.path.exists(rdir),
        "files_found": [],
        "missing_files": [],
        "confusion_matrix_shape": None,
        "confusion_matrix_total": None,
        "per_class_count": None,
        "warnings": [],
    }

    if not os.path.exists(rdir):
        val_info["warnings"].append(f"Result directory '{rdir}' does not exist.")
        return metrics, sources, val_info

    # 1. Inspect results.json (Validation results)
    results_path = os.path.join(rdir, "results.json")
    if os.path.exists(results_path):
        val_info["files_found"].append("results.json")
        try:
            with open(results_path, "r", encoding="utf-8") as f:
                res = json.load(f)
            metrics["execution_device"] = res.get("device", "Not recorded")
            sources.append({"model": info["display_name"], "normalized_metric": "execution_device", "value": metrics["execution_device"], "source_file": "results.json", "original_key": "device"})

            best_row = res.get("best_row", {})
            if best_row:
                metrics["best_epoch"] = best_row.get("epoch", None)
                metrics["best_phase"] = best_row.get("phase", None)
                val_summary = best_row.get("val_summary", {})
                metrics["val_accuracy"] = val_summary.get("accuracy", None)
                metrics["val_macro_f1"] = val_summary.get("macro_f1", res.get("best_score", None))

                sources.append({"model": info["display_name"], "normalized_metric": "val_accuracy", "value": metrics["val_accuracy"], "source_file": "results.json", "original_key": "best_row.val_summary.accuracy"})
                sources.append({"model": info["display_name"], "normalized_metric": "val_macro_f1", "value": metrics["val_macro_f1"], "source_file": "results.json", "original_key": "best_row.val_summary.macro_f1"})
                sources.append({"model": info["display_name"], "normalized_metric": "best_epoch", "value": metrics["best_epoch"], "source_file": "results.json", "original_key": "best_row.epoch"})
        except Exception as e:
            val_info["warnings"].append(f"Error parsing results.json: {e}")
    else:
        val_info["missing_files"].append("results.json")

    # 2. Inspect history.json (Training time)
    history_path = os.path.join(rdir, "history.json")
    if os.path.exists(history_path):
        val_info["files_found"].append("history.json")
        try:
            with open(history_path, "r", encoding="utf-8") as f:
                hist = json.load(f)
            if isinstance(hist, list):
                total_sec = sum(r.get("seconds", 0.0) for r in hist if isinstance(r, dict))
                metrics["training_time_seconds"] = total_sec
                sources.append({"model": info["display_name"], "normalized_metric": "training_time_seconds", "value": total_sec, "source_file": "history.json", "original_key": "sum(seconds)"})
        except Exception as e:
            val_info["warnings"].append(f"Error parsing history.json: {e}")
    else:
        val_info["missing_files"].append("history.json")

    # 3. Inspect test_results.json (Official Test Evaluation)
    test_path = os.path.join(rdir, "test_results.json")
    if os.path.exists(test_path):
        val_info["files_found"].append("test_results.json")
        try:
            with open(test_path, "r", encoding="utf-8") as f:
                tres = json.load(f)
            metrics["test_loss"] = tres.get("test_loss", None)
            metrics["test_top3_accuracy"] = tres.get("topk_accuracy", tres.get("top3_accuracy", None))

            sources.append({"model": info["display_name"], "normalized_metric": "test_loss", "value": metrics["test_loss"], "source_file": "test_results.json", "original_key": "test_loss"})
            sources.append({"model": info["display_name"], "normalized_metric": "test_top3_accuracy", "value": metrics["test_top3_accuracy"], "source_file": "test_results.json", "original_key": "topk_accuracy"})

            summary = tres.get("summary", {})
            metrics["test_accuracy"] = summary.get("accuracy", None)
            metrics["test_macro_precision"] = summary.get("macro_precision", None)
            metrics["test_macro_recall"] = summary.get("macro_recall", None)
            metrics["test_macro_f1"] = summary.get("macro_f1", None)

            sources.append({"model": info["display_name"], "normalized_metric": "test_accuracy", "value": metrics["test_accuracy"], "source_file": "test_results.json", "original_key": "summary.accuracy"})
            sources.append({"model": info["display_name"], "normalized_metric": "test_macro_precision", "value": metrics["test_macro_precision"], "source_file": "test_results.json", "original_key": "summary.macro_precision"})
            sources.append({"model": info["display_name"], "normalized_metric": "test_macro_recall", "value": metrics["test_macro_recall"], "source_file": "test_results.json", "original_key": "summary.macro_recall"})
            sources.append({"model": info["display_name"], "normalized_metric": "test_macro_f1", "value": metrics["test_macro_f1"], "source_file": "test_results.json", "original_key": "summary.macro_f1"})
        except Exception as e:
            val_info["warnings"].append(f"Error parsing test_results.json: {e}")
    else:
        val_info["missing_files"].append("test_results.json")

    # 4. Inspect test_confusion_matrix.npy
    cm_path = os.path.join(rdir, "test_confusion_matrix.npy")
    if os.path.exists(cm_path):
        val_info["files_found"].append("test_confusion_matrix.npy")
        try:
            cm = np.load(cm_path)
            val_info["confusion_matrix_shape"] = list(cm.shape)
            val_info["confusion_matrix_total"] = int(np.sum(cm))
            metrics["test_sample_count"] = int(np.sum(cm))
            sources.append({"model": info["display_name"], "normalized_metric": "test_sample_count", "value": metrics["test_sample_count"], "source_file": "test_confusion_matrix.npy", "original_key": "np.sum(cm)"})
        except Exception as e:
            val_info["warnings"].append(f"Error loading confusion matrix: {e}")
    else:
        val_info["missing_files"].append("test_confusion_matrix.npy")

    # 5. Inspect test_per_class.csv
    per_class_path = os.path.join(rdir, "test_per_class.csv")
    if os.path.exists(per_class_path):
        val_info["files_found"].append("test_per_class.csv")
        try:
            df_pc = pd.read_csv(per_class_path)
            val_info["per_class_count"] = len(df_pc)
        except Exception as e:
            val_info["warnings"].append(f"Error reading per_class csv: {e}")
    else:
        val_info["missing_files"].append("test_per_class.csv")

    return metrics, sources, val_info


def format_pct(val: Optional[float], decimals: int = 2) -> str:
    """Format float as percentage string."""
    if val is None or pd.isna(val):
        return "Not recorded"
    return f"{val * 100:.{decimals}f}%"


def format_float(val: Optional[float], decimals: int = 4) -> str:
    """Format float."""
    if val is None or pd.isna(val):
        return "Not recorded"
    return f"{val:.{decimals}f}"


def format_int(val: Optional[int]) -> str:
    """Format int."""
    if val is None or pd.isna(val):
        return "Not recorded"
    return f"{val:,}"


def generate_tables(metrics_list: List[Dict[str, Any]], output_dir: str) -> None:
    """Generate CSV, Markdown, and LaTeX comparison tables."""
    # 1. Comprehensive CSV
    df = pd.DataFrame(metrics_list)
    csv_path = os.path.join(output_dir, "metrics_table.csv")
    df.to_csv(csv_path, index=False)

    # 2. Markdown Table
    md_rows = []
    md_rows.append("# Cross-Model Performance Comparison\n")
    md_rows.append("## Official Four-Model Comparison\n")
    md_rows.append("| Model | Architecture Family | Training Approach | Pretrained | Test Accuracy | Macro Precision | Macro Recall | Macro-F1 | Top-3 Accuracy | Parameters |")
    md_rows.append("| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |")

    official_models = [m for m in metrics_list if m["is_official"]]
    for m in official_models:
        pre_str = "Yes" if m["pretrained"] else "No"
        md_rows.append(
            f"| **{m['display_name']}** | {m['family']} | {m['training_approach']} | {pre_str} | "
            f"{format_pct(m['test_accuracy'])} | {format_pct(m['test_macro_precision'])} | "
            f"{format_pct(m['test_macro_recall'])} | {format_pct(m['test_macro_f1'])} | "
            f"{format_pct(m['test_top3_accuracy'])} | {format_int(m['param_count'])} |"
        )

    ref_models = [m for m in metrics_list if not m["is_official"]]
    if ref_models:
        md_rows.append("\n## Additional Reference Benchmark\n")
        md_rows.append("| Model | Architecture Family | Training Approach | Pretrained | Test Accuracy | Macro Precision | Macro Recall | Macro-F1 | Top-3 Accuracy | Parameters |")
        md_rows.append("| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |")
        for m in ref_models:
            pre_str = "Yes" if m["pretrained"] else "No"
            md_rows.append(
                f"| **{m['display_name']}** | {m['family']} | {m['training_approach']} | {pre_str} | "
                f"{format_pct(m['test_accuracy'])} | {format_pct(m['test_macro_precision'])} | "
                f"{format_pct(m['test_macro_recall'])} | {format_pct(m['test_macro_f1'])} | "
                f"{format_pct(m['test_top3_accuracy'])} | {format_int(m['param_count'])} |"
            )

    md_path = os.path.join(output_dir, "metrics_table.md")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("\n".join(md_rows) + "\n")

    # 3. LaTeX Table
    tex_rows = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\small",
        r"\caption{Comparative Performance of Deep Learning Architectures on PlantVillage Test Set (3,096 Images)}",
        r"\label{tab:model_comparison}",
        r"\begin{tabular}{llccccccc}",
        r"\hline",
        r"\textbf{Model} & \textbf{Family} & \textbf{Pretrained} & \textbf{Accuracy} & \textbf{Macro-P} & \textbf{Macro-R} & \textbf{Macro-F1} & \textbf{Top-3 Acc} & \textbf{Params} \\",
        r"\hline",
        r"\multicolumn{9}{l}{\textit{\textbf{Official Four-Model Comparison}}} \\",
    ]

    for m in official_models:
        name_tex = m["display_name"].replace("_", r"\_")
        family_tex = m["family"].replace("/", r"/").replace("&", r"\&")
        pre_tex = "Yes" if m["pretrained"] else "No"
        acc_tex = format_pct(m["test_accuracy"]).replace("%", r"\%")
        p_tex = format_pct(m["test_macro_precision"]).replace("%", r"\%")
        r_tex = format_pct(m["test_macro_recall"]).replace("%", r"\%")
        f1_tex = format_pct(m["test_macro_f1"]).replace("%", r"\%")
        top3_tex = format_pct(m["test_top3_accuracy"]).replace("%", r"\%")
        params_tex = format_int(m["param_count"])

        tex_rows.append(
            f"{name_tex} & {family_tex} & {pre_tex} & {acc_tex} & {p_tex} & {r_tex} & {f1_tex} & {top3_tex} & {params_tex} \\\\"
        )

    if ref_models:
        tex_rows.append(r"\hline")
        tex_rows.append(r"\multicolumn{9}{l}{\textit{\textbf{Additional Reference Benchmark}}} \\")
        for m in ref_models:
            name_tex = m["display_name"].replace("_", r"\_")
            family_tex = m["family"].replace("/", r"/").replace("&", r"\&")
            pre_tex = "Yes" if m["pretrained"] else "No"
            acc_tex = format_pct(m["test_accuracy"]).replace("%", r"\%")
            p_tex = format_pct(m["test_macro_precision"]).replace("%", r"\%")
            r_tex = format_pct(m["test_macro_recall"]).replace("%", r"\%")
            f1_tex = format_pct(m["test_macro_f1"]).replace("%", r"\%")
            top3_tex = format_pct(m["test_top3_accuracy"]).replace("%", r"\%")
            params_tex = format_int(m["param_count"])

            tex_rows.append(
                f"{name_tex} & {family_tex} & {pre_tex} & {acc_tex} & {p_tex} & {r_tex} & {f1_tex} & {top3_tex} & {params_tex} \\\\"
            )

    tex_rows.extend([
        r"\hline",
        r"\end{tabular}",
        r"\end{table*}",
    ])

    tex_path = os.path.join(output_dir, "metrics_table.tex")
    with open(tex_path, "w", encoding="utf-8") as f:
        f.write("\n".join(tex_rows) + "\n")


def generate_visualizations(metrics_list: List[Dict[str, Any]], output_dir: str) -> None:
    """Generate all required comparison charts and plots."""
    # Filter models with recorded test metrics
    valid_models = [m for m in metrics_list if m["test_accuracy"] is not None]
    if not valid_models:
        print("No models with test results found for visualization.")
        return

    names = [m["display_name"] for m in valid_models]
    colors = [MODEL_COLORS.get(m["display_name"], "#34495e") for m in valid_models]

    # 1. Accuracy & Macro-F1 Grouped Bar Chart
    fig, ax = plt.subplots(figsize=(9, 5), dpi=300)
    x = np.arange(len(names))
    width = 0.35

    accs = [m["test_accuracy"] * 100 for m in valid_models]
    f1s = [m["test_macro_f1"] * 100 for m in valid_models]

    rects1 = ax.bar(x - width/2, accs, width, label="Test Accuracy (%)", color="#2980b9", edgecolor="black", linewidth=0.8)
    rects2 = ax.bar(x + width/2, f1s, width, label="Test Macro-F1 (%)", color="#27ae60", edgecolor="black", linewidth=0.8)

    ax.set_ylabel("Score (%)", fontsize=11, fontweight="bold")
    ax.set_title("PlantVillage Disease Classifier: Test Accuracy vs Macro-F1", fontsize=12, fontweight="bold", pad=12)
    ax.set_xticks(x)
    ax.set_xticklabels(names, fontsize=10, fontweight="bold")
    ax.set_ylim(0, 105)
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    ax.legend(frameon=True, loc="lower right")

    for rect in rects1:
        h = rect.get_height()
        ax.annotate(f"{h:.2f}%", xy=(rect.get_x() + rect.get_width() / 2, h), xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=8, fontweight="bold")
    for rect in rects2:
        h = rect.get_height()
        ax.annotate(f"{h:.2f}%", xy=(rect.get_x() + rect.get_width() / 2, h), xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=8, fontweight="bold")

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "accuracy_f1_comparison.png"))
    plt.close(fig)

    # 2. Precision & Recall Grouped Bar Chart
    fig, ax = plt.subplots(figsize=(9, 5), dpi=300)
    precs = [m["test_macro_precision"] * 100 for m in valid_models]
    recs = [m["test_macro_recall"] * 100 for m in valid_models]

    rects1 = ax.bar(x - width/2, precs, width, label="Macro Precision (%)", color="#8e44ad", edgecolor="black", linewidth=0.8)
    rects2 = ax.bar(x + width/2, recs, width, label="Macro Recall (%)", color="#e67e22", edgecolor="black", linewidth=0.8)

    ax.set_ylabel("Score (%)", fontsize=11, fontweight="bold")
    ax.set_title("PlantVillage Disease Classifier: Macro Precision vs Macro Recall", fontsize=12, fontweight="bold", pad=12)
    ax.set_xticks(x)
    ax.set_xticklabels(names, fontsize=10, fontweight="bold")
    ax.set_ylim(0, 105)
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    ax.legend(frameon=True, loc="lower right")

    for rect in rects1:
        h = rect.get_height()
        ax.annotate(f"{h:.2f}%", xy=(rect.get_x() + rect.get_width() / 2, h), xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=8, fontweight="bold")
    for rect in rects2:
        h = rect.get_height()
        ax.annotate(f"{h:.2f}%", xy=(rect.get_x() + rect.get_width() / 2, h), xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=8, fontweight="bold")

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "precision_recall_comparison.png"))
    plt.close(fig)

    # 3. Top-3 Accuracy Bar Chart
    fig, ax = plt.subplots(figsize=(8, 4.5), dpi=300)
    top3s = [m["test_top3_accuracy"] * 100 for m in valid_models]
    bars = ax.bar(names, top3s, color=colors, edgecolor="black", linewidth=0.8, width=0.5)

    ax.set_ylabel("Top-3 Accuracy (%)", fontsize=11, fontweight="bold")
    ax.set_title("PlantVillage Disease Classifier: Top-3 Accuracy Across Models", fontsize=12, fontweight="bold", pad=12)
    ax.set_ylim(0, 110)
    ax.grid(axis="y", linestyle="--", alpha=0.5)

    for bar in bars:
        h = bar.get_height()
        ax.annotate(f"{h:.2f}%", xy=(bar.get_x() + bar.get_width() / 2, h), xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=9, fontweight="bold")

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "top3_accuracy_comparison.png"))
    plt.close(fig)

    # 4. All Metrics Grouped Comparison
    fig, ax = plt.subplots(figsize=(12, 6), dpi=300)
    n_metrics = 5
    bar_w = 0.15
    metric_labels = ["Accuracy", "Macro Precision", "Macro Recall", "Macro-F1", "Top-3 Accuracy"]
    palette = ["#3498db", "#9b59b6", "#e67e22", "#2ecc71", "#1abc9c"]

    for i, m_label in enumerate(metric_labels):
        if i == 0:
            vals = [m["test_accuracy"] * 100 for m in valid_models]
        elif i == 1:
            vals = [m["test_macro_precision"] * 100 for m in valid_models]
        elif i == 2:
            vals = [m["test_macro_recall"] * 100 for m in valid_models]
        elif i == 3:
            vals = [m["test_macro_f1"] * 100 for m in valid_models]
        else:
            vals = [m["test_top3_accuracy"] * 100 for m in valid_models]

        offsets = (i - (n_metrics - 1) / 2) * bar_w
        rects = ax.bar(x + offsets, vals, bar_w, label=m_label, color=palette[i], edgecolor="black", linewidth=0.6)

    ax.set_ylabel("Score (%)", fontsize=11, fontweight="bold")
    ax.set_title("Unified Multi-Metric Model Comparison", fontsize=13, fontweight="bold", pad=12)
    ax.set_xticks(x)
    ax.set_xticklabels(names, fontsize=11, fontweight="bold")
    ax.set_ylim(0, 110)
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    ax.legend(frameon=True, loc="lower right", ncol=3)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "all_metrics_comparison.png"))
    plt.close(fig)

    # 5. Model Tradeoffs (Parameters vs Macro-F1)
    models_with_params = [m for m in valid_models if m.get("param_count") is not None]
    if len(models_with_params) >= 2:
        fig, ax = plt.subplots(figsize=(8.5, 5.5), dpi=300)
        for m in models_with_params:
            p_millions = m["param_count"] / 1e6
            f1_val = m["test_macro_f1"] * 100
            c = MODEL_COLORS.get(m["display_name"], "#34495e")
            ax.scatter(p_millions, f1_val, color=c, s=140, edgecolors="black", linewidth=1.2, zorder=5)
            offset_y = 1.0 if m["display_name"] != "ResNet-18" else -1.5
            ax.annotate(
                f"{m['display_name']}\n({p_millions:.2f}M, {f1_val:.1f}%)",
                (p_millions, f1_val),
                textcoords="offset points",
                xytext=(0, offset_y * 10),
                ha="center",
                fontsize=9,
                fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.8, edgecolor="gray", linewidth=0.5)
            )

        ax.set_xlabel("Trainable Parameters (Millions)", fontsize=11, fontweight="bold")
        ax.set_ylabel("Test Macro-F1 (%)", fontsize=11, fontweight="bold")
        ax.set_title("Model Efficiency Trade-off: Parameter Count vs. Test Macro-F1", fontsize=12, fontweight="bold", pad=12)
        ax.grid(True, linestyle="--", alpha=0.5)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "model_tradeoffs.png"))
        plt.close(fig)

    # 6. Combined Normalized Confusion Matrices
    cm_dict = {}
    for m in valid_models:
        cm_file = os.path.join(m["result_dir"], "test_confusion_matrix.npy")
        if os.path.exists(cm_file):
            cm_dict[m["display_name"]] = np.load(cm_file)

    if len(cm_dict) >= 2:
        n_plots = len(cm_dict)
        cols = 2
        rows = (n_plots + 1) // 2
        fig, axes = plt.subplots(rows, cols, figsize=(14, 6 * rows), dpi=300)
        axes = np.array(axes).reshape(-1)

        for idx, (m_name, cm) in enumerate(cm_dict.items()):
            ax = axes[idx]
            # normalize by row
            row_sums = cm.sum(axis=1, keepdims=True)
            cm_norm = np.divide(cm.astype("float"), row_sums, out=np.zeros_like(cm, dtype=float), where=row_sums != 0)

            im = ax.imshow(cm_norm, interpolation="nearest", cmap="Blues", vmin=0, vmax=1)
            ax.set_title(f"{m_name} (Normalized CM)", fontsize=12, fontweight="bold")
            ax.set_xlabel("Predicted Class ID", fontsize=9)
            ax.set_ylabel("True Class ID", fontsize=9)
            ax.set_xticks(range(15))
            ax.set_yticks(range(15))
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

        # Hide any unused axes
        for j in range(idx + 1, len(axes)):
            axes[j].axis("off")

        plt.suptitle("Normalized Confusion Matrices Across Models (15 Classes)", fontsize=14, fontweight="bold", y=0.99)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "combined_normalized_confusion_matrices.png"))
        plt.close(fig)

    # 7. Optional Per-Class F1 Heatmap
    pc_dfs = {}
    for m in valid_models:
        pc_file = os.path.join(m["result_dir"], "test_per_class.csv")
        if os.path.exists(pc_file):
            pc_dfs[m["display_name"]] = pd.read_csv(pc_file)

    if len(pc_dfs) >= 2:
        # Align on class_name
        base_df = next(iter(pc_dfs.values()))
        class_names = base_df["class_name"].tolist() if "class_name" in base_df.columns else [f"Class_{i}" for i in range(15)]
        
        per_class_table = pd.DataFrame({"class_name": class_names})
        f1_matrix = []

        for m_name in names:
            if m_name in pc_dfs:
                df_pc = pc_dfs[m_name]
                # Ensure ordering aligns with class_names
                if "class_name" in df_pc.columns:
                    df_pc = df_pc.set_index("class_name").reindex(class_names).reset_index()
                per_class_table[f"{m_name}_f1"] = df_pc["f1-score"]
                per_class_table[f"{m_name}_precision"] = df_pc["precision"]
                per_class_table[f"{m_name}_recall"] = df_pc["recall"]
                f1_matrix.append(df_pc["f1-score"].values)
            else:
                per_class_table[f"{m_name}_f1"] = np.nan
                per_class_table[f"{m_name}_precision"] = np.nan
                per_class_table[f"{m_name}_recall"] = np.nan

        per_class_table.to_csv(os.path.join(output_dir, "per_class_comparison.csv"), index=False)

        if f1_matrix:
            f1_arr = np.array(f1_matrix).T  # Shape: (15, num_models)
            fig, ax = plt.subplots(figsize=(8 + len(names), 8), dpi=300)
            im = ax.imshow(f1_arr, cmap="YlGnBu", aspect="auto", vmin=0, vmax=1)

            ax.set_xticks(range(len(names)))
            ax.set_xticklabels(names, fontsize=10, fontweight="bold")
            ax.set_yticks(range(len(class_names)))
            ax.set_yticklabels(class_names, fontsize=9)
            ax.set_title("Per-Class F1-Score Comparison Heatmap", fontsize=12, fontweight="bold", pad=12)

            for r in range(f1_arr.shape[0]):
                for c in range(f1_arr.shape[1]):
                    val = f1_arr[r, c]
                    color = "white" if val > 0.6 else "black"
                    ax.text(c, r, f"{val:.2f}", ha="center", va="center", color=color, fontsize=8, fontweight="bold")

            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, "per_class_f1_heatmap.png"))
            plt.close(fig)


def generate_factual_observations(metrics_list: List[Dict[str, Any]], output_dir: str) -> None:
    """Generate reports/comparison/factual_observations.md with objective data findings."""
    valid_official = [m for m in metrics_list if m["is_official"] and m["test_accuracy"] is not None]
    ref_models = [m for m in metrics_list if not m["is_official"] and m["test_accuracy"] is not None]

    lines = [
        "# Automated Factual Observations\n",
        "This document contains strictly empirical observations computed from the saved official evaluation outputs.\n",
        "## 1. Official Model Performance Metrics\n",
    ]

    if valid_official:
        # Sort by test macro-F1 descending
        by_f1 = sorted(valid_official, key=lambda x: x["test_macro_f1"], reverse=True)
        by_acc = sorted(valid_official, key=lambda x: x["test_accuracy"], reverse=True)
        by_top3 = sorted(valid_official, key=lambda x: x["test_top3_accuracy"], reverse=True)

        top_f1_model = by_f1[0]
        top_acc_model = by_acc[0]
        top_top3_model = by_top3[0]

        lines.append(f"- **Highest Recorded Test Accuracy**: **{top_acc_model['display_name']}** at **{top_acc_model['test_accuracy']*100:.2f}%**.")
        lines.append(f"- **Highest Recorded Test Macro-F1**: **{top_f1_model['display_name']}** at **{top_f1_model['test_macro_f1']*100:.2f}%**.")
        lines.append(f"- **Highest Recorded Top-3 Accuracy**: **{top_top3_model['display_name']}** at **{top_top3_model['test_top3_accuracy']*100:.2f}%**.\n")

        lines.append("### Performance Rankings (Among Models with Recorded Results)\n")
        lines.append("1. **By Test Macro-F1**:")
        for idx, m in enumerate(by_f1, 1):
            lines.append(f"   {idx}. {m['display_name']}: {m['test_macro_f1']*100:.2f}%")
        lines.append("2. **By Test Accuracy**:")
        for idx, m in enumerate(by_acc, 1):
            lines.append(f"   {idx}. {m['display_name']}: {m['test_accuracy']*100:.2f}%")
        lines.append("3. **By Top-3 Accuracy**:")
        for idx, m in enumerate(by_top3, 1):
            lines.append(f"   {idx}. {m['display_name']}: {m['test_top3_accuracy']*100:.2f}%\n")

        lines.append("### Empirical Performance Gaps (Percentage-Point Differences)\n")
        # Pairwise differences
        resnet = next((m for m in valid_official if m["display_name"] == "ResNet-18"), None)
        mlp = next((m for m in valid_official if m["display_name"] == "MLP"), None)
        cnn = next((m for m in valid_official if m["display_name"] == "Custom CNN"), None)

        if resnet and mlp:
            acc_diff = (resnet["test_accuracy"] - mlp["test_accuracy"]) * 100
            f1_diff = (resnet["test_macro_f1"] - mlp["test_macro_f1"]) * 100
            lines.append(f"- **ResNet-18 vs. MLP**: ResNet-18 exceeded MLP by **+{acc_diff:.2f} percentage points** in Test Accuracy and **+{f1_diff:.2f} percentage points** in Macro-F1.")

        if resnet and cnn:
            acc_diff = (resnet["test_accuracy"] - cnn["test_accuracy"]) * 100
            f1_diff = (resnet["test_macro_f1"] - cnn["test_macro_f1"]) * 100
            lines.append(f"- **ResNet-18 vs. Custom CNN**: ResNet-18 exceeded Custom CNN by **+{acc_diff:.2f} percentage points** in Test Accuracy and **+{f1_diff:.2f} percentage points** in Macro-F1.")

        if mlp and cnn:
            acc_diff = (mlp["test_accuracy"] - cnn["test_accuracy"]) * 100
            f1_diff = (mlp["test_macro_f1"] - cnn["test_macro_f1"]) * 100
            lines.append(f"- **MLP vs. Custom CNN**: MLP exceeded Custom CNN by **+{acc_diff:.2f} percentage points** in Test Accuracy and **+{f1_diff:.2f} percentage points** in Macro-F1.")

    vit = next((m for m in metrics_list if m["display_name"] == "ViT-B/16"), None)
    if vit and vit["test_accuracy"] is None:
        lines.append("\n### ViT-B/16 Status Note\n")
        lines.append("- Official full-dataset ViT-B/16 training was aborted prior to execution due to CPU-only PyTorch build in the active environment. Official metrics are marked as *Not recorded*.")

    if ref_models:
        lines.append("\n## 2. Additional Reference Benchmark (EfficientNet-B0)\n")
        eff = ref_models[0]
        lines.append(f"- **EfficientNet-B0 Test Accuracy**: {eff['test_accuracy']*100:.2f}%")
        lines.append(f"- **EfficientNet-B0 Test Macro-F1**: {eff['test_macro_f1']*100:.2f}%")
        lines.append(f"- **EfficientNet-B0 Top-3 Accuracy**: {eff['test_top3_accuracy']*100:.2f}%")
        if resnet:
            acc_diff = (resnet["test_accuracy"] - eff["test_accuracy"]) * 100
            f1_diff = (resnet["test_macro_f1"] - eff["test_macro_f1"]) * 100
            lines.append(f"- **ResNet-18 vs. EfficientNet-B0**: ResNet-18 exceeded EfficientNet-B0 by **+{acc_diff:.2f} percentage points** in Test Accuracy and **+{f1_diff:.2f} percentage points** in Macro-F1.")

    obs_path = os.path.join(output_dir, "factual_observations.md")
    with open(obs_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def run_comparison(output_dir: str = "reports/comparison", include_efficientnet: bool = True) -> Dict[str, Any]:
    """Execute complete comparison aggregation and report generation."""
    os.makedirs(output_dir, exist_ok=True)

    split_hash = compute_sha256("data/splits/split.csv")
    label_map_hash = compute_sha256("data/splits/label_map.json")

    metrics_list: List[Dict[str, Any]] = []
    all_sources: List[Dict[str, Any]] = []
    validation_reports: Dict[str, Any] = {
        "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "split_file": "data/splits/split.csv",
        "split_file_sha256": split_hash,
        "label_map_file": "data/splits/label_map.json",
        "label_map_sha256": label_map_hash,
        "models": {},
    }

    for model_key, info in MODEL_REGISTRY.items():
        if not info["is_official"] and not include_efficientnet:
            continue
        m_dict, s_list, v_info = normalize_metrics(model_key, info)
        metrics_list.append(m_dict)
        all_sources.extend(s_list)
        validation_reports["models"][model_key] = v_info

    # 1. Output metric sources lineage CSV
    sources_df = pd.DataFrame(all_sources)
    sources_df.to_csv(os.path.join(output_dir, "metric_sources.csv"), index=False)

    # 2. Output validation report JSON
    with open(os.path.join(output_dir, "validation_report.json"), "w", encoding="utf-8") as f:
        json.dump(validation_reports, f, indent=2)

    # 3. Generate summary JSON
    valid_official = [m for m in metrics_list if m["is_official"] and m["test_accuracy"] is not None]
    summary_json = {
        "generation_timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "split_sha256": split_hash,
        "label_map_sha256": label_map_hash,
        "official_models_evaluated": [m["display_name"] for m in valid_official],
        "official_models_pending": [m["display_name"] for m in metrics_list if m["is_official"] and m["test_accuracy"] is None],
        "benchmark_models": [m["display_name"] for m in metrics_list if not m["is_official"]],
        "best_model_by_test_accuracy": max(valid_official, key=lambda x: x["test_accuracy"])["display_name"] if valid_official else None,
        "best_model_by_test_macro_f1": max(valid_official, key=lambda x: x["test_macro_f1"])["display_name"] if valid_official else None,
        "best_model_by_top3_accuracy": max(valid_official, key=lambda x: x["test_top3_accuracy"])["display_name"] if valid_official else None,
        "metrics": metrics_list,
    }
    with open(os.path.join(output_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary_json, f, indent=2)

    # 4. Generate Tables (CSV, MD, TeX)
    generate_tables(metrics_list, output_dir)

    # 5. Generate Visualizations
    generate_visualizations(metrics_list, output_dir)

    # 6. Generate Factual Observations
    generate_factual_observations(metrics_list, output_dir)

    print(f"Comparison reports and figures generated successfully in '{output_dir}'")
    return summary_json


def main():
    parser = argparse.ArgumentParser(description="Cross-model comparison and reporting framework.")
    parser.add_argument("--output-dir", type=str, default="reports/comparison", help="Directory to save comparison outputs")
    parser.add_argument("--include-efficientnet", action="store_true", default=True, help="Include EfficientNet-B0 benchmark")
    args = parser.parse_args()

    run_comparison(output_dir=args.output_dir, include_efficientnet=args.include_efficientnet)


if __name__ == "__main__":
    main()
