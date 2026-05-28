"""
Generate Chapter 3 methodology figures from the existing SQL injection project
artifacts.

This script creates:
1. Sample-size sensitivity graph
2. Class-distribution chart for benign vs malicious queries
3. Injection-type distribution chart
4. Runtime dashboard summary from middleware logs

The script is intentionally data-driven so the figures can be regenerated after
new training runs or after additional API traffic has been logged.
"""

import json
import os
import re
from collections import Counter

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PLOTS_DIR = os.path.join(BASE_DIR, "sql_injection_plots")
DATASET_PATH = os.path.join(BASE_DIR, "rbsqli_dataset_1k.csv")
LOG_PATH = os.path.join(BASE_DIR, "sql_injection_logs.json")

SAMPLE_REPORTS = {
    5000: "/Users/oeze/Downloads/SAMPLE_SIZE_5000.txt",
    10000: "/Users/oeze/Downloads/SAMPLE_SIZE_10000.txt",
    50000: "/Users/oeze/Downloads/SAMPLE_SIZE_50000.txt",
    100000: "/Users/oeze/Downloads/SAMPLE_SIZE_100000.txt",
}


def ensure_output_dir():
    """Create the plot directory if it does not already exist."""
    os.makedirs(PLOTS_DIR, exist_ok=True)


def extract_metric(text, pattern, cast=float, default=None):
    """Extract one metric from a report file using a regular expression."""
    match = re.search(pattern, text, re.S)
    if not match:
        return default
    value = match.group(1).replace(",", "")
    return cast(value)


def load_sample_size_summary():
    """
    Build a compact summary table from the saved training run reports.

    The metrics use the final selected best-model summary written by task1.py.
    """
    rows = []

    for sample_size, path in SAMPLE_REPORTS.items():
        if not os.path.exists(path):
            continue

        report_text = open(path, "r", encoding="utf-8", errors="ignore").read()

        row = {
            "sample_size": sample_size,
            "best_model": extract_metric(report_text, r"SELECTED BEST MODEL: (.+)", cast=str, default="Unknown"),
            "accuracy": extract_metric(report_text, r"SELECTED BEST MODEL:.*?Accuracy: ([0-9.]+)", default=None),
            "precision": extract_metric(report_text, r"SELECTED BEST MODEL:.*?Precision: ([0-9.]+)", default=None),
            "recall": extract_metric(report_text, r"SELECTED BEST MODEL:.*?Recall: ([0-9.]+)", default=None),
            "f1_score": extract_metric(report_text, r"SELECTED BEST MODEL:.*?F1 Score: ([0-9.]+)", default=None),
            "weighted_f1": extract_metric(report_text, r"SELECTED BEST MODEL:.*?Weighted F1: ([0-9.]+)", default=None),
            "auc": extract_metric(report_text, r"SELECTED BEST MODEL:.*?AUC: ([0-9.]+)", default=None),
            "tfidf_features": extract_metric(report_text, r"TF-IDF features created: ([0-9,]+) features", cast=int, default=None),
            "combined_features": extract_metric(report_text, r"Train: \([0-9,]+, ([0-9,]+)\)", cast=int, default=None),
        }

        rows.append(row)

    summary_df = pd.DataFrame(rows).sort_values("sample_size")
    summary_df.to_csv(os.path.join(PLOTS_DIR, "figure_3_4_sample_size_summary.csv"), index=False)
    return summary_df


def load_model_f1_by_sample_size():
    """
    Extract test F1-scores for Random Forest, DNN, LSTM, and GRU from the
    saved training run reports.

    The current run logs print the model comparison rows in this order:
    model_name, accuracy, precision, recall, f1_score, weighted_f1, auc.
    """
    rows = []

    for sample_size, path in SAMPLE_REPORTS.items():
        if not os.path.exists(path):
            continue

        report_text = open(path, "r", encoding="utf-8", errors="ignore").read()

        row = {
            "sample_size": sample_size,
            "Random Forest": extract_metric(report_text, r"SELECTED BEST MODEL:.*?F1 Score: ([0-9.]+)", default=None),
            "DNN": extract_metric(report_text, r"Deep Neural Network [0-9.]+ [0-9.]+ [0-9.]+ ([0-9.]+) [0-9.]+ [0-9.]+", default=None),
            "LSTM": extract_metric(report_text, r"LSTM \(Character-level\) [0-9.]+ [0-9.]+ [0-9.]+ ([0-9.]+) [0-9.]+ [0-9.]+", default=None),
            "GRU": extract_metric(report_text, r"GRU \(Character-level\) [0-9.]+ [0-9.]+ [0-9.]+ ([0-9.]+) [0-9.]+ [0-9.]+", default=None),
        }
        rows.append(row)

    comparison_df = pd.DataFrame(rows).sort_values("sample_size")
    comparison_df.to_csv(os.path.join(PLOTS_DIR, "figure_3_4_model_f1_by_sample_size.csv"), index=False)
    return comparison_df


def generate_sample_size_sensitivity_graph(summary_df):
    """Generate Figure 3.4 using the training-run summaries."""
    if summary_df.empty:
        return None

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].plot(summary_df["sample_size"], summary_df["f1_score"], marker="o", linewidth=2.5, label="F1-score")
    axes[0].plot(summary_df["sample_size"], summary_df["weighted_f1"], marker="s", linewidth=2.0, label="Weighted F1")
    axes[0].plot(summary_df["sample_size"], summary_df["accuracy"], marker="^", linewidth=2.0, label="Accuracy")
    axes[0].set_title("Figure 3.4a: Sample Size vs Performance")
    axes[0].set_xlabel("Sample Size")
    axes[0].set_ylabel("Score")
    axes[0].set_ylim(0.95, 1.01)
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    axes[1].plot(summary_df["sample_size"], summary_df["tfidf_features"], marker="o", linewidth=2.5, color="#2a9d8f", label="TF-IDF features")
    axes[1].plot(summary_df["sample_size"], summary_df["combined_features"], marker="s", linewidth=2.0, color="#e76f51", label="Combined feature width")
    axes[1].set_title("Figure 3.4b: Sample Size vs Feature Space")
    axes[1].set_xlabel("Sample Size")
    axes[1].set_ylabel("Feature Count")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()

    plt.tight_layout()
    output_path = os.path.join(PLOTS_DIR, "figure_3_4_sample_size_sensitivity.png")
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    return output_path


def generate_model_f1_line_chart(comparison_df):
    """
    Generate a dedicated line chart comparing test F1-score against sample size
    for the four main models discussed in the dissertation.
    """
    if comparison_df.empty:
        return None

    plt.figure(figsize=(10, 6))

    model_styles = {
        "Random Forest": {"color": "#1b5e20", "marker": "o"},
        "DNN": {"color": "#1565c0", "marker": "s"},
        "LSTM": {"color": "#ef6c00", "marker": "^"},
        "GRU": {"color": "#6a1b9a", "marker": "D"},
    }

    for model_name, style in model_styles.items():
        plt.plot(
            comparison_df["sample_size"],
            comparison_df[model_name],
            linewidth=2.5,
            marker=style["marker"],
            color=style["color"],
            label=model_name,
        )

    plt.title("Sample Size Against Test F1-score for Random Forest, DNN, LSTM and GRU")
    plt.xlabel("Sample Size")
    plt.ylabel("Test F1-score")
    plt.ylim(0.2, 1.02)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()

    output_path = os.path.join(PLOTS_DIR, "figure_3_4_model_f1_line_chart.png")
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    return output_path


def load_dataset_for_distribution(nrows=100000):
    """
    Load a manageable slice of the dataset for descriptive charts.

    Using a fixed upper bound keeps the chart generation quick while still
    reflecting the same scale used in the larger training experiments.
    """
    return pd.read_csv(DATASET_PATH, nrows=nrows, low_memory=False)


def generate_class_distribution_chart(df):
    """Generate Figure 3.5: benign vs malicious query distribution."""
    counts = df["vulnerability_status"].value_counts()

    plt.figure(figsize=(8, 5))
    palette = ["#4caf50" if label == "No" else "#d32f2f" for label in counts.index]
    sns.barplot(x=counts.index, y=counts.values, palette=palette)

    for index, value in enumerate(counts.values):
        percentage = (value / counts.sum()) * 100
        plt.text(index, value, f"{value:,}\n({percentage:.2f}%)", ha="center", va="bottom", fontsize=10)

    plt.title("Figure 3.5: Class Distribution of Benign and Malicious Queries")
    plt.xlabel("Vulnerability Status")
    plt.ylabel("Number of Queries")
    plt.tight_layout()

    output_path = os.path.join(PLOTS_DIR, "figure_3_5_class_distribution.png")
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    return output_path


def generate_injection_type_distribution_chart(df):
    """Generate Figure 3.6: attack-type frequency distribution."""
    type_counts = df["injection_type"].fillna("Unknown").value_counts()

    plt.figure(figsize=(10, 6))
    sns.barplot(x=type_counts.index, y=type_counts.values, palette="viridis")

    for index, value in enumerate(type_counts.values):
        percentage = (value / type_counts.sum()) * 100
        plt.text(index, value, f"{value:,}\n({percentage:.2f}%)", ha="center", va="bottom", fontsize=9)

    plt.title("Figure 3.6: Distribution of SQL Injection Types")
    plt.xlabel("Injection Type")
    plt.ylabel("Number of Queries")
    plt.xticks(rotation=25, ha="right")
    plt.tight_layout()

    output_path = os.path.join(PLOTS_DIR, "figure_3_6_injection_type_distribution.png")
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    return output_path


def load_runtime_logs():
    """Load the middleware log file if it exists."""
    if not os.path.exists(LOG_PATH):
        return []
    with open(LOG_PATH, "r", encoding="utf-8") as file_handle:
        return json.load(file_handle)


def normalise_attack_type(log_entry):
    """
    Extract the most useful attack type for analytics.

    If the top-level attack type is 'Unknown', the function attempts to recover
    a more specific label from the nested prediction details.
    """
    top_level = log_entry.get("attack_type")
    if top_level and top_level != "Unknown":
        return top_level

    for detail in log_entry.get("prediction", {}).get("details", []):
        attack_type = detail.get("attack_type")
        if attack_type and attack_type != "Unknown":
            return attack_type

    return top_level or "Benign/No Attack"


def generate_runtime_dashboard(logs, output_name="runtime_dashboard_summary", title_suffix=""):
    """Generate a four-panel runtime dashboard from middleware logs."""
    if not logs:
        return None

    runtime_df = pd.DataFrame(logs)
    runtime_df["is_malicious"] = runtime_df["is_malicious"].astype(bool)
    runtime_df["blocked"] = runtime_df["blocked"].astype(bool)
    runtime_df["attack_type_clean"] = runtime_df.apply(normalise_attack_type, axis=1)
    runtime_df["timestamp"] = pd.to_datetime(runtime_df["timestamp"], errors="coerce")

    status_counts = Counter({"Blocked": int(runtime_df["blocked"].sum()), "Allowed": int((~runtime_df["blocked"]).sum())})
    malicious_counts = Counter({"Malicious": int(runtime_df["is_malicious"].sum()), "Benign": int((~runtime_df["is_malicious"]).sum())})
    attack_counts = runtime_df["attack_type_clean"].value_counts().head(8)
    route_counts = runtime_df["route"].value_counts().head(8)

    fig, axes = plt.subplots(2, 2, figsize=(15, 10))

    axes[0, 0].pie(status_counts.values(), labels=status_counts.keys(), autopct="%1.1f%%", colors=["#d32f2f", "#388e3c"], startangle=90)
    axes[0, 0].set_title("Blocked vs Allowed Requests")

    sns.barplot(x=list(malicious_counts.keys()), y=list(malicious_counts.values()), palette=["#ef5350", "#66bb6a"], ax=axes[0, 1])
    axes[0, 1].set_title("Malicious vs Benign Decisions")
    axes[0, 1].set_ylabel("Request Count")

    sns.barplot(x=attack_counts.values, y=attack_counts.index, palette="magma", ax=axes[1, 0])
    axes[1, 0].set_title("Most Common Detected Attack Types")
    axes[1, 0].set_xlabel("Count")
    axes[1, 0].set_ylabel("Attack Type")

    sns.barplot(x=route_counts.values, y=route_counts.index, palette="Blues_r", ax=axes[1, 1])
    axes[1, 1].set_title("Most Frequently Tested Routes")
    axes[1, 1].set_xlabel("Count")
    axes[1, 1].set_ylabel("Route")

    plt.suptitle(f"Runtime Dashboard Summary from Middleware Logs{title_suffix}", fontsize=15, y=1.02)
    plt.tight_layout()

    output_path = os.path.join(PLOTS_DIR, f"{output_name}.png")
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()

    runtime_summary = pd.DataFrame([
        {
            "total_requests": len(runtime_df),
            "blocked_requests": int(runtime_df["blocked"].sum()),
            "allowed_requests": int((~runtime_df["blocked"]).sum()),
            "malicious_predictions": int(runtime_df["is_malicious"].sum()),
            "benign_predictions": int((~runtime_df["is_malicious"]).sum()),
        }
    ])
    runtime_summary.to_csv(os.path.join(PLOTS_DIR, f"{output_name}.csv"), index=False)

    return output_path


def build_dataset_batch_dataframe(logs):
    """
    Convert `/dataset_batch_detect` log entries into a flat dataframe that can
    be used for dissertation analytics.

    Each logged row contains the sampled dataset metadata in `request_data`,
    along with the model decision in the top-level prediction fields.
    """
    rows = []

    for entry in logs:
        if entry.get("route") != "/dataset_batch_detect":
            continue

        request_data = entry.get("request_data", {})
        row_preview = request_data.get("row_preview", {})

        rows.append(
            {
                "batch_id": request_data.get("batch_id"),
                "sample_size": request_data.get("sample_size"),
                "sql_query": row_preview.get("sql_query"),
                "true_label": row_preview.get("vulnerability_status"),
                "injection_type": row_preview.get("injection_type"),
                "predicted_malicious": bool(entry.get("is_malicious", False)),
                "blocked": bool(entry.get("blocked", False)),
                "confidence": float(entry.get("confidence", 0.0)),
                "attack_type": entry.get("attack_type") or "Unknown",
            }
        )

    return pd.DataFrame(rows)


def generate_dataset_batch_reports(logs):
    """
    Generate analytics for the automated deployment-behaviour runs.

    The output is useful for the dissertation because it gives route-level and
    dataset-level summaries from the same middleware execution path.
    """
    batch_df = build_dataset_batch_dataframe(logs)
    if batch_df.empty:
        return None

    batch_df.to_csv(os.path.join(PLOTS_DIR, "dataset_batch_logs.csv"), index=False)

    class_counts = batch_df["true_label"].fillna("Unknown").value_counts()
    injection_counts = batch_df["injection_type"].fillna("Unknown").value_counts()
    attack_counts = batch_df["attack_type"].fillna("Unknown").value_counts()
    block_counts = pd.Series({
        "Blocked": int(batch_df["blocked"].sum()),
        "Allowed": int((~batch_df["blocked"]).sum())
    })

    # Figure 1: class distribution from batch logs
    plt.figure(figsize=(8, 5))
    sns.barplot(x=class_counts.index, y=class_counts.values, palette=["#4caf50", "#d32f2f", "#607d8b"][: len(class_counts)])
    plt.title("Dataset Batch Run: Class Distribution")
    plt.xlabel("True Label")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "dataset_batch_class_distribution.png"), dpi=300, bbox_inches="tight")
    plt.close()

    # Figure 2: injection type distribution from batch logs
    plt.figure(figsize=(10, 6))
    sns.barplot(x=injection_counts.index, y=injection_counts.values, palette="viridis")
    plt.title("Dataset Batch Run: Injection-Type Distribution")
    plt.xlabel("Injection Type")
    plt.ylabel("Count")
    plt.xticks(rotation=25, ha="right")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "dataset_batch_injection_type_distribution.png"), dpi=300, bbox_inches="tight")
    plt.close()

    # Figure 3: attack type distribution from model decisions
    plt.figure(figsize=(10, 6))
    sns.barplot(x=attack_counts.index, y=attack_counts.values, palette="magma")
    plt.title("Dataset Batch Run: Predicted Attack-Type Distribution")
    plt.xlabel("Attack Type")
    plt.ylabel("Count")
    plt.xticks(rotation=25, ha="right")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "dataset_batch_attack_type_distribution.png"), dpi=300, bbox_inches="tight")
    plt.close()

    # Figure 4: blocked vs allowed
    plt.figure(figsize=(7, 5))
    sns.barplot(x=block_counts.index, y=block_counts.values, palette=["#d32f2f", "#388e3c"])
    plt.title("Dataset Batch Run: Blocked vs Allowed")
    plt.xlabel("Decision")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "dataset_batch_block_allow_distribution.png"), dpi=300, bbox_inches="tight")
    plt.close()

    # Batch run summary by sample size
    sample_summary = (
        batch_df.groupby("sample_size")
        .agg(
            total_queries=("sql_query", "count"),
            malicious_predictions=("predicted_malicious", "sum"),
            blocked_requests=("blocked", "sum"),
            avg_confidence=("confidence", "mean"),
        )
        .reset_index()
        .sort_values("sample_size")
    )
    sample_summary.to_csv(os.path.join(PLOTS_DIR, "dataset_batch_sample_size_summary.csv"), index=False)

    plt.figure(figsize=(10, 6))
    plt.plot(sample_summary["sample_size"], sample_summary["malicious_predictions"], marker="o", linewidth=2.5, label="Malicious predictions")
    plt.plot(sample_summary["sample_size"], sample_summary["blocked_requests"], marker="s", linewidth=2.5, label="Blocked requests")
    plt.plot(sample_summary["sample_size"], sample_summary["avg_confidence"], marker="^", linewidth=2.5, label="Average confidence")
    plt.title("Automated Deployment Behaviour by Sample Size")
    plt.xlabel("Sample Size")
    plt.ylabel("Count / Confidence")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "dataset_batch_sample_size_behaviour.png"), dpi=300, bbox_inches="tight")
    plt.close()

    batch_logs = pd.DataFrame(logs)
    if not batch_logs.empty:
        batch_logs = batch_logs[batch_logs["route"] == "/dataset_batch_detect"].copy()
    if not batch_logs.empty:
        generate_runtime_dashboard(
            batch_logs.to_dict("records"),
            output_name="runtime_dashboard_summary_dataset_batch",
            title_suffix=" - Dataset Batch Only",
        )

    return {
        "class_distribution": class_counts.to_dict(),
        "injection_distribution": injection_counts.to_dict(),
        "attack_distribution": attack_counts.to_dict(),
        "block_distribution": block_counts.to_dict(),
    }


def main():
    """Generate all requested Chapter 3 figures."""
    ensure_output_dir()
    sns.set_theme(style="whitegrid")

    summary_df = load_sample_size_summary()
    sample_plot = generate_sample_size_sensitivity_graph(summary_df)
    model_f1_df = load_model_f1_by_sample_size()
    model_f1_plot = generate_model_f1_line_chart(model_f1_df)

    dataset_df = load_dataset_for_distribution()
    class_plot = generate_class_distribution_chart(dataset_df)
    injection_plot = generate_injection_type_distribution_chart(dataset_df)

    logs = load_runtime_logs()
    runtime_plot = generate_runtime_dashboard(logs)
    dataset_batch_reports = generate_dataset_batch_reports(logs)

    print("Generated figures:")
    for path in [sample_plot, model_f1_plot, class_plot, injection_plot, runtime_plot]:
        if path:
            print(f" - {path}")
    if dataset_batch_reports:
        print(" - dataset_batch_logs.csv")
        print(" - dataset_batch_class_distribution.png")
        print(" - dataset_batch_injection_type_distribution.png")
        print(" - dataset_batch_attack_type_distribution.png")
        print(" - dataset_batch_block_allow_distribution.png")
        print(" - dataset_batch_sample_size_summary.csv")
        print(" - dataset_batch_sample_size_behaviour.png")
        print(" - runtime_dashboard_summary_dataset_batch.png")
        print(" - runtime_dashboard_summary_dataset_batch.csv")


if __name__ == "__main__":
    main()
