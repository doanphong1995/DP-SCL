"""CSV and text-report writers for DP-SCL runs."""

import csv

from .metrics import fmt, fmt_mean_std


def write_csv(path, rows, fieldnames):
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_report(path, config, rows, summary_rows):
    with open(path, "w", encoding="utf-8") as handle:
        handle.write("DP-SCL EXPERIMENT RESULTS\n")
        handle.write(f"Dataset: {config['dataset']} ({config['dataset_name']})\n")
        handle.write(f"Split: {config['split']}\n")
        handle.write(f"Seeds: {config['seeds']}\n")
        handle.write(f"Max epochs: {config['max_epochs']}\n")
        handle.write(f"Early stopping: Val AUC, patience={config['patience']}\n")
        handle.write(f"lambda_con: {config['lambda_con']}\n")
        handle.write(f"temperature: {config['temperature']}\n\n")

        for row in rows:
            if row["status"] != "ok":
                handle.write(f"Seed {row['seed']:>6} | status={row['status']}\n")
                continue
            handle.write(
                f"Seed {row['seed']:>6} | best_epoch={row['best_epoch']} "
                f"stopped={row['stopped_epoch']} threshold={fmt(row['threshold'])} | "
                f"AUC={fmt(row['test_auc'])} ACC={fmt(row['test_acc'])} "
                f"Precision={fmt(row['test_precision'])} Recall={fmt(row['test_recall'])} "
                f"F1={fmt(row['test_f1'])}\n"
            )

        handle.write("\nSUMMARY TABLE\n")
        handle.write("Model, AUC, ACC, Precision, Recall, F1\n")
        for row in summary_rows:
            handle.write(
                f"{row['model']}, {fmt_mean_std(row, 'auc')}, "
                f"{fmt_mean_std(row, 'acc')}, {fmt_mean_std(row, 'precision')}, "
                f"{fmt_mean_std(row, 'recall')}, {fmt_mean_std(row, 'f1')}\n"
            )

