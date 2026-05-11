from __future__ import annotations

import csv
import html
import json
import math
from pathlib import Path


ROOT = Path(__file__).resolve().parent
SVG_OUT = ROOT / "comparativo_modelos_banner.svg"
CSV_OUT = ROOT / "comparativo_modelos_banner.csv"


def read_json(path: Path) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def as_float(value):
    try:
        if value is None:
            return None
        value = float(value)
        if math.isnan(value):
            return None
        return value
    except (TypeError, ValueError):
        return None


def short_label(run_dir: Path) -> str:
    name = run_dir.name
    if name.startswith("run_qnn_"):
        suffix = name.removeprefix("run_qnn_")[-4:]
        return f"QNN-{suffix}"
    if name.endswith("_noDataClean"):
        return "CNN-noClean"
    if name.endswith("_best"):
        name = name.removesuffix("_best")
    if name.startswith("run_"):
        parts = name.split("_")
        if len(parts) >= 3 and parts[1].isdigit():
            return f"CNN-{parts[1][4:8]}"
    return name.replace("run_", "")


def load_history_metric(run_dir: Path):
    history = read_json(run_dir / "history.json")
    if not history:
        return None, None, None, None

    val_acc = history.get("val_accuracy") or history.get("val_acc")
    val_auc = history.get("val_auc_ovr") or history.get("val_auc")
    val_loss = history.get("val_loss")
    if not isinstance(val_acc, list) or not val_acc:
        return None, None, None, None

    values = [as_float(v) for v in val_acc]
    valid_values = [(idx, v) for idx, v in enumerate(values) if v is not None]
    if not valid_values:
        return None, None, None, None

    idx, acc = max(valid_values, key=lambda item: item[1])
    auc = None
    loss = None
    if isinstance(val_auc, list) and idx < len(val_auc):
        auc = as_float(val_auc[idx])
    if isinstance(val_loss, list) and idx < len(val_loss):
        loss = as_float(val_loss[idx])
    return acc, auc, loss, "best_val_history"


def load_classic_metric(run_dir: Path) -> dict | None:
    candidates = []
    candidates.extend(sorted(run_dir.glob("results_final.json")))
    candidates.extend(sorted(run_dir.glob("results_*.json")))

    for path in candidates:
        data = read_json(path)
        test = data.get("test_metrics") or {}
        final = data.get("final_metrics") or {}

        acc = (
            as_float(test.get("accuracy"))
            or as_float(data.get("final_accuracy"))
            or as_float(data.get("val_accuracy"))
            or as_float(final.get("val_accuracy"))
        )
        auc = (
            as_float(test.get("auc_ovr"))
            or as_float(data.get("final_auc_ovr"))
            or as_float(data.get("val_auc_ovr"))
            or as_float(final.get("val_auc_ovr"))
        )
        loss = (
            as_float(test.get("loss"))
            or as_float(data.get("final_loss"))
            or as_float(data.get("val_loss"))
            or as_float(final.get("val_loss"))
        )

        if acc is not None:
            return {
                "run": run_dir.name,
                "label": short_label(run_dir),
                "family": "Classico",
                "accuracy": acc,
                "auc": auc,
                "loss": loss,
                "source": path.name,
            }

    acc, auc, loss, source = load_history_metric(run_dir)
    if acc is None:
        return None
    return {
        "run": run_dir.name,
        "label": short_label(run_dir),
        "family": "Classico",
        "accuracy": acc,
        "auc": auc,
        "loss": loss,
        "source": source,
    }


def load_qnn_metric(run_dir: Path) -> dict | None:
    test = read_json(run_dir / "test_results.json")
    summary = read_json(run_dir / "metrics_summary.json")
    acc = as_float(test.get("compile_metrics")) or as_float(summary.get("val_accuracy_manual"))
    loss = as_float(test.get("loss"))

    if acc is None:
        hist_acc, _, hist_loss, source = load_history_metric(run_dir)
        acc = hist_acc
        loss = hist_loss
    else:
        source = "test_results.json"

    if acc is None:
        return None

    return {
        "run": run_dir.name,
        "label": short_label(run_dir),
        "family": "Quantico",
        "accuracy": acc,
        "auc": None,
        "loss": loss,
        "source": source,
    }


def collect_metrics() -> list[dict]:
    rows = []
    for run_dir in sorted(ROOT.glob("run_*")):
        if not run_dir.is_dir():
            continue
        if run_dir.name.startswith("run_qnn_"):
            row = load_qnn_metric(run_dir)
        else:
            row = load_classic_metric(run_dir)
        if row is not None:
            rows.append(row)

    rows.sort(key=lambda row: row["accuracy"], reverse=True)
    return rows


def write_csv(rows: list[dict]) -> None:
    with CSV_OUT.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=["rank", "label", "family", "accuracy", "auc", "loss", "source", "run"],
        )
        writer.writeheader()
        for rank, row in enumerate(rows, start=1):
            writer.writerow(
                {
                    "rank": rank,
                    "label": row["label"],
                    "family": row["family"],
                    "accuracy": f"{row['accuracy']:.6f}",
                    "auc": "" if row["auc"] is None else f"{row['auc']:.6f}",
                    "loss": "" if row["loss"] is None else f"{row['loss']:.6f}",
                    "source": row["source"],
                    "run": row["run"],
                }
            )


def svg_text(x, y, text, size=28, weight=500, fill="#26313d", anchor="start"):
    return (
        f'<text x="{x}" y="{y}" font-size="{size}" font-weight="{weight}" '
        f'fill="{fill}" text-anchor="{anchor}" dominant-baseline="middle">'
        f"{html.escape(text)}</text>"
    )


def write_svg(rows: list[dict]) -> None:
    width = 1600
    height = 540
    margin_left = 230
    margin_right = 120
    top = 118
    row_h = 42
    bar_h = 22
    chart_w = width - margin_left - margin_right
    max_acc = 0.80
    min_acc = 0.30
    best = rows[0]["accuracy"] if rows else 0

    def x_for(value: float) -> float:
        value = max(min_acc, min(max_acc, value))
        return margin_left + (value - min_acc) / (max_acc - min_acc) * chart_w

    parts = [
        '<svg xmlns="http://www.w3.org/2000/svg" width="1600" height="540" viewBox="0 0 1600 540">',
        '<rect width="1600" height="540" fill="#f7f7f2"/>',
        '<style>text{font-family:Inter,Segoe UI,Arial,sans-serif}.small{letter-spacing:.3px}</style>',
        svg_text(64, 48, "Comparativo dos Modelos", 38, 760, "#17212b"),
        svg_text(64, 86, "Acuracia em teste; validacao quando teste nao esta disponivel", 20, 460, "#607080"),
        svg_text(1384, 58, "Classico", 20, 650, "#2a6f97"),
        '<rect x="1345" y="49" width="26" height="16" rx="4" fill="#2a6f97"/>',
        svg_text(1500, 58, "Quantico", 20, 650, "#9a6a1d"),
        '<rect x="1461" y="49" width="26" height="16" rx="4" fill="#d29a2e"/>',
    ]

    for tick in [0.30, 0.40, 0.50, 0.60, 0.70, 0.80]:
        x = x_for(tick)
        parts.append(f'<line x1="{x:.1f}" y1="104" x2="{x:.1f}" y2="478" stroke="#d9ded9" stroke-width="1"/>')
        parts.append(svg_text(x, 506, f"{tick * 100:.0f}%", 16, 450, "#7b8790", "middle"))

    for idx, row in enumerate(rows):
        y = top + idx * row_h
        acc = row["accuracy"]
        x0 = x_for(min_acc)
        x1 = x_for(acc)
        color = "#2a6f97" if row["family"] == "Classico" else "#d29a2e"
        if idx == 0:
            color = "#1f8a70"
        label_color = "#17212b" if idx < 3 else "#33404a"

        parts.append(svg_text(64, y + bar_h / 2, row["label"], 20, 680 if idx == 0 else 560, label_color))
        parts.append(f'<rect x="{x0:.1f}" y="{y:.1f}" width="{chart_w:.1f}" height="{bar_h}" rx="7" fill="#e8ebe6"/>')
        parts.append(f'<rect x="{x0:.1f}" y="{y:.1f}" width="{max(6, x1 - x0):.1f}" height="{bar_h}" rx="7" fill="{color}"/>')
        parts.append(svg_text(x1 + 16, y + bar_h / 2, f"{acc * 100:.1f}%", 20, 720 if idx == 0 else 600, color))

        if idx == 0:
            parts.append(svg_text(x1 + 91, y + bar_h / 2, "melhor", 15, 700, "#1f8a70"))

    if rows:
        parts.append(
            svg_text(
                64,
                526,
                f"Melhor resultado: {rows[0]['label']} ({best * 100:.1f}%). Dados em comparativo_modelos_banner.csv",
                15,
                480,
                "#6b7680",
            )
        )

    parts.append("</svg>")
    SVG_OUT.write_text("\n".join(parts), encoding="utf-8")


def main() -> None:
    rows = collect_metrics()
    if not rows:
        raise SystemExit("Nenhuma metrica encontrada.")
    write_csv(rows)
    write_svg(rows)
    print(f"Gerado: {SVG_OUT.name}")
    print(f"Dados: {CSV_OUT.name}")
    for rank, row in enumerate(rows, start=1):
        print(f"{rank:02d}. {row['label']:12s} {row['family']:8s} {row['accuracy'] * 100:5.1f}%  {row['source']}")


if __name__ == "__main__":
    main()
