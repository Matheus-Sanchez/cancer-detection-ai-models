from __future__ import annotations

import csv
import html
import json
import math
from pathlib import Path


ROOT = Path(__file__).resolve().parent
SVG_OUT = ROOT / "curvas_treinamento_todos_modelos.svg"
CSV_OUT = ROOT / "curvas_treinamento_todos_modelos.csv"


CLASSIC_COLORS = [
    "#005f73",
    "#0a9396",
    "#1d4ed8",
    "#0891b2",
    "#4361ee",
    "#2f4858",
    "#0077b6",
]

QUANTUM_COLORS = [
    "#ff006e",
    "#ffbe0b",
]


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
        return f"QNN-{name.removeprefix('run_qnn_')[-4:]}"
    if name.endswith("_noDataClean"):
        return "CNN-noClean"
    if name.endswith("_best"):
        name = name.removesuffix("_best")
    if name.startswith("run_"):
        parts = name.split("_")
        if len(parts) >= 3 and parts[1].isdigit():
            return f"CNN-{parts[1][4:8]}"
    return name.replace("run_", "")


def load_curves(run_dir: Path) -> dict | None:
    history = read_json(run_dir / "history.json")
    if not history:
        return None

    train_acc = history.get("accuracy") or history.get("acc") or []
    val_acc = history.get("val_accuracy") or history.get("val_acc") or []
    train_loss = history.get("loss") or []
    val_loss = history.get("val_loss") or []
    n_epochs = max(len(train_acc), len(val_acc), len(train_loss), len(val_loss))
    if n_epochs == 0 or not val_acc:
        return None

    points = []
    for index in range(n_epochs):
        points.append(
            {
                "epoch": index + 1,
                "train_accuracy": as_float(train_acc[index]) if index < len(train_acc) else None,
                "val_accuracy": as_float(val_acc[index]) if index < len(val_acc) else None,
                "train_loss": as_float(train_loss[index]) if index < len(train_loss) else None,
                "val_loss": as_float(val_loss[index]) if index < len(val_loss) else None,
            }
        )

    family = "Quantico" if run_dir.name.startswith("run_qnn_") else "Classico"
    valid_val = [p["val_accuracy"] for p in points if p["val_accuracy"] is not None]
    best_val = max(valid_val) if valid_val else None

    return {
        "run": run_dir.name,
        "label": short_label(run_dir),
        "family": family,
        "points": points,
        "epochs": n_epochs,
        "best_val_accuracy": best_val,
    }


def collect_runs() -> list[dict]:
    runs = []
    for run_dir in sorted(ROOT.glob("run_*")):
        if run_dir.is_dir():
            run = load_curves(run_dir)
            if run is not None:
                runs.append(run)
    runs.sort(key=lambda run: run["best_val_accuracy"] or 0, reverse=True)
    classic_index = 0
    quantum_index = 0
    for run in runs:
        if run["family"] == "Quantico":
            run["color"] = QUANTUM_COLORS[quantum_index % len(QUANTUM_COLORS)]
            run["dash"] = "14 8"
            quantum_index += 1
        else:
            run["color"] = CLASSIC_COLORS[classic_index % len(CLASSIC_COLORS)]
            run["dash"] = ""
            classic_index += 1
    return runs


def write_csv(runs: list[dict]) -> None:
    with CSV_OUT.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "label",
                "family",
                "run",
                "epoch",
                "train_accuracy",
                "val_accuracy",
                "train_loss",
                "val_loss",
            ],
        )
        writer.writeheader()
        for run in runs:
            for point in run["points"]:
                writer.writerow(
                    {
                        "label": run["label"],
                        "family": run["family"],
                        "run": run["run"],
                        "epoch": point["epoch"],
                        "train_accuracy": "" if point["train_accuracy"] is None else f"{point['train_accuracy']:.6f}",
                        "val_accuracy": "" if point["val_accuracy"] is None else f"{point['val_accuracy']:.6f}",
                        "train_loss": "" if point["train_loss"] is None else f"{point['train_loss']:.6f}",
                        "val_loss": "" if point["val_loss"] is None else f"{point['val_loss']:.6f}",
                    }
                )


def svg_text(x, y, text, size=22, weight=500, fill="#26313d", anchor="start"):
    return (
        f'<text x="{x}" y="{y}" font-size="{size}" font-weight="{weight}" '
        f'fill="{fill}" text-anchor="{anchor}" dominant-baseline="middle">'
        f"{html.escape(text)}</text>"
    )


def polyline(points: list[tuple[float, float]], color: str, width: float = 3.0, opacity: float = 0.92, dash: str = "") -> str:
    coords = " ".join(f"{x:.1f},{y:.1f}" for x, y in points)
    dash_attr = f' stroke-dasharray="{dash}"' if dash else ""
    return (
        f'<polyline points="{coords}" fill="none" stroke="{color}" '
        f'stroke-width="{width}" stroke-linecap="round" stroke-linejoin="round" opacity="{opacity}"{dash_attr}/>'
    )


def write_svg(runs: list[dict]) -> None:
    width = 1200
    height = 560
    plot_x = 92
    plot_y = 200
    plot_w = 1040
    plot_h = 266
    legend_x = 92
    legend_y = 104
    legend_cols = 3
    legend_cell_w = 360
    legend_row_h = 30
    y_min = 0.25
    y_max = 0.80
    max_epoch = max(run["epochs"] for run in runs)

    def x_for(epoch: int) -> float:
        if max_epoch <= 1:
            return plot_x
        return plot_x + ((epoch - 1) / (max_epoch - 1)) * plot_w

    def y_for(value: float) -> float:
        value = max(y_min, min(y_max, value))
        return plot_y + (y_max - value) / (y_max - y_min) * plot_h

    parts = [
        '<svg xmlns="http://www.w3.org/2000/svg" width="1200" height="560" viewBox="0 0 1200 560">',
        '<rect width="1200" height="560" fill="#ffffff"/>',
        '<style>text{font-family:Inter,Segoe UI,Arial,sans-serif}</style>',
        svg_text(48, 38, "Curvas de Treinamento", 32, 760, "#17212b"),
        svg_text(48, 72, "Acuracia de validacao por epoca: classicos em azul, quanticos tracejados em cores quentes", 17, 460, "#607080"),
        f'<rect x="{plot_x}" y="{plot_y}" width="{plot_w}" height="{plot_h}" rx="10" fill="#ffffff"/>',
    ]

    for index, run in enumerate(runs):
        row = index // legend_cols
        col = index % legend_cols
        x = legend_x + col * legend_cell_w
        y = legend_y + row * legend_row_h
        best = run["best_val_accuracy"] or 0
        dash_attr = f' stroke-dasharray="{run["dash"]}"' if run["dash"] else ""
        legend_width = 6 if run["family"] == "Quantico" else 5
        label_size = 13 if run["label"] == "CNN-noClean" else 14
        parts.append(f'<line x1="{x}" y1="{y}" x2="{x + 34}" y2="{y}" stroke="{run["color"]}" stroke-width="{legend_width}" stroke-linecap="round"{dash_attr}/>')
        parts.append(svg_text(x + 46, y, run["label"], label_size, 640 if index == 0 else 540, "#25313b"))
        parts.append(svg_text(x + 328, y, f"{best * 100:.1f}%", 14, 650, run["color"], "end"))

    for tick in [0.30, 0.40, 0.50, 0.60, 0.70, 0.80]:
        y = y_for(tick)
        parts.append(f'<line x1="{plot_x}" y1="{y:.1f}" x2="{plot_x + plot_w}" y2="{y:.1f}" stroke="#d9ded9" stroke-width="1"/>')
        parts.append(svg_text(plot_x - 18, y, f"{tick * 100:.0f}%", 16, 460, "#7b8790", "end"))

    epoch_ticks = [1, 25, 50, 75, 100]
    for epoch in epoch_ticks:
        if epoch > max_epoch:
            continue
        x = x_for(epoch)
        parts.append(f'<line x1="{x:.1f}" y1="{plot_y}" x2="{x:.1f}" y2="{plot_y + plot_h}" stroke="#e5e8e3" stroke-width="1"/>')
        parts.append(svg_text(x, plot_y + plot_h + 31, str(epoch), 16, 460, "#7b8790", "middle"))

    parts.extend(
        [
            f'<line x1="{plot_x}" y1="{plot_y + plot_h}" x2="{plot_x + plot_w}" y2="{plot_y + plot_h}" stroke="#9da9a6" stroke-width="1.4"/>',
            f'<line x1="{plot_x}" y1="{plot_y}" x2="{plot_x}" y2="{plot_y + plot_h}" stroke="#9da9a6" stroke-width="1.4"/>',
            svg_text(plot_x + plot_w / 2, plot_y + plot_h + 56, "Epoca", 17, 560, "#607080", "middle"),
            svg_text(plot_x - 70, plot_y - 24, "Val. acc.", 17, 600, "#607080"),
        ]
    )

    for run in runs:
        pts = []
        for point in run["points"]:
            val = point["val_accuracy"]
            if val is not None:
                pts.append((x_for(point["epoch"]), y_for(val)))
        if len(pts) >= 2:
            width_line = 4.8 if run["family"] == "Quantico" else 3.2
            if run is runs[0]:
                width_line = 4.3
            opacity = 0.98 if run["family"] == "Quantico" else 0.82
            parts.append(polyline(pts, run["color"], width_line, opacity, run["dash"]))
            end_x, end_y = pts[-1]
            radius = 5.8 if run["family"] == "Quantico" else 4.4
            parts.append(f'<circle cx="{end_x:.1f}" cy="{end_y:.1f}" r="{radius}" fill="{run["color"]}" opacity="0.95"/>')

    if runs:
        top = runs[0]
        parts.append(
            svg_text(
                64,
                536,
                f"Melhor pico de validacao: {top['label']} ({(top['best_val_accuracy'] or 0) * 100:.1f}%). Dados em curvas_treinamento_todos_modelos.csv",
                14,
                480,
                "#6b7680",
            )
        )

    parts.append("</svg>")
    SVG_OUT.write_text("\n".join(parts), encoding="utf-8")


def main() -> None:
    runs = collect_runs()
    if not runs:
        raise SystemExit("Nenhuma curva encontrada.")
    write_csv(runs)
    write_svg(runs)
    print(f"Gerado: {SVG_OUT.name}")
    print(f"Dados: {CSV_OUT.name}")
    for index, run in enumerate(runs, start=1):
        print(f"{index:02d}. {run['label']:12s} {run['family']:8s} epocas={run['epochs']:3d} melhor_val_acc={(run['best_val_accuracy'] or 0) * 100:5.1f}%")


if __name__ == "__main__":
    main()
