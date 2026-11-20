"""Figures and tables for the README and the model cards, from the files train.py writes.

    uv run --group train python training/report.py
"""
from __future__ import annotations

import json
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "training" / "reports"
MODELS = ROOT / "models"
INK, INK2, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#fcfcfb"
ORIGINAL, CLEAN = "#c3c1b8", "#2a78d6"
FAKE, REAL = "#eb6834", "#2a78d6"
SEQ = matplotlib.colors.LinearSegmentedColormap.from_list("blues", ["#f3f2ee", "#9cc3ef", "#2a78d6", "#123f7a"])

plt.rcParams.update({"figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "axes.edgecolor": GRID,
                     "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK, "font.size": 10,
                     "axes.spines.top": False, "axes.spines.right": False, "savefig.dpi": 160})


def comparison(results: pd.DataFrame) -> None:
    names = list(results.loc[results["setup"] == "clean", "model"])
    y = np.arange(len(names))
    fig, ax = plt.subplots(figsize=(8.6, 3.4))
    for off, setup, color, label in [(-0.19, "original", ORIGINAL, "Original notebook pipeline"),
                                     (0.19, "clean", CLEAN, "Clean (publisher markers and duplicates removed)")]:
        vals = results[results["setup"] == setup].set_index("model").loc[names, "validation_accuracy"]
        ax.barh(y + off, vals, height=0.36, color=color, edgecolor=SURFACE, linewidth=2, label=label)
        for yi, v in zip(y + off, vals):
            ax.text(v - 0.004, yi, f"{v:.1%}", va="center", ha="right", fontsize=9,
                    color=INK if setup == "original" else "white")
    ax.set_yticks(y, names)
    ax.invert_yaxis()
    ax.set_xlim(0.8, 1.0)
    ax.set_xticks(np.arange(0.8, 1.001, 0.05))
    ax.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=0))
    ax.grid(axis="x", color=GRID)
    ax.set_axisbelow(True)
    ax.set_xlabel(f"Accuracy on the {int(results['validation_n'].iloc[0]):,} validation articles that are not\n"
                  "copies of a training article (axis starts at 80 %)", fontsize=9)
    ax.legend(loc="lower left", bbox_to_anchor=(0, 1.02), ncol=2, frameon=False, fontsize=9)
    fig.tight_layout()
    fig.savefig(REPORTS / "comparison.png")
    plt.close(fig)


def confusion(key: str, info: dict) -> None:
    c = info["validation"]["confusion"]
    m = np.array([[c["fake_as_fake"], c["fake_as_real"]], [c["real_as_fake"], c["real_as_real"]]])
    share = m / m.sum(axis=1, keepdims=True)
    fig, ax = plt.subplots(figsize=(3.8, 3.3))
    ax.imshow(share, cmap=SEQ, vmin=0, vmax=1)
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{m[i, j]:,}\n{share[i, j]:.1%}", ha="center", va="center", fontsize=10,
                    color="white" if share[i, j] > 0.55 else INK)
    ax.set_xticks([0, 1], ["Fake", "Real"])
    ax.set_yticks([0, 1], ["Fake", "Real"])
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(f"{info['name']}\nvalidation set, n={info['validation']['n']:,}", fontsize=10, color=INK)
    for s in ax.spines.values():
        s.set_visible(False)
    fig.tight_layout()
    (REPORTS / key).mkdir(exist_ok=True)
    fig.savefig(REPORTS / key / "confusion_matrix.png")
    plt.close(fig)


def top_words(n: int = 12) -> pd.DataFrame:
    model = joblib.load(MODELS / "fake-news-tfidf-logreg" / "model.joblib")
    names, coef = model["vec"].get_feature_names_out(), model["clf"].coef_[0]
    order = np.argsort(coef)
    words = pd.DataFrame({"term": names[np.r_[order[:n], order[-n:]]], "coefficient": coef[np.r_[order[:n], order[-n:]]]})
    words = words.sort_values("coefficient")
    fig, ax = plt.subplots(figsize=(7, 5.6))
    colors = [FAKE if c < 0 else REAL for c in words["coefficient"]]
    ax.barh(words["term"], words["coefficient"], color=colors, edgecolor=SURFACE, linewidth=2)
    ax.axvline(0, color=MUTED, linewidth=1)
    ax.grid(axis="x", color=GRID)
    ax.set_axisbelow(True)
    ax.set_xlabel("Logistic regression coefficient  ←  fake · real  →")
    ax.set_title("Terms that push the TF-IDF + logistic regression model", fontsize=10, color=INK, loc="left")
    fig.tight_layout()
    fig.savefig(REPORTS / "top_terms.png")
    plt.close(fig)
    words.to_csv(REPORTS / "top_terms.csv", index=False)
    return words


def main() -> None:
    results = pd.read_csv(REPORTS / "results.csv")
    comparison(results)
    infos = {}
    for key in results["key"].unique():
        info = json.loads((MODELS / f"fake-news-{key}" / "model_info.json").read_text())
        infos[key] = info
        confusion(key, info)
        (REPORTS / key / "metrics.json").write_text(json.dumps(
            {k: info[k] for k in ("test", "validation", "validation_by_subject", "data")}, indent=2))
    top_words()

    lines = ["| Model | Setup | Test accuracy | Validation accuracy | Validation ROC AUC | US_News (unseen subject) |",
             "|---|---|---|---|---|---|"]
    for _, r in results.sort_values(["model", "setup"], ascending=[True, False]).iterrows():
        lines.append(f"| {r['model']} | {r['setup']} | {r['test_accuracy']:.1%} | {r['validation_accuracy']:.1%} | "
                     f"{r['validation_roc_auc']:.3f} | {r['val_acc_US_News']:.1%} |")
    (REPORTS / "comparison.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
