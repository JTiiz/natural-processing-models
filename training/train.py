"""Train and evaluate the three pipelines in two setups.

    uv run --group train python training/train.py
    uv run --group train python training/train.py --quick     # 20 % of the articles, smoke test

Setups
  original  the notebooks' pipeline: letters-only text, duplicates kept, random 80/20 split.
  clean     publisher markers removed (Reuters dateline, image credits, links, handles),
            duplicate articles dropped before the split, and validation articles that also
            appear in training removed. These three models are saved to models/ and used
            by the app.

Both setups are scored on an internal 20 % test split and on validation_data.csv, whose true
labels are recovered from the subject column. Two of its four subjects (US_News, Middle-east)
never appear in training, so it also measures how the models cope with unseen sections.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from fakenews import data as D  # noqa: E402

REPORTS = ROOT / "training" / "reports"
MODELS = ROOT / "models"
SEED = 42

MODEL_NAMES = {
    "tfidf-logreg": "TF-IDF + Logistic regression",
    "bow-naive-bayes": "Bag of words + Naive Bayes",
    "tfidf-random-forest": "TF-IDF + Random forest",
}


def make_model(key: str, setup: str) -> Pipeline:
    if setup == "original":  # exactly the configuration of notebooks/05_comparison_and_validation
        if key == "tfidf-logreg":
            return Pipeline([("vec", TfidfVectorizer(stop_words="english", ngram_range=(1, 2), max_features=5000)),
                             ("clf", LogisticRegression(max_iter=1000, random_state=SEED))])
        if key == "bow-naive-bayes":
            return Pipeline([("vec", CountVectorizer(stop_words="english", max_features=5000)),
                             ("clf", MultinomialNB())])
        return Pipeline([("vec", TfidfVectorizer(stop_words="english", max_features=3000)),
                         ("clf", RandomForestClassifier(n_estimators=200, n_jobs=-1, random_state=SEED))])
    if key == "tfidf-logreg":
        return Pipeline([("vec", TfidfVectorizer(stop_words="english", ngram_range=(1, 2), min_df=3, max_df=0.8,
                                                 sublinear_tf=True, max_features=200_000)),
                         ("clf", LogisticRegression(C=4, max_iter=2000, random_state=SEED))])
    if key == "bow-naive-bayes":
        return Pipeline([("vec", CountVectorizer(stop_words="english", ngram_range=(1, 2), min_df=3,
                                                 max_features=50_000)),
                         ("clf", MultinomialNB(alpha=0.1))])
    if key == "tfidf-random-forest":
        return Pipeline([("vec", TfidfVectorizer(stop_words="english", min_df=3, max_features=5000)),
                         ("clf", RandomForestClassifier(n_estimators=300, min_samples_leaf=2, n_jobs=-1,
                                                        random_state=SEED))])
    raise KeyError(key)


def scores(y: np.ndarray, p_real: np.ndarray) -> dict:
    pred = (p_real >= 0.5).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return {"accuracy": float(accuracy_score(y, pred)), "f1_fake": float(f1_score(y, pred, pos_label=0)),
            "f1_real": float(f1_score(y, pred, pos_label=1)), "roc_auc": float(roc_auc_score(y, p_real)),
            "confusion": {"fake_as_fake": int(tn), "fake_as_real": int(fp),
                          "real_as_fake": int(fn), "real_as_real": int(tp)},
            "n": int(len(y))}


def prepare(setup: str, train: pd.DataFrame, val: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    tr, va = train.copy(), val.copy()
    if setup == "original":
        tr["x"] = (tr["title"].astype(str) + " " + tr["text"].astype(str)).map(D.basic_clean)
        va["x"] = (va["title"].astype(str) + " " + va["text"].astype(str)).map(D.basic_clean)
        return tr, va, {}
    tr["x"] = [D.clean_article(a, b) for a, b in zip(tr["title"], tr["text"])]
    va["x"] = [D.clean_article(a, b) for a, b in zip(va["title"], va["text"])]
    before_tr, before_va = len(tr), len(va)
    tr = tr[tr["x"].str.len() >= 20].drop_duplicates("x")
    va = va[(va["x"].str.len() >= 20) & ~va["x"].isin(set(tr["x"]))].drop_duplicates("x")
    return tr, va, {"train_dropped": before_tr - len(tr), "validation_dropped": before_va - len(va)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--only", choices=list(MODEL_NAMES))
    args = ap.parse_args()
    t0 = time.perf_counter()

    val = D.load_validation()
    train = D.load_training(D.download_corpus(), val)
    if args.quick:
        train = train.sample(frac=0.2, random_state=SEED)
        val = val.sample(frac=0.2, random_state=SEED)
    out_dir = REPORTS / ("quick" if args.quick else "")
    out_dir.mkdir(parents=True, exist_ok=True)

    rows, dataset = [], {"source": f"https://huggingface.co/datasets/{D.DATASET_REPO}",
                         "revision": D.DATASET_REVISION, "articles_total": int(len(train) + len(val))}
    val_predictions, original_acc = None, {}
    # Both setups are compared on the same validation articles: the ones that are not a
    # copy of a training article once the text is cleaned (815 of the 4,956 are).
    unique_val = prepare("clean", train, val)[1].index
    for setup in ("original", "clean"):
        tr, va, dropped = prepare(setup, train, val)
        if setup == "original":
            va_all, va = va, va.loc[unique_val]
        fit, test = train_test_split(tr, test_size=0.2, stratify=tr["label"], random_state=SEED)
        dataset[setup] = {"train": int(len(fit)), "test": int(len(test)), "validation": int(len(va)), **dropped,
                          "validation_subjects": va["subject"].value_counts().to_dict()}
        print(setup, json.dumps(dataset[setup]), flush=True)
        if setup == "clean":
            val_predictions = va[["title", "subject", "label"]].copy()
        for key, name in MODEL_NAMES.items():
            if args.only and key != args.only:
                continue
            model = make_model(key, setup)
            t = time.perf_counter()
            model.fit(fit["x"], fit["label"])
            seconds = time.perf_counter() - t
            p_test = model.predict_proba(test["x"])[:, 1]
            p_val = model.predict_proba(va["x"])[:, 1]
            m_test, m_val = scores(test["label"].to_numpy(), p_test), scores(va["label"].to_numpy(), p_val)
            by_subject = {s: float(accuracy_score(g["label"], (p_val[va.index.get_indexer(g.index)] >= 0.5)))
                          for s, g in va.groupby("subject")}
            extra = {}
            if setup == "original":  # what the notebook measured: all 4,956, copies included
                extra["validation_accuracy_with_copies"] = float(accuracy_score(
                    va_all["label"], model.predict(va_all["x"])))
            rows.append({"setup": setup, "model": name, "key": key,
                         "test_accuracy": m_test["accuracy"], "test_roc_auc": m_test["roc_auc"], **extra,
                         "validation_n": m_val["n"],
                         "validation_accuracy": m_val["accuracy"], "validation_roc_auc": m_val["roc_auc"],
                         "validation_f1_fake": m_val["f1_fake"], "validation_f1_real": m_val["f1_real"],
                         **{f"val_acc_{s}": a for s, a in by_subject.items()}, "train_seconds": round(seconds, 1)})
            print(f"{setup:8s} {name:30s} test={m_test['accuracy']:.4f} val={m_val['accuracy']:.4f} "
                  f"auc_val={m_val['roc_auc']:.4f} {({s: round(a, 3) for s, a in by_subject.items()})} "
                  f"({seconds:.0f}s)", flush=True)
            if setup != "clean":
                original_acc[key] = m_val["accuracy"]
                continue
            val_predictions[key] = p_val.round(4)
            if args.quick:
                continue
            folder = MODELS / f"fake-news-{key}"
            folder.mkdir(parents=True, exist_ok=True)
            joblib.dump(model, folder / "model.joblib", compress=3)
            info = {"key": key, "name": name, "labels": D.LABELS, "input": "fakenews.data.clean_article(title, text)",
                    "sklearn": __import__("sklearn").__version__, "train_seconds": round(seconds, 1),
                    "test": m_test, "validation": m_val,
                    "original_validation_accuracy": original_acc.get(key), "validation_by_subject": by_subject,
                    "data": dataset["clean"]}
            (folder / "model_info.json").write_text(json.dumps(info, indent=2))

    results = pd.DataFrame(rows)
    results.to_csv(out_dir / "results.csv", index=False)
    val_predictions.to_csv(out_dir / "validation_predictions.csv.gz", index=False)
    (out_dir / "dataset.json").write_text(json.dumps(dataset, indent=2))
    print(results.drop(columns=["key"]).round(4).to_string(index=False))
    print(f"total {time.perf_counter() - t0:.0f}s")


if __name__ == "__main__":
    main()
