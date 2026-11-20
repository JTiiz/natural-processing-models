"""Publish the three classifiers to the Hugging Face Hub, each with a model card.

    uv run --group train python training/upload_to_hub.py --dry-run   # writes the card previews
    uv run --group train python training/upload_to_hub.py             # uploads (after `hf auth login`)
"""
from __future__ import annotations

import argparse
import json
import shutil
import tempfile
from pathlib import Path

from huggingface_hub import HfApi

ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "training" / "reports"
MODELS = ROOT / "models"
PROJECT_URL = "https://github.com/JTiiz/natural-processing-models"
DATASET = "haoxianc/gilchr_fake-and-real-news-dataset"

DESCRIPTIONS = {
    "tfidf-logreg": "TF-IDF on words and word pairs (sublinear term frequency, terms in at least 3 articles) "
                    "and an L2 logistic regression (C=4). The most reliable of the three.",
    "bow-naive-bayes": "Word and word-pair counts (50,000 most frequent) and multinomial Naive Bayes (alpha=0.1).",
    "tfidf-random-forest": "TF-IDF on the 5,000 most frequent words and a random forest of 300 trees "
                           "(at least 2 articles per leaf).",
}

CARD = """---
library_name: sklearn
pipeline_tag: text-classification
language: [en]
tags: [sklearn, text-classification, fake-news, tf-idf]
datasets: [{dataset}]
metrics: [accuracy, roc_auc]
---

# Fake news classifier · {name}

Labels an English news article as **fake** (0) or **real** (1). {description}

The model is a scikit-learn `Pipeline` (vectorizer + classifier) that takes the article's headline and
text after `clean_article()` from the [project repository]({project_url}).

| Metric | Value |
|---|---|
| Accuracy, validation set (n={val_n:,}) | {val_acc:.1%} |
| ROC AUC, validation set | {val_auc:.3f} |
| Accuracy, unseen section US_News (n={us_n}) | {us_acc:.1%} |
| Accuracy, internal 20 % test split (n={test_n:,}) | {test_acc:.1%} |
| The notebook's original pipeline (uncleaned text), validation set | {orig_acc:.1%} |

![Confusion matrix](confusion_matrix.png)

## What it learns, and what it does not

The ISOT dataset labels articles by **source**: real articles are Reuters wires and fake articles come
from sites flagged by PolitiFact and Wikipedia (2015-2018). The model therefore learns the writing
style of agency news ("said", "spokesman", the day of the week) against that of those sites
("video", "watch", "just"). It does not check facts: a made-up story written like an agency wire
is classified as real.

Before training, the text is stripped of the markers that reveal the publisher: the Reuters dateline
("WASHINGTON (Reuters) -", present in 99 % of the real articles and in none of the fake ones), image
credits, links and @handles. Duplicate articles are removed, and validation articles that are copies
of a training article (815 of 4,956) are left out of the scores.

## Data

ISOT Fake News dataset as given in the Ironhack project: `data.csv` (train) and
`validation_data.csv` (external validation, true labels recovered from the `subject` column).
Read from [{dataset}](https://huggingface.co/datasets/{dataset}) (MIT), whose labels are inverted
and flipped back. Train {train:,} articles, test {test_n:,}, validation {val_n:,}.

## Usage

scikit-learn must be version {sklearn}.

```python
import joblib
from huggingface_hub import hf_hub_download
from fakenews.data import clean_article   # from the project repository

model = joblib.load(hf_hub_download("{repo_id}", "model.joblib"))   # only load pickles you trust
text = clean_article("Headline of the article", "Body of the article...")
p_fake, p_real = model.predict_proba([text])[0]
```
"""


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--user", default="jramduran", help="Hub account (only used with --dry-run)")
    args = ap.parse_args()
    api = HfApi()
    user = args.user if args.dry_run else api.whoami()["name"]

    for key, description in DESCRIPTIONS.items():
        src = MODELS / f"fake-news-{key}"
        info = json.loads((src / "model_info.json").read_text())
        repo = f"{user}/fake-news-{key}"
        us = info["validation_by_subject"].get("US_News")
        card = CARD.format(
            name=info["name"], description=description, dataset=DATASET, project_url=PROJECT_URL, repo_id=repo,
            sklearn=info["sklearn"], val_n=info["validation"]["n"], val_acc=info["validation"]["accuracy"],
            val_auc=info["validation"]["roc_auc"], us_acc=us, us_n=info["data"]["validation_subjects"]["US_News"],
            test_n=info["test"]["n"], test_acc=info["test"]["accuracy"],
            orig_acc=info["original_validation_accuracy"], train=info["data"]["train"])
        files = {p.name: p for p in src.iterdir() if p.is_file()}
        files["confusion_matrix.png"] = REPORTS / key / "confusion_matrix.png"
        size = sum(p.stat().st_size for p in files.values()) / 1e6
        print(f"{repo}: {sorted(files)} ({size:.0f} MB)")
        if args.dry_run:
            (REPORTS / key / "MODEL_CARD_PREVIEW.md").write_text(card, encoding="utf-8")
            continue
        tmp = Path(tempfile.mkdtemp(prefix="hub_"))
        for name, p in files.items():
            shutil.copy2(p, tmp / name)
        (tmp / "README.md").write_text(card, encoding="utf-8")
        api.create_repo(repo, repo_type="model", exist_ok=True)
        api.upload_folder(repo_id=repo, folder_path=str(tmp),
                          commit_message="Upload model from the fake news classifier project")
        shutil.rmtree(tmp, ignore_errors=True)
        print(f"  -> https://huggingface.co/{repo}")


if __name__ == "__main__":
    main()
