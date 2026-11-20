"""Data loading and text cleaning shared by training and the app.

The articles are the ISOT Fake News dataset (University of Victoria): real articles
from Reuters and fake articles from sites flagged by PolitiFact and Wikipedia, 2015-2018.
The Ironhack project gave it as two files:

  data.csv             39,942 labelled articles (0 = fake, 1 = real)
  validation_data.csv   4,956 articles to label (label column = 2), kept in data/

data.csv is not redistributed here; the same articles are read from a public mirror of
the full ISOT dataset on the Hugging Face Hub, pinned to a fixed revision. The mirror's
`target` column is inverted (1 = fake), so it is flipped to the project's convention.
"""
from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
VALIDATION_CSV = ROOT / "data" / "validation_data.csv"

DATASET_REPO = "haoxianc/gilchr_fake-and-real-news-dataset"
DATASET_REVISION = "a067c985b7ddab9b2f8c16c83e9763b67d20b3cf"
DATASET_FILE = "fake_real_news.csv"

LABELS = {0: "Fake", 1: "Real"}
# In ISOT the label follows the source: Reuters sections are real, the rest fake.
REAL_SUBJECTS = {"politicsNews", "worldnews"}

# --- cleaning --------------------------------------------------------------
# Every real article starts with a Reuters dateline ("WASHINGTON (Reuters) - ") and
# almost no fake article does, so the word "reuters" alone separates the classes.
# The fake sites end their posts with image credits ("Featured image via Getty Images")
# and embedded tweets. These markers say where an article was published, not what it
# says, so the "clean" setup removes them before the text reaches a model.
_DATELINE = re.compile(r"^.{0,120}?\(\s*reuters\s*\)\s*-\s*", re.IGNORECASE)
_URL = re.compile(r"https?://\S+|www\.\S+|pic\.twitter\.com/\S+|\S+\.com/\S*", re.IGNORECASE)
_HANDLE = re.compile(r"@\w+")
_SOURCE_PHRASES = re.compile(
    r"featured image via|featured image|image via|photo by|getty images|21st century wire|"
    r"\(\s*reuters\s*\)|reuters", re.IGNORECASE)
_NON_LETTERS = re.compile(r"[^a-z\s]")
_SPACES = re.compile(r"\s+")


def basic_clean(text: str) -> str:
    """The notebook's cleaning: lower case, letters only."""
    text = _NON_LETTERS.sub(" ", str(text).lower())
    return _SPACES.sub(" ", text).strip()


def clean_article(title: str, text: str) -> str:
    """Title + body without the publisher markers described above, then `basic_clean`."""
    body = _DATELINE.sub("", str(text).strip())
    joined = f"{title} {body}"
    joined = _URL.sub(" ", joined)
    joined = _HANDLE.sub(" ", joined)
    joined = _SOURCE_PHRASES.sub(" ", joined)
    return basic_clean(joined)


# --- loading ---------------------------------------------------------------

def download_corpus(cache_dir: str | Path | None = None) -> Path:
    """Local path to the full ISOT CSV (116 MB, downloaded once, then cached)."""
    from huggingface_hub import hf_hub_download

    return Path(hf_hub_download(repo_id=DATASET_REPO, repo_type="dataset", filename=DATASET_FILE,
                                revision=DATASET_REVISION, cache_dir=cache_dir))


def _key(df: pd.DataFrame) -> pd.Series:
    return df["title"].astype(str).str.strip() + "|" + df["text"].astype(str).str.strip()


def load_validation(path: str | Path = VALIDATION_CSV) -> pd.DataFrame:
    """validation_data.csv with its true label recovered from the subject."""
    val = pd.read_csv(path)
    val["label"] = val["subject"].isin(REAL_SUBJECTS).astype(int)
    return val


def load_training(corpus_csv: str | Path, validation: pd.DataFrame) -> pd.DataFrame:
    """Every ISOT article that is not in validation_data.csv (the project's data.csv)."""
    df = pd.read_csv(corpus_csv)
    df["label"] = 1 - df["target"].astype(int)  # mirror: 1 = fake -> project: 1 = real
    assert (df["label"] == df["subject"].isin(REAL_SUBJECTS)).all(), "unexpected labels in mirror"
    df = df[~_key(df).isin(set(_key(validation)))]
    return df[["label", "title", "text", "subject", "date"]].reset_index(drop=True)
