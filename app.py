"""Fake News Classifier: three scikit-learn text pipelines judge a news article.

    uv run streamlit run app.py
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import joblib
import numpy as np
import plotly.graph_objects as go
import streamlit as st
import yaml

from fakenews.data import clean_article

APP_DIR = Path(__file__).resolve().parent
MODELS_ROOT = Path(os.getenv("MODELS_ROOT", APP_DIR / "models"))
REGISTRY = APP_DIR / "models" / "models.yaml"
EXAMPLES = json.loads((APP_DIR / "data" / "examples.json").read_text(encoding="utf-8"))
OWN = "Write your own"
INK, INK2, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#fcfcfb"
FAKE, REAL = "#eb6834", "#2a78d6"

st.set_page_config(page_title="Fake News Classifier", page_icon="📰", layout="wide")
st.markdown("""<style>
.block-container {padding-top: 1.6rem; padding-bottom: 1rem; max-width: 1320px;}
h1 {font-size: 2rem !important; margin-bottom: 0 !important;}
.muted {color: #52514e; font-size: 0.92rem;}
.pill {display:inline-block; padding:0.1rem 0.55rem; margin:0 0.3rem 0.3rem 0; border-radius:999px;
       background:#f3f2ee; color:#52514e; font-size:0.82rem;}
.truth {font-size:0.92rem; color:#52514e; margin-top:-0.4rem;}
</style>""", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Loading (cached for the life of the server)
# ---------------------------------------------------------------------------

def resolve(entry: dict) -> tuple[Path, str]:
    local = MODELS_ROOT / entry["folder"]
    if (local / "model_info.json").exists():
        return local, "local"
    from huggingface_hub import snapshot_download

    return Path(snapshot_download(entry["hub_id"], allow_patterns=["*.json", "*.joblib"])), entry["hub_id"]


@st.cache_resource(show_spinner="Loading the three models (downloaded from Hugging Face the first time)...")
def load_models() -> dict[str, dict]:
    out = {}
    for entry in yaml.safe_load(REGISTRY.read_text(encoding="utf-8"))["models"]:
        path, source = resolve(entry)
        out[entry["key"]] = {**entry, "source": source, "model": joblib.load(path / "model.joblib"),
                             "info": json.loads((path / "model_info.json").read_text(encoding="utf-8"))}
    return out


def term_contributions(pipeline, text: str, n: int = 12) -> list[tuple[str, float]]:
    """Terms of this article with the largest weight × TF-IDF value in the logistic regression."""
    x = pipeline["vec"].transform([text])
    names = pipeline["vec"].get_feature_names_out()
    contrib = x.multiply(pipeline["clf"].coef_[0]).tocsr()
    idx, vals = contrib.indices, contrib.data
    top = np.argsort(-np.abs(vals))[:n]
    return [(names[idx[i]], float(vals[i])) for i in top]


def pick_example() -> None:
    choice = st.session_state["example"]
    ex = next((e for e in EXAMPLES if e["short"] == choice), None)
    st.session_state["title"] = ex["title"] if ex else ""
    st.session_state["body"] = ex["text"] if ex else ""


# ---------------------------------------------------------------------------
# Page
# ---------------------------------------------------------------------------

models = load_models()
first = next(iter(models.values()))
if "example" not in st.session_state:
    st.session_state["example"] = EXAMPLES[0]["short"]
    pick_example()

st.title("Fake News Classifier")
st.markdown(
    "<p class='muted'>Three scikit-learn pipelines trained on "
    f"{first['info']['data']['train']:,} news articles (ISOT dataset, 2015-2018) label an article as fake or "
    "real. They learn how real agency news is written compared with articles from sites known for fake "
    "news. They do not check facts.</p>", unsafe_allow_html=True)

left, right = st.columns([1, 2], gap="large")
with left:
    st.selectbox("Article", [e["short"] for e in EXAMPLES] + [OWN], key="example", on_change=pick_example)
    ex = next((e for e in EXAMPLES if e["short"] == st.session_state["example"]), None)
    if ex and ex["label"] is not None:
        st.markdown(f"<p class='truth'>From the validation set, never seen in training · actual label: "
                    f"<b>{'Real' if ex['label'] == 1 else 'Fake'}</b> ({ex['subject']})</p>", unsafe_allow_html=True)
    elif ex:
        st.markdown("<p class='truth'>Made up for this demo: a nonsense story written like agency news.</p>",
                    unsafe_allow_html=True)
    title = st.text_input("Headline", key="title")
    body = st.text_area("Text", key="body", height=260)
    st.markdown(f"**Models** · accuracy on {first['info']['validation']['n']:,} held-out articles")
    for m in models.values():
        src = ("local copy" if m["source"] == "local" else
               f"<a href='https://huggingface.co/{m['hub_id']}' title='{m['hub_id']}' target='_blank'>"
               "🤗 Hugging Face ↗</a>")
        st.markdown(f"**{m['name']}** {m['info']['validation']['accuracy']:.1%}<br>"
                    f"<span class='muted'>{src}</span>", unsafe_allow_html=True)

with right:
    tab_check, tab_eval = st.tabs(["Verdict", "How reliable are they?"])

    with tab_check:
        text = clean_article(title, body)
        if len(text.split()) < 5:
            st.info("Write or paste a headline and some text (in English) to classify it.")
        else:
            keys = list(models)
            p_fake = [float(models[k]["model"].predict_proba([text])[0, 0]) for k in keys]
            names = [models[k]["name"] for k in keys]
            votes = sum(p >= 0.5 for p in p_fake)
            verdict = ("All three models say **fake**" if votes == 3 else "All three models say **real**"
                       if votes == 0 else f"The models disagree: {votes} of 3 say **fake**")
            st.markdown(f"#### {verdict}")
            # One 100 % bar per model: fake share on the left, real share on the right;
            # the label sits in the larger segment.
            fig = go.Figure()
            for side, probs, color in [("Fake", p_fake, FAKE), ("Real", [1 - p for p in p_fake], REAL)]:
                fig.add_trace(go.Bar(
                    x=probs, y=names, orientation="h", name=side,
                    marker=dict(color=color, line=dict(color=SURFACE, width=2)),
                    text=[f"{side} {p:.0%}" if p >= 0.5 else "" for p in probs],
                    textposition="inside", insidetextanchor="middle", textfont=dict(color="#ffffff", size=13),
                    hovertemplate=f"%{{y}}: %{{x:.1%}} {side.lower()}<extra></extra>"))
            fig.update_layout(
                barmode="stack", height=230, margin=dict(l=10, r=20, t=10, b=10), paper_bgcolor=SURFACE,
                plot_bgcolor=SURFACE, font=dict(color=INK2), bargap=0.4, yaxis=dict(autorange="reversed"),
                legend=dict(orientation="h", y=1.14, x=0, font=dict(color=INK), traceorder="normal"),
                xaxis=dict(title="Probability given by each model (the larger share is its verdict)", range=[0, 1],
                           tickformat=".0%", gridcolor=GRID))
            st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

            st.markdown("**Why?** The words of this article that weigh most in the logistic regression")
            terms = term_contributions(models["tfidf-logreg"]["model"], text)
            if terms:
                terms = sorted(terms, key=lambda t: t[1])
                fig2 = go.Figure(go.Bar(
                    x=[c for _, c in terms], y=[t for t, _ in terms], orientation="h",
                    marker=dict(color=[FAKE if c < 0 else REAL for _, c in terms], cornerradius=3,
                                line=dict(color=SURFACE, width=1)),
                    hovertemplate="%{y}: %{x:+.2f}<extra></extra>"))
                fig2.add_vline(x=0, line=dict(color=MUTED, width=1))
                lim = max(abs(c) for _, c in terms) * 1.1
                fig2.update_layout(
                    height=40 + 22 * len(terms), margin=dict(l=10, r=20, t=10, b=10), paper_bgcolor=SURFACE,
                    plot_bgcolor=SURFACE, font=dict(color=INK2), showlegend=False, bargap=0.25,
                    xaxis=dict(title="← pushes towards fake · pushes towards real →", range=[-lim, lim],
                               gridcolor=GRID, zeroline=False))
                st.plotly_chart(fig2, use_container_width=True, config={"displayModeBar": False})
            with st.expander("Text the models see (after cleaning)"):
                st.caption("Lower case, letters only, without the Reuters dateline, image credits, links and "
                           "@handles, which reveal the publisher rather than the content.")
                st.write(text[:1500] + ("..." if len(text) > 1500 else ""))

    with tab_eval:
        st.markdown(
            f"Each model is scored on `validation_data.csv`: {first['info']['validation']['n']:,} articles that no model saw in training and "
            "that are not copies of a training article. One of its sections (US_News) does not appear in "
            "training at all.")
        res = {k: m["info"]["validation"]["accuracy"] for k, m in models.items()}
        orig = {k: m["info"].get("original_validation_accuracy") for k, m in models.items()}
        fig3 = go.Figure()
        if all(v is not None for v in orig.values()):
            fig3.add_trace(go.Bar(y=[models[k]["name"] for k in models], x=list(orig.values()), orientation="h",
                                  name="Original notebook pipeline", marker=dict(color="#c3c1b8", cornerradius=3),
                                  text=[f"{v:.1%}" for v in orig.values()], textposition="inside",
                                  insidetextanchor="end", textfont=dict(color=INK),
                                  hovertemplate="%{y}: %{x:.1%}<extra>original</extra>"))
        fig3.add_trace(go.Bar(y=[models[k]["name"] for k in models], x=list(res.values()), orientation="h",
                              name="Clean (used in this app)", marker=dict(color=REAL, cornerradius=3),
                              text=[f"{v:.1%}" for v in res.values()], textposition="inside",
                              insidetextanchor="end", textfont=dict(color="#ffffff"),
                              hovertemplate="%{y}: %{x:.1%}<extra>clean</extra>"))
        fig3.update_layout(
            barmode="group", height=300, margin=dict(l=10, r=20, t=10, b=10), paper_bgcolor=SURFACE,
            plot_bgcolor=SURFACE, font=dict(color=INK2), bargap=0.3, bargroupgap=0.08,
            legend=dict(orientation="h", y=1.12, x=0, font=dict(color=INK)), yaxis=dict(autorange="reversed"),
            xaxis=dict(title="Accuracy on the validation set (axis starts at 80 %)", range=[0.8, 1],
                       tickformat=".0%", gridcolor=GRID))
        st.plotly_chart(fig3, use_container_width=True, config={"displayModeBar": False})
        st.markdown(
            "- **The original 99.9 % came from one word.** Every real article starts with a Reuters dateline "
            "(\"WASHINGTON (Reuters) -\") and no fake one does. Without it, the random forest drops from 98.9 % "
            "to 90.3 %; the logistic regression barely moves.\n"
            "- **815 validation articles were copies of training articles**, published again under another "
            "section. They are left out of every score here.\n"
            "- **What is left is style.** Agency news says *said*, *spokesman* and the day of the week; the fake "
            "sites write *video*, *watch*, *just*. Try the invented story: nonsense written like agency news "
            "passes as real.")
