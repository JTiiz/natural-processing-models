"""Write data/examples.json: the articles offered in the app.

    uv run python training/examples.py

All but the last come from validation_data.csv, which no model is trained on. The last one
is invented for the demo (fictional country and people): a nonsense story written in the
style of a news agency, to show that the models judge the style of an article, not its facts.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from fakenews import data as D  # noqa: E402

PICKS = [  # (title in validation_data.csv, short label shown in the app)
    ("Hurricane Irma moves away from Barbuda: NHC", "Hurricane report"),
    ("Russian military chief meets NATO General to soothe war games fears: Ifax", "NATO meeting"),
    ("A PICTURE IS WORTH A THOUSAND WORDS: A Lone Socialist Takes This Ironic Message To Trump", "Protest sign"),
    ("North Korea: Will World War III Kick Off This Week?", "World War III?"),
    ("Trump Revives Keystone and Dakota Access Pipelines", "Pipelines"),
]
MADE_UP = {
    "short": "Invented story, agency style",
    "title": "Eldoria to run public clocks backwards on Sundays to save energy, ministry says",
    "text": (
        "VALMORA (Eldoria) - The government of Eldoria said on Wednesday that all public clocks would run "
        "backwards on Sundays from next month, a measure it said would cut electricity use by 12 percent. "
        "\"The plan has been studied carefully by our engineers,\" Energy Minister Lena Varga told reporters in "
        "the capital. A ministry spokesman said in a statement that the change would apply to railway stations, "
        "schools and government buildings, but not to hospitals. Opposition lawmakers said on Thursday they would "
        "challenge the decision in parliament, and a poll published on Tuesday showed that 41 percent of voters "
        "supported it. The ministry declined to comment on how the savings had been calculated."),
    "subject": "invented for this demo",
    "label": None,
}


def main() -> None:
    val = D.load_validation().drop_duplicates("title").set_index("title")
    examples = []
    for title, short in PICKS:
        row = val.loc[title]
        examples.append({"short": short, "title": title, "text": row["text"].strip(),
                         "subject": row["subject"], "label": int(row["label"])})
    examples.append(MADE_UP)
    out = ROOT / "data" / "examples.json"
    out.write_text(json.dumps(examples, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"{len(examples)} examples -> {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
