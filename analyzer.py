"""Game Review Analyzer: fetches Google Play reviews, classifies them with Claude,
and produces a Markdown report of the top player issues and requests."""
from __future__ import annotations

import argparse
import csv
import os
import time
from collections import Counter
from datetime import date
from typing import Literal

import anthropic
from pydantic import BaseModel, Field

def _load_dotenv(path: str = os.path.join(os.path.dirname(__file__), ".env")) -> None:
    """Load KEY=VALUE lines from a local .env file (gitignored) into the environment."""
    if os.path.exists(path):
        for line in open(path, encoding="utf-8"):
            if "=" in line and not line.lstrip().startswith("#"):
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip().strip('"'))


_load_dotenv()

PROVIDER = os.environ.get("LLM_PROVIDER", "gemini")  # "gemini" (free tier) or "anthropic"
MODEL = os.environ.get(
    "ANALYZER_MODEL", "gemini-3.5-flash-lite" if PROVIDER == "gemini" else "claude-opus-5-5")

Category = Literal["bug", "performance", "monetization", "difficulty",
                   "feature_request", "praise", "other"]


class ReviewLabel(BaseModel):
    index: int = Field(description="Index of the review in the input list")
    category: Category
    sentiment: Literal["positive", "neutral", "negative"]
    topic: str = Field(description="Short 2-5 word English topic, e.g. 'crash after update'")


class BatchLabels(BaseModel):
    labels: list[ReviewLabel]


SYSTEM = (
    "You label mobile game reviews for a game studio's product team. "
    "For every review return exactly one label with its index. "
    "Categories: bug (crashes, freezes, broken features), performance (lag, battery, loading), "
    "monetization (ads, prices, pay-to-win), difficulty (levels too hard/easy, balance), "
    "feature_request (asks for something new), praise (positive with no complaint), other. "
    "Topics must be short, in English, and reuse the same wording for the same issue "
    "so they can be counted."
)


def fetch_reviews(app_id: str, count: int, lang: str = "en", country: str = "us") -> list[dict]:
    """Fetch the newest reviews from Google Play."""
    from google_play_scraper import Sort, reviews

    result, _ = reviews(app_id, lang=lang, country=country, sort=Sort.NEWEST, count=count)
    return [{"text": r["content"], "score": r["score"]} for r in result if r.get("content")]


def load_csv(path: str) -> list[dict]:
    """Load reviews from a CSV with columns: text, score."""
    with open(path, encoding="utf-8") as f:
        return [{"text": row["text"], "score": int(row.get("score") or 0)}
                for row in csv.DictReader(f) if row.get("text")]


def _prompt(batch: list[dict]) -> str:
    lines = "\n".join(f"[{i}] ({r['score']}/5) {r['text']}" for i, r in enumerate(batch))
    return f"Label these reviews:\n{lines}"


def _label_with_anthropic(client, batch: list[dict]) -> BatchLabels | None:
    response = client.messages.parse(
        model=MODEL,
        max_tokens=16000,
        output_config={"effort": "low"},
        system=SYSTEM,
        messages=[{"role": "user", "content": _prompt(batch)}],
        output_format=BatchLabels,
    )
    if response.stop_reason == "refusal":
        return None
    return response.parsed_output


def _label_with_gemini(client, batch: list[dict]) -> BatchLabels | None:
    from google.genai import errors, types

    config = types.GenerateContentConfig(
        response_mime_type="application/json", response_schema=BatchLabels)
    for attempt in range(4):
        try:
            response = client.models.generate_content(
                model=MODEL, contents=f"{SYSTEM}\n\n{_prompt(batch)}", config=config)
            return response.parsed
        except errors.ServerError as e:  # 503 "high demand" on the free tier is temporary
            if attempt == 3:
                print(f"  server still busy ({e.code}); skipping this batch")
                return None
            wait = 15 * (attempt + 1)
            print(f"  server busy ({e.code}), retrying in {wait}s...")
            time.sleep(wait)


def make_client():
    if PROVIDER == "gemini":
        from google import genai
        return genai.Client()  # reads GEMINI_API_KEY
    return anthropic.Anthropic()  # reads ANTHROPIC_API_KEY


def classify_batch(client, batch: list[dict], offset: int) -> list[ReviewLabel]:
    """Ask the LLM to label one batch of reviews; returns labels with global indexes."""
    label_fn = _label_with_gemini if PROVIDER == "gemini" else _label_with_anthropic
    parsed = label_fn(client, batch)
    if parsed is None:
        print(f"  ! batch at {offset} was not labeled")
        return []
    return [l.model_copy(update={"index": l.index + offset})
            for l in parsed.labels if 0 <= l.index < len(batch)]


def build_report(app_id: str, reviews: list[dict], labels: list[ReviewLabel]) -> str:
    """Turn labels into a Markdown report. Pure function, easy to test."""
    total = len(labels)
    cats = Counter(l.category for l in labels)
    sents = Counter(l.sentiment for l in labels)
    problems = Counter(l.topic.lower() for l in labels
                       if l.category not in ("praise", "feature_request"))
    requests = Counter(l.topic.lower() for l in labels if l.category == "feature_request")
    pct = lambda n: f"{(100 * n / total):.0f}%" if total else "0%"

    out = [f"# Review Report: {app_id}", "",
           f"_{date.today().isoformat()} · {total} of {len(reviews)} reviews labeled · model: {MODEL}_", "",
           "## Sentiment", ""]
    out += [f"- **{s}**: {sents[s]} ({pct(sents[s])})" for s in ("positive", "neutral", "negative")]
    out += ["", "## Categories", "", "| Category | Count | Share |", "|---|---|---|"]
    out += [f"| {c} | {n} | {pct(n)} |" for c, n in cats.most_common()]
    out += ["", "## Top 5 player problems", ""]
    out += [f"{i}. {t} ({n})" for i, (t, n) in enumerate(problems.most_common(5), 1)] or ["_none_"]
    out += ["", "## Top 5 feature requests", ""]
    out += [f"{i}. {t} ({n})" for i, (t, n) in enumerate(requests.most_common(5), 1)] or ["_none_"]
    return "\n".join(out) + "\n"


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--app-id", help="Google Play package, e.g. com.example.game")
    src.add_argument("--csv", help="CSV file with columns text,score")
    p.add_argument("--count", type=int, default=100)
    p.add_argument("--batch-size", type=int, default=25)
    p.add_argument("--out", default="report.md")
    p.add_argument("--delay", type=float, default=7.0,
                   help="Seconds between LLM calls (free tiers limit requests per minute)")
    args = p.parse_args()

    reviews = fetch_reviews(args.app_id, args.count) if args.app_id else load_csv(args.csv)
    print(f"Loaded {len(reviews)} reviews")

    client = make_client()
    labels: list[ReviewLabel] = []
    for start in range(0, len(reviews), args.batch_size):
        if start and args.delay:
            time.sleep(args.delay)  # stay under free-tier requests-per-minute limits
        labels += classify_batch(client, reviews[start:start + args.batch_size], start)
        print(f"  labeled {len(labels)}/{len(reviews)}")

    report = build_report(args.app_id or args.csv, reviews, labels)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        f.write(report)
    print(f"Report written to {args.out}")


if __name__ == "__main__":
    main()
