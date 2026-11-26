# Game Review Analyzer 🎮🤖

An LLM-powered automation that turns hundreds of Google Play reviews into a short, actionable report for a game team: **what players complain about most, and what they ask for.**

Reading reviews by hand is slow and repetitive. This tool:

1. **Fetches** the newest reviews of any Android game from Google Play (or loads them from a CSV).
2. **Classifies** each review with an LLM (Gemini or Claude): category (bug, performance, monetization, difficulty, feature request, praise), sentiment, and a short normalized topic.
3. **Aggregates** the labels into a Markdown report: sentiment split, category table, top 5 problems and top 5 feature requests.

## Design decisions

- **Structured outputs:** responses are validated against a Pydantic schema (`messages.parse`), so the pipeline never breaks on malformed JSON.
- **Batching:** reviews are sent 25 at a time. That means fewer API calls, lower cost, and more consistent topics within a batch.
- **Normalized topics:** the prompt asks the model to reuse the same wording for the same issue, so topics can be counted.
- **Verification over trust:** invalid indexes are dropped and refusals are handled. The aggregation logic is a pure function covered by unit tests, so the LLM is mocked in tests.

## Usage

Works with **Google Gemini (free tier, default)** or **Anthropic Claude**.

```bash
pip install -r requirements.txt

# Gemini (free key from https://aistudio.google.com/apikey)
export GEMINI_API_KEY=...           # PowerShell: $env:GEMINI_API_KEY="..."
python analyzer.py --app-id com.example.game --count 200 --out report.md

# Claude
export LLM_PROVIDER=anthropic ANTHROPIC_API_KEY=...
python analyzer.py --csv sample_reviews.csv
```

- `ANALYZER_MODEL` picks the model. Defaults: `gemini-3.5-flash-lite` and `claude-opus-5-5`.
- `--delay` (7 seconds by default) spaces out requests to stay within free-tier per-minute limits.

## Tests

```bash
pytest -q
```

## Possible next steps
- Weekly scheduled run with a diff against last week ("new issues this week")
- Post the report to Slack
- Simple dashboard with trend charts
