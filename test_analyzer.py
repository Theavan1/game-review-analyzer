from types import SimpleNamespace

import analyzer
from analyzer import BatchLabels, ReviewLabel, build_report, classify_batch


def label(i, cat, sent, topic):
    return ReviewLabel(index=i, category=cat, sentiment=sent, topic=topic)


def test_build_report_counts_problems_and_requests():
    reviews = [{"text": f"r{i}", "score": 3} for i in range(4)]
    labels = [label(0, "bug", "negative", "Crash after update"),
              label(1, "bug", "negative", "crash after update"),
              label(2, "feature_request", "neutral", "offline mode"),
              label(3, "praise", "positive", "fun gameplay")]
    report = build_report("com.test", reviews, labels)
    assert "1. crash after update (2)" in report
    assert "1. offline mode (1)" in report
    assert "fun gameplay" not in report.split("## Top 5 player problems")[1].split("##")[0]
    assert "**negative**: 2 (50%)" in report


PARSED = BatchLabels(labels=[label(0, "bug", "negative", "lag"),
                             label(9, "bug", "negative", "bogus")])


def test_gemini_offsets_indexes_and_drops_invalid(monkeypatch):
    monkeypatch.setattr(analyzer, "PROVIDER", "gemini")
    fake = SimpleNamespace(models=SimpleNamespace(
        generate_content=lambda **kw: SimpleNamespace(parsed=PARSED)))
    out = classify_batch(fake, [{"text": "a", "score": 1}], offset=50)
    assert [l.index for l in out] == [50]


def test_gemini_unparsed_response_is_skipped(monkeypatch):
    monkeypatch.setattr(analyzer, "PROVIDER", "gemini")
    fake = SimpleNamespace(models=SimpleNamespace(
        generate_content=lambda **kw: SimpleNamespace(parsed=None)))
    assert classify_batch(fake, [{"text": "a", "score": 1}], 0) == []


def test_anthropic_path_and_refusal(monkeypatch):
    monkeypatch.setattr(analyzer, "PROVIDER", "anthropic")
    ok = SimpleNamespace(messages=SimpleNamespace(
        parse=lambda **kw: SimpleNamespace(stop_reason="end_turn", parsed_output=PARSED)))
    assert [l.index for l in classify_batch(ok, [{"text": "a", "score": 1}], 0)] == [0]
    refused = SimpleNamespace(messages=SimpleNamespace(
        parse=lambda **kw: SimpleNamespace(stop_reason="refusal", parsed_output=None)))
    assert classify_batch(refused, [{"text": "a", "score": 1}], 0) == []


def test_load_csv(tmp_path):
    p = tmp_path / "r.csv"
    p.write_text("text,score\nGreat game,5\n,3\n", encoding="utf-8")
    assert analyzer.load_csv(str(p)) == [{"text": "Great game", "score": 5}]
