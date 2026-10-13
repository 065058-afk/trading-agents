from __future__ import annotations

import json

import numpy as np
import pandas as pd

from mcp_server import signals


def _market_frames(count: int = 340) -> dict[str, pd.DataFrame]:
    dates = pd.bdate_range(end=pd.Timestamp.now().normalize(), periods=count)
    frames = {}
    for offset, symbol in enumerate(("AAA", "BBB", "CCC", "SPY", "QQQ")):
        rng = np.random.default_rng(500 + offset)
        daily_returns = rng.normal(0.0003 + offset * 0.0001, 0.012, count)
        close = 100.0 * np.exp(np.cumsum(daily_returns))
        frames[symbol] = pd.DataFrame(
            {
                "open": close * 0.999,
                "high": close * 1.01,
                "low": close * 0.99,
                "close": close,
                "volume": rng.integers(100_000, 500_000, count),
            },
            index=dates,
        )
    return frames


def _assert_valid_scores(scores, symbols):
    assert set(scores) == set(symbols)
    assert all(np.isfinite(score) and -1.0 <= score <= 1.0 for score in scores.values())


def test_signals_return_bounded_finite_scores(monkeypatch, tmp_path):
    frames = _market_frames()
    symbols = ["AAA", "BBB", "CCC"]
    monkeypatch.setattr(signals, "history_cache", lambda requested: frames)
    monkeypatch.setattr(signals, "MODEL_DIR", tmp_path / "models")
    monkeypatch.setattr(signals, "LOG_DIR", tmp_path / "logs")
    monkeypatch.setattr(
        signals,
        "_fetch_news_headlines",
        lambda requested, student: {symbol: [f"Headline for {symbol}"] for symbol in requested},
    )
    monkeypatch.setattr(
        signals,
        "_ask_groq_batch",
        lambda headlines_by_symbol: {
            symbol: {
                "sentiment": -0.9 if symbol == "AAA" else 0.25,
                "confidence": 0.9,
                "event_flags": ["fraud"] if symbol == "AAA" else [],
                "veto": symbol == "AAA",
            }
            for symbol in headlines_by_symbol
        },
    )

    as_of = frames["AAA"].index[-1]
    outputs = [
        signals.signal_trend(symbols, as_of=as_of),
        signals.signal_rotation(symbols, as_of=as_of),
        signals.signal_breakout(symbols, as_of=as_of),
        signals.signal_dip(symbols, as_of=as_of),
        signals.signal_ml(symbols, as_of=as_of),
        signals.signal_news_tilt(symbols),
    ]

    for scores in outputs:
        _assert_valid_scores(scores, symbols)
    assert outputs[-1].vetoes == {"AAA"}
    assert abs(outputs[-1]["AAA"]) <= 0.2
    assert list((tmp_path / "models").glob("lightgbm_signal_*.joblib"))
    log_lines = (tmp_path / "logs" / "ml_walk_forward.jsonl").read_text(encoding="utf-8").splitlines()
    fold_entries = [json.loads(line) for line in log_lines if '"fold"' in line]
    assert fold_entries
    assert all("ic" in entry and "hit_rate" in entry for entry in fold_entries)


def test_dry_ml_retrain_can_skip_walk_forward_evaluation(monkeypatch, tmp_path):
    frames = _market_frames()
    symbols = ["AAA", "BBB", "CCC"]
    monkeypatch.setattr(signals, "MODEL_DIR", tmp_path / "models")
    monkeypatch.setattr(signals, "LOG_DIR", tmp_path / "logs")
    monkeypatch.setattr(
        signals,
        "_walk_forward_folds",
        lambda panel: (_ for _ in ()).throw(AssertionError("dry retraining ran walk-forward folds")),
    )

    scores = signals.signal_ml(
        symbols,
        history=frames,
        persist=False,
        walk_forward=False,
    )

    _assert_valid_scores(scores, symbols)
    assert not (tmp_path / "models").exists()


def test_price_signals_and_ml_do_not_use_future_bars(monkeypatch, tmp_path):
    original = _market_frames()
    symbols = ["AAA", "BBB", "CCC"]
    as_of = original["AAA"].index[280]
    modified = {symbol: frame.copy() for symbol, frame in original.items()}
    for frame in modified.values():
        future = frame.index > as_of
        frame.loc[future, "close"] *= 20.0
        frame.loc[future, "high"] *= 20.0
        frame.loc[future, "low"] *= 20.0
        frame.loc[future, "volume"] *= 100

    current = original
    monkeypatch.setattr(signals, "history_cache", lambda requested: current)
    monkeypatch.setattr(signals, "MODEL_DIR", tmp_path / "models")
    monkeypatch.setattr(signals, "LOG_DIR", tmp_path / "logs")
    functions = (
        signals.signal_trend,
        signals.signal_rotation,
        signals.signal_breakout,
        signals.signal_dip,
        signals.signal_ml,
    )
    first = [function(symbols, as_of=as_of) for function in functions]

    current = modified
    second = [function(symbols, as_of=as_of) for function in functions]

    for before, after in zip(first, second):
        assert before == after


def test_rotation_scores_are_cached_for_the_iso_week(monkeypatch, tmp_path):
    frames = _market_frames()
    symbols = ["AAA", "BBB", "CCC"]
    calls = []
    monkeypatch.setattr(signals, "CACHE_DIR", tmp_path)

    def load_history(requested, as_of=None, source=None):
        calls.append(list(requested))
        return frames

    monkeypatch.setattr(signals, "_load_history", load_history)

    first = signals.signal_rotation(symbols)
    second = signals.signal_rotation(symbols)

    assert first == second
    assert len(calls) == 1


def test_earnings_guard_returns_only_requested_symbols(monkeypatch):
    request_args = {}

    class Response:
        def raise_for_status(self):
            return None

        def json(self):
            return {"earningsCalendar": [{"symbol": "AAA"}, {"symbol": "OTHER"}]}

    monkeypatch.setenv("FINNHUB_API_KEY", "test-token")

    def get(url, **kwargs):
        request_args.update(url=url, **kwargs)
        return Response()

    monkeypatch.setattr(signals.requests, "get", get)

    assert signals.earnings_guard(["AAA", "BBB"], 7) == ["AAA"]
    assert "token" not in request_args["url"]
    assert "token" not in request_args["params"]
    assert request_args["headers"]["X-Finnhub-Token"] == "test-token"


def test_alpaca_news_set_payload_is_read():
    article = type("Article", (), {"headline": "Test headline", "symbols": ["AAA"]})()
    response = type("NewsSet", (), {"data": {"news": [article]}})()

    assert signals._alpaca_news_items(response) == [article]


def test_finnhub_auth_rejection_keeps_alpaca_headlines(monkeypatch, caplog):
    from alpaca.data.historical.news import NewsClient

    article = type("Article", (), {"headline": "Alpaca headline", "symbols": ["AAA"]})()
    monkeypatch.setenv("FINNHUB_API_KEY", "rejected-test-token")
    monkeypatch.setenv("ALPACA_KEY_A", "alpaca-key")
    monkeypatch.setenv("ALPACA_SECRET_A", "alpaca-secret")
    monkeypatch.setattr(NewsClient, "get_news", lambda self, request: {"news": [article]})

    class UnauthorizedResponse:
        status_code = 401

        def raise_for_status(self):
            raise AssertionError("auth failures should be handled before HTTPError formatting")

    monkeypatch.setattr(signals.requests, "get", lambda *args, **kwargs: UnauthorizedResponse())

    headlines = signals._fetch_news_headlines(["AAA", "BBB"], "A")

    assert headlines["AAA"] == ["Alpaca headline"]
    assert headlines["BBB"] == []
    assert "rejected-test-token" not in caplog.text
    assert "Alpaca headlines only" in caplog.text


def test_groq_uses_available_configured_model(monkeypatch):
    request_args = {}
    monkeypatch.setenv("GROQ_API_KEY", "test-groq-key")
    monkeypatch.setenv("GROQ_MODEL", "openai/gpt-oss-20b")

    class Response:
        status_code = 200

        def raise_for_status(self):
            return None

        def json(self):
            return {
                "choices": [
                    {
                        "message": {
                            "content": json.dumps(
                                {
                                    "assessments": {
                                        "AAA": {
                                            "sentiment": 0.4,
                                            "confidence": 0.8,
                                            "event_flags": [],
                                            "veto": False,
                                        }
                                    }
                                }
                            )
                        }
                    }
                ]
            }

    def post(url, **kwargs):
        request_args.update(url=url, **kwargs)
        return Response()

    monkeypatch.setattr(signals.requests, "post", post)
    assessment = signals._ask_groq("AAA", ["Positive earnings report", "Older headline"])

    assert request_args["json"]["model"] == "openai/gpt-oss-20b"
    assert request_args["json"]["max_completion_tokens"] == 4096
    assert json.loads(request_args["json"]["messages"][1]["content"])["stocks"]["AAA"] == [
        "Positive earnings report"
    ]
    assert assessment["sentiment"] == 0.4
    assert "test-groq-key" not in request_args["url"]


def test_groq_404_returns_neutral_and_logs_model_problem(monkeypatch, caplog):
    monkeypatch.setenv("GROQ_API_KEY", "test-groq-key")

    class NotFoundResponse:
        status_code = 404

        def raise_for_status(self):
            raise AssertionError("404 should provide the model-specific error")

    monkeypatch.setattr(signals.requests, "post", lambda *args, **kwargs: NotFoundResponse())

    assessment = signals._ask_groq("AAA", [])

    assert assessment == signals._neutral_groq_assessment()
    assert "check GROQ_MODEL" in caplog.text
    assert "test-groq-key" not in caplog.text


def test_news_tilt_uses_one_groq_call_for_all_symbols(monkeypatch):
    symbols = ["AAA", "BBB", "CCC"]
    monkeypatch.setattr(
        signals,
        "_fetch_news_headlines",
        lambda requested, student: {symbol: [f"News for {symbol}"] for symbol in requested},
    )
    calls = []

    def assess(headlines_by_symbol):
        calls.append(headlines_by_symbol)
        return {symbol: signals._neutral_groq_assessment() for symbol in headlines_by_symbol}

    monkeypatch.setattr(signals, "_ask_groq_batch", assess)

    scores = signals.signal_news_tilt(symbols)

    assert len(calls) == 1
    assert set(calls[0]) == set(symbols)
    assert scores == {symbol: 0.0 for symbol in symbols}


def test_groq_429_returns_neutral_assessment_per_symbol(monkeypatch, caplog):
    monkeypatch.setenv("GROQ_API_KEY", "test-groq-key")

    class RateLimitedResponse:
        status_code = 429

    monkeypatch.setattr(signals.requests, "post", lambda *args, **kwargs: RateLimitedResponse())
    assessments = signals._ask_groq_batch({"AAA": ["Headline"], "BBB": []})

    assert assessments == {
        "AAA": signals._neutral_groq_assessment(),
        "BBB": signals._neutral_groq_assessment(),
    }
    assert "test-groq-key" not in caplog.text


def test_groq_400_returns_neutral_and_logs_sanitized_reason(monkeypatch, caplog):
    monkeypatch.setenv("GROQ_API_KEY", "test-groq-key")

    class BadRequestResponse:
        status_code = 400

    monkeypatch.setattr(signals.requests, "post", lambda *args, **kwargs: BadRequestResponse())
    assessments = signals._ask_groq_batch({"AAA": ["Headline"]})

    assert assessments["AAA"] == signals._neutral_groq_assessment()
    assert "rejected the request (HTTP 400)" in caplog.text
    assert "test-groq-key" not in caplog.text


def test_groq_network_failure_returns_neutral_assessment(monkeypatch, caplog):
    monkeypatch.setenv("GROQ_API_KEY", "test-groq-key")

    def fail_request(*args, **kwargs):
        raise signals.requests.Timeout("private request details")

    monkeypatch.setattr(signals.requests, "post", fail_request)
    assessments = signals._ask_groq_batch({"AAA": ["Headline"]})

    assert assessments["AAA"] == signals._neutral_groq_assessment()
    assert "request failed (Timeout)" in caplog.text
    assert "private request details" not in caplog.text


def test_unexpected_groq_http_status_returns_neutral_assessment(monkeypatch, caplog):
    monkeypatch.setenv("GROQ_API_KEY", "test-groq-key")

    class UnexpectedResponse:
        status_code = 418

    monkeypatch.setattr(signals.requests, "post", lambda *args, **kwargs: UnexpectedResponse())
    assessments = signals._ask_groq_batch({"AAA": []})

    assert assessments["AAA"] == signals._neutral_groq_assessment()
    assert "request failed (HTTP 418)" in caplog.text
    assert "test-groq-key" not in caplog.text


def test_malformed_groq_score_is_neutral():
    assert signals._normalize_groq_assessment({"sentiment": "not-a-number"}) == signals._neutral_groq_assessment()


def test_earnings_guard_raises_sanitized_finnhub_auth_error(monkeypatch):
    monkeypatch.setenv("FINNHUB_API_KEY", "rejected-test-token")

    class UnauthorizedResponse:
        status_code = 401

        def raise_for_status(self):
            raise AssertionError("auth failures should be handled before HTTPError formatting")

    monkeypatch.setattr(signals.requests, "get", lambda *args, **kwargs: UnauthorizedResponse())

    try:
        signals.earnings_guard(["AAA"], 7)
        assert False, "Expected FinnhubAuthenticationError"
    except signals.FinnhubAuthenticationError as error:
        assert "verify the key" in str(error)
        assert "rejected-test-token" not in str(error)