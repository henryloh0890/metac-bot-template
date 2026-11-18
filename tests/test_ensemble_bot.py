"""Offline tests: no network, no API keys. Run from the repo root with
    uv run --with pytest pytest tests -q
"""

import asyncio
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
os.environ.setdefault("OPENROUTER_API_KEY", "test-key")
os.environ.setdefault("METACULUS_TOKEN", "test-token")

import ensemble_bot  # noqa: E402
from ensemble_bot import EnsembleBot, clip_binary, make_llm, models_from_env  # noqa: E402
from forecasting_tools import BinaryPrediction, BinaryQuestion, GeneralLlm  # noqa: E402

MODELS = ["openrouter/a/model-a", "openrouter/b/model-b", "openrouter/c/model-c"]
FAKE_PROB = {"openrouter/a/model-a": 0.20, "openrouter/b/model-b": 0.30, "openrouter/c/model-c": 0.90}


def make_question() -> BinaryQuestion:
    return BinaryQuestion(
        question_text="Will X happen before 2026-11-30?",
        background_info="Some background.",
        resolution_criteria="Resolves YES if X happens.",
        fine_print="",
        page_url="https://www.metaculus.com/questions/1/",
        id_of_question=1,
        id_of_post=1,
    )


def make_bot(monkeypatch, models=MODELS, runs=6) -> EnsembleBot:
    async def fake_invoke(self, prompt):
        if "Probability" in prompt and self.model in FAKE_PROB:
            return f"reasoning... Probability: {round(FAKE_PROB[self.model] * 100)}%"
        return "fake research"

    async def fake_structure_output(text, output_type, **kwargs):
        pct = float(text.rsplit("Probability:", 1)[1].strip().rstrip("%"))
        return BinaryPrediction(prediction_in_decimal=pct / 100)

    monkeypatch.setattr(GeneralLlm, "invoke", fake_invoke)
    monkeypatch.setattr(ensemble_bot, "structure_output", fake_structure_output)
    return EnsembleBot(
        research_reports_per_question=1,
        predictions_per_research_report=runs,
        publish_reports_to_metaculus=False,
        skip_previously_forecasted_questions=False,
        forecaster_models=models,
        llms={"researcher": GeneralLlm(model="openrouter/r/researcher"),
              "parser": GeneralLlm(model="openrouter/p/parser"),
              "summarizer": GeneralLlm(model="openrouter/s/summarizer")},
    )


def test_clip_binary():
    assert clip_binary(0.0) == 0.02
    assert clip_binary(1.0) == 0.98
    assert clip_binary(0.4) == 0.4


def test_models_from_env(monkeypatch):
    monkeypatch.delenv("BOT_MODELS", raising=False)
    assert models_from_env() == ensemble_bot.DEFAULT_MODELS
    monkeypatch.setenv("BOT_MODELS", " a/x , b/y ,")
    assert models_from_env() == ["a/x", "b/y"]


def test_reasoning_models_get_no_temperature():
    assert make_llm("openrouter/openai/gpt-5").litellm_kwargs.get("temperature") is None
    assert make_llm("openrouter/anthropic/claude-sonnet-4.5").litellm_kwargs.get("temperature") == 0.3


def test_default_llm_rotates_across_panel(monkeypatch):
    bot = make_bot(monkeypatch)
    seen = [bot.get_llm("default", "llm").model for _ in range(6)]
    assert seen == MODELS * 2
    assert bot.get_llm("parser", "llm").model == "openrouter/p/parser"


def test_full_pipeline_uses_every_model_and_takes_median(monkeypatch):
    bot = make_bot(monkeypatch)
    report = asyncio.run(bot.forecast_question(make_question()))
    # 6 runs: 0.2, 0.3, 0.9 twice each -> median 0.3
    assert abs(report.prediction - 0.30) < 1e-9
    text = report.explanation
    for m in MODELS:
        assert f"[model: {m}]" in text


def test_prompt_contains_key_steps(monkeypatch):
    bot = make_bot(monkeypatch)
    prompt = bot.build_binary_prompt(make_question(), "NEWS")
    for needle in ["Reference class", "Status quo", "NEWS", "Probability: ZZ%"]:
        assert needle in prompt


# ---------------------------------------------------------------- model list

CATALOG = [
    {"id": "anthropic/claude-sonnet-5.5", "created": 300, "pricing": {"prompt": "0.000003", "completion": "0.000015"}},
    {"id": "anthropic/claude-opus-5.5", "created": 310, "pricing": {"prompt": "0.000015", "completion": "0.000075"}},
    {"id": "anthropic/claude-haiku-5.5", "created": 305, "pricing": {"prompt": "0.000001", "completion": "0.000005"}},
    {"id": "anthropic/claude-sonnet-5.5:batch", "created": 301, "pricing": {"prompt": "0.0000015", "completion": "0.0000075"}},
    {"id": "openai/gpt-6-luna", "created": 290, "pricing": {"prompt": "0.000002", "completion": "0.000010"}},
    {"id": "openai/gpt-6-luna-mini", "created": 295, "pricing": {"prompt": "0.0000004", "completion": "0.0000016"}},
    {"id": "google/gemini-4-pro", "created": 280, "pricing": {"prompt": "0.000002", "completion": "0.000012"}},
    {"id": "google/gemini-4-flash", "created": 285, "pricing": {"prompt": "0.0000003", "completion": "0.0000025"}},
]


def test_resolve_keeps_available_and_replaces_retired():
    requested = [
        "openrouter/anthropic/claude-sonnet-5.5",  # still served
        "openrouter/openai/gpt-5",  # retired
        "openrouter/google/gemini-2.5-pro",  # retired
    ]
    assert ensemble_bot.resolve_models(requested, CATALOG) == [
        "openrouter/anthropic/claude-sonnet-5.5",
        "openrouter/openai/gpt-6-luna",
        "openrouter/google/gemini-4-pro",
    ]


def test_replacement_skips_expensive_small_and_batch_variants():
    # opus is newest but too expensive; haiku/:batch are not main models
    assert ensemble_bot.best_in_family("anthropic", CATALOG) == "anthropic/claude-sonnet-5.5"
    assert ensemble_bot.best_in_family("anthropic", CATALOG, cheap=True) == "anthropic/claude-haiku-5.5"


def test_resolve_without_catalog_keeps_request():
    assert ensemble_bot.resolve_models(["openrouter/x/y"], None) == ["openrouter/x/y"]


def test_helper_llms_never_use_retired_template_defaults(monkeypatch):
    for k in ("ASKNEWS_CLIENT_ID", "ASKNEWS_SECRET", "ASKNEWS_API_KEY"):
        monkeypatch.delenv(k, raising=False)
    llms = ensemble_bot.helper_llms(["openrouter/anthropic/claude-sonnet-5.5"], CATALOG)
    assert llms["parser"].model == "openrouter/anthropic/claude-haiku-5.5"
    assert llms["researcher"].model == "openrouter/anthropic/claude-sonnet-5.5:online"
    monkeypatch.setenv("ASKNEWS_CLIENT_ID", "id")
    monkeypatch.setenv("ASKNEWS_SECRET", "s")
    assert ensemble_bot.helper_llms(["openrouter/a/b"], CATALOG)["researcher"] == "asknews/news-summaries"


def test_bot_init_resolves_models_from_live_list(monkeypatch):
    monkeypatch.setattr(ensemble_bot, "fetch_openrouter_models", lambda: CATALOG)
    monkeypatch.delenv("BOT_MODELS", raising=False)
    bot = EnsembleBot(publish_reports_to_metaculus=False)
    panel = [llm.model for llm in bot._forecasters]
    assert "openrouter/openai/gpt-5" not in panel
    assert bot.get_llm("parser", "llm").model == "openrouter/anthropic/claude-haiku-5.5"


def test_failed_research_does_not_kill_the_question(monkeypatch):
    bot = make_bot(monkeypatch)

    async def broken(self, question):
        raise RuntimeError("model retired")

    monkeypatch.setattr(ensemble_bot.FallTemplateBot2026, "run_research", broken)
    report = asyncio.run(bot.forecast_question(make_question()))
    assert abs(report.prediction - 0.30) < 1e-9


# ---------------------------------------------------------------- research aid

def test_research_brief_lists_questions_and_skips_errors(monkeypatch, tmp_path):
    bot = make_bot(monkeypatch)
    report = asyncio.run(bot.forecast_question(make_question()))
    path = tmp_path / "brief.md"
    n = ensemble_bot.write_research_brief([report, RuntimeError("failed one")], str(path))
    text = path.read_text(encoding="utf-8")
    assert n == 1
    assert "Will X happen before 2026-11-30?" in text
    assert "https://www.metaculus.com/questions/1/" in text
    assert "30.0%" in text
    assert "Nothing was posted" in text


def test_research_modes_never_publish(monkeypatch, tmp_path):
    seen = {}

    class FakeClient:
        CURRENT_METACULUS_CUP_ID = "cup"

        def get_all_open_questions_from_tournament(self, tid):
            return [make_question()]

    def fake_bot(**kwargs):
        seen.update(kwargs)
        return make_bot(monkeypatch)

    monkeypatch.setattr(ensemble_bot, "MetaculusClient", FakeClient)
    monkeypatch.setattr(ensemble_bot, "EnsembleBot", fake_bot)
    monkeypatch.setattr(ensemble_bot, "RESEARCH_DIR", str(tmp_path))
    monkeypatch.setattr(sys, "argv", ["ensemble_bot.py", "--mode", "cup_research"])
    ensemble_bot.main()
    assert seen["publish_reports_to_metaculus"] is False
    assert len(list(tmp_path.glob("brief-*.md"))) == 1


def test_every_llm_caps_output_tokens():
    # An uncapped request reserves 64k tokens and fails on a low balance.
    assert make_llm("openrouter/a/b").litellm_kwargs.get("max_tokens") == ensemble_bot.MAX_OUTPUT_TOKENS
    llms = ensemble_bot.helper_llms(["openrouter/anthropic/claude-sonnet-5.5"], CATALOG)
    for name in ("parser", "summarizer"):
        assert llms[name].litellm_kwargs.get("max_tokens") == ensemble_bot.MAX_OUTPUT_TOKENS
