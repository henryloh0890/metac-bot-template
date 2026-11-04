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
