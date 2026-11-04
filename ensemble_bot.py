"""
Ensemble forecasting bot for the Metaculus FutureEval bot tournaments.

Built on top of the official Metaculus template (github.com/Metaculus/metac-bot-template).
Put this file in the root of your fork, next to main.py, and point the GitHub
workflows at it (see SETUP.md).

What it changes compared with the template:
1. Model ensemble: each question gets several forecasts, rotated across
   different model families (Claude, GPT, Gemini by default). The template
   uses one model five times. Different families make different mistakes,
   so the median of a mixed panel is usually better calibrated.
2. A stronger binary prompt: reference class and base rate first, then a
   strict reading of the resolution criteria, then the status quo, then
   the news. MiniBench questions are written by an AI and resolve within
   about two weeks, so the prompt reminds the model how little time is left.
3. Safer extremes: binary forecasts are clipped to 2%..98% (template: 1%..99%).
   With a log score, one confident miss costs more than many small gains.
4. A --dry-run flag that forecasts without posting anything to Metaculus.

Settings (GitHub repository variables or .env):
  BOT_MODELS   comma-separated litellm model names for the forecasting panel.
               Default: three OpenRouter models, see DEFAULT_MODELS.
  BOT_RUNS     total forecasts per question (default 6, i.e. 2 per model).
"""

from __future__ import annotations

import argparse
import asyncio
import itertools
import logging
import os
from datetime import datetime
from typing import Literal

from main import FallTemplateBot2026  # the official template class

from bot_helpers import (
    check_environment,
    print_run_summary_banner,
    print_startup_banner,
)
from forecasting_tools import (
    BinaryPrediction,
    BinaryQuestion,
    GeneralLlm,
    MetaculusClient,
    ReasonedPrediction,
    clean_indents,
    structure_output,
)

logger = logging.getLogger(__name__)

# Paid from the free Metaculus/OpenRouter credits (they cover OpenAI, Anthropic
# and Google only). Swap in newer model names with the BOT_MODELS variable.
DEFAULT_MODELS = [
    "openrouter/anthropic/claude-sonnet-4.5",
    "openrouter/openai/gpt-5",
    "openrouter/google/gemini-2.5-pro",
]

BINARY_MIN = 0.02
BINARY_MAX = 0.98

# Reasoning models (GPT-5, o-series) reject a custom temperature.
_NO_TEMPERATURE_MARKERS = ("gpt-5", "/o1", "/o3", "/o4")


def make_llm(model: str) -> GeneralLlm:
    kwargs: dict = {"model": model, "timeout": 300, "allowed_tries": 2}
    if not any(marker in model for marker in _NO_TEMPERATURE_MARKERS):
        kwargs["temperature"] = 0.3
    return GeneralLlm(**kwargs)


def models_from_env() -> list[str]:
    raw = os.getenv("BOT_MODELS", "").strip()
    if not raw:
        return list(DEFAULT_MODELS)
    return [m.strip() for m in raw.split(",") if m.strip()]


def clip_binary(p: float) -> float:
    return max(BINARY_MIN, min(BINARY_MAX, p))


class EnsembleBot(FallTemplateBot2026):
    """Template bot with a rotating multi-model forecasting panel."""

    def __init__(self, *args, forecaster_models: list[str] | None = None, **kwargs):
        super().__init__(*args, **kwargs)
        models = forecaster_models or models_from_env()
        self._forecasters = [make_llm(m) for m in models]
        self._forecaster_cycle = itertools.cycle(self._forecasters)
        logger.info(f"Forecasting panel: {[llm.model for llm in self._forecasters]}")

    def get_llm(self, purpose="default", guarantee_type=None):  # type: ignore[override]
        # Every forecast call asks for the "default" llm; hand out the next
        # panel member each time so the runs spread across model families.
        if purpose == "default" and self._forecasters:
            llm = next(self._forecaster_cycle)
            return llm.model if guarantee_type == "string_name" else llm
        return super().get_llm(purpose, guarantee_type)

    # ------------------------------------------------------------------ binary

    def build_binary_prompt(self, question: BinaryQuestion, research: str) -> str:
        today = datetime.now().strftime("%Y-%m-%d")
        close = getattr(question, "scheduled_resolution_time", None)
        close_text = close.strftime("%Y-%m-%d") if close else "unknown"
        return clean_indents(
            f"""
            You are an experienced superforecaster. Your forecast is scored with a
            log score, so be calibrated: confident only when the evidence is strong.

            Question:
            {question.question_text}

            Background:
            {question.background_info}

            Resolution criteria (these have not yet been met):
            {question.resolution_criteria}

            {question.fine_print}

            News and research gathered for you:
            {research}

            Today is {today}. The question is scheduled to resolve on {close_text}.

            Work through these steps in order and write each one out:
            1. Restate exactly what must happen for YES, and by when. Note any
               wording that makes YES harder or easier than it first looks.
            2. Reference class: name one or two comparable situations and give
               an approximate base rate for this kind of event in a window this long.
            3. Status quo: what happens if nothing changes? With little time
               left, the status quo usually wins.
            4. Evidence: which items in the research move you away from the
               base rate, in which direction, and how much? Ignore items that
               are old, off-topic or unsourced.
            5. Write the strongest short case for YES and for NO.
            6. Give your final probability. Avoid values below 3% or above 97%
               unless the outcome is nearly settled already.
            {self._get_conditional_disclaimer_if_necessary(question)}

            The last thing you write is your final answer as: "Probability: ZZ%", 0-100
            """
        )

    async def _run_forecast_on_binary(
        self, question: BinaryQuestion, research: str
    ) -> ReasonedPrediction[float]:
        prompt = self.build_binary_prompt(question, research)
        llm = self.get_llm("default", "llm")
        reasoning = await llm.invoke(prompt)
        parsed: BinaryPrediction = await structure_output(
            reasoning,
            BinaryPrediction,
            model=self.get_llm("parser", "llm"),
            num_validation_samples=self._structure_output_validation_samples,
            additional_instructions=(
                "The text gives a probability forecast for a binary question. "
                + self._create_resolved_question_parsing_message()
            ),
        )
        p = clip_binary(parsed.prediction_in_decimal)
        logger.info(f"{llm.model} forecast {p:.3f} on {question.page_url}")
        return ReasonedPrediction(
            prediction_value=p, reasoning=f"[model: {llm.model}]\n{reasoning}"
        )


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    parser = argparse.ArgumentParser(description="Run the ensemble forecasting bot")
    parser.add_argument(
        "--mode",
        choices=["tournament", "minibench", "test_questions"],
        default="tournament",
        help="tournament = Fall FutureEval + MiniBench; minibench = MiniBench only",
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="forecast but do not post to Metaculus"
    )
    args = parser.parse_args()
    mode: Literal["tournament", "minibench", "test_questions"] = args.mode
    publish = not args.dry_run

    check_environment(strict=True)
    print_startup_banner(mode, will_publish=publish)

    # GitHub passes an empty string when the variable is not set.
    runs = int(os.getenv("BOT_RUNS") or "6")
    bot = EnsembleBot(
        research_reports_per_question=1,
        predictions_per_research_report=runs,
        use_research_summary_to_forecast=False,
        publish_reports_to_metaculus=publish,
        folder_to_save_reports_to=None,
        skip_previously_forecasted_questions=True,
        extra_metadata_in_explanation=True,
    )

    client = MetaculusClient()
    targets = {
        "tournament": [client.CURRENT_AI_COMPETITION_ID, client.CURRENT_MINIBENCH_ID],
        "minibench": [client.CURRENT_MINIBENCH_ID],
        "test_questions": ["bot-testing-area"],
    }[mode]
    if mode == "test_questions":
        bot.skip_previously_forecasted_questions = False

    reports = []
    for target in targets:
        reports += asyncio.run(bot.forecast_on_tournament(target, return_exceptions=True))

    bot.log_report_summary(reports)
    urls = {
        "tournament": "https://www.metaculus.com/tournament/fall-futureeval-2026/",
        "minibench": "https://www.metaculus.com/tournament/minibench/",
        "test_questions": "https://www.metaculus.com/tournament/bot-testing-area/",
    }
    print_run_summary_banner(reports, will_publish=publish, tournament_url=urls[mode])


if __name__ == "__main__":
    main()
