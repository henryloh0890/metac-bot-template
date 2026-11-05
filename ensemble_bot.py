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
import re
from datetime import datetime
from typing import Literal

from main import FallTemplateBot2026  # the official template class

from bot_helpers import (
    check_environment,
    print_run_summary_banner,
    print_startup_banner,
)
import requests
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
# and Google only). Models get retired often, so each name below is checked
# against OpenRouter's live model list at startup; a retired one is replaced by
# the newest affordable model of the same family (see resolve_models).
DEFAULT_MODELS = [
    "openrouter/anthropic/claude-sonnet-4.5",
    "openrouter/openai/gpt-5",
    "openrouter/google/gemini-2.5-pro",
]

OPENROUTER_MODELS_URL = "https://openrouter.ai/api/v1/models"
# Max price per token (USD) for an automatically chosen replacement model.
MAX_PROMPT_PRICE = 5e-6  # $5 per million input tokens
MAX_COMPLETION_PRICE = 2.5e-5  # $25 per million output tokens
# Name parts that mark variants we do not want as a forecaster.
_SKIP_VARIANT = re.compile(
    r"(:|(?<![a-z])(mini|nano|lite|flash|haiku|audio|image|vision|search|oss|"
    r"preview|embed|tts|realtime|codex|instruct|banana|exp)(?![a-z]))",
    re.IGNORECASE,
)
# Small, cheap variants used for parsing and summaries. (?<![a-z]) stops
# "gemini" from counting as "mini".
_SMALL_VARIANT = re.compile(r"(?<![a-z])(mini|haiku|flash)(?![a-z])", re.IGNORECASE)

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


def fetch_openrouter_models() -> list[dict] | None:
    try:
        resp = requests.get(OPENROUTER_MODELS_URL, timeout=30)
        resp.raise_for_status()
        return resp.json()["data"]
    except Exception as e:  # network trouble must not stop the bot
        logger.warning(f"Could not fetch the OpenRouter model list: {e}")
        return None


def _affordable(model: dict) -> bool:
    try:
        pricing = model.get("pricing", {})
        return (
            float(pricing.get("prompt", 1)) <= MAX_PROMPT_PRICE
            and float(pricing.get("completion", 1)) <= MAX_COMPLETION_PRICE
        )
    except (TypeError, ValueError):
        return False


def best_in_family(vendor: str, catalog: list[dict], cheap: bool = False) -> str | None:
    """Newest affordable main model of a vendor, e.g. 'anthropic'.

    cheap=True picks the newest small model instead (for parsing/summaries).
    """
    candidates = []
    for m in catalog:
        mid = m.get("id", "")
        if not mid.startswith(vendor + "/") or not _affordable(m):
            continue
        name = mid.split("/", 1)[1]
        is_small = bool(_SMALL_VARIANT.search(name))
        if cheap:
            if not is_small or ":" in name or re.search(r"audio|image|search|preview|tts", name):
                continue
        elif _SKIP_VARIANT.search(name):
            continue
        candidates.append((m.get("created", 0), mid))
    if not candidates:
        return None
    return max(candidates)[1]


def resolve_models(requested: list[str], catalog: list[dict] | None) -> list[str]:
    """Keep requested models OpenRouter still serves; replace retired ones."""
    if catalog is None:
        return requested
    available = {m.get("id") for m in catalog}
    resolved: list[str] = []
    for model in requested:
        if not model.startswith("openrouter/"):
            resolved.append(model)  # direct provider key, cannot check
            continue
        slug = model.removeprefix("openrouter/")
        if slug in available:
            resolved.append(model)
            continue
        vendor = slug.split("/", 1)[0]
        replacement = best_in_family(vendor, catalog)
        if replacement:
            logger.warning(f"{slug} is no longer on OpenRouter; using {replacement}")
            resolved.append("openrouter/" + replacement)
        else:
            logger.warning(f"{slug} is no longer on OpenRouter and has no replacement; dropped")
    if not resolved:
        raise RuntimeError("No usable forecasting model found on OpenRouter")
    return list(dict.fromkeys(resolved))


def helper_llms(panel: list[str], catalog: list[dict] | None) -> dict:
    """Parser, summarizer and researcher that are known to exist.

    The template's defaults (gpt-4o-mini, gpt-4o-search-preview) have been
    retired on OpenRouter, which broke every question in the first live run.
    """
    cheap = None
    if catalog is not None:
        for vendor in ("anthropic", "google", "openai"):
            cheap = best_in_family(vendor, catalog, cheap=True)
            if cheap:
                cheap = "openrouter/" + cheap
                break
    cheap = cheap or panel[0]
    llms: dict = {"parser": make_llm(cheap), "summarizer": make_llm(cheap)}
    has_asknews = (os.getenv("ASKNEWS_CLIENT_ID") and os.getenv("ASKNEWS_SECRET")) or os.getenv(
        "ASKNEWS_API_KEY"
    )
    if has_asknews:
        llms["researcher"] = "asknews/news-summaries"
    else:
        # OpenRouter's ":online" suffix adds a web search to any model.
        llms["researcher"] = GeneralLlm(model=panel[0] + ":online", temperature=0.1, timeout=300)
    return llms


def clip_binary(p: float) -> float:
    return max(BINARY_MIN, min(BINARY_MAX, p))


class EnsembleBot(FallTemplateBot2026):
    """Template bot with a rotating multi-model forecasting panel."""

    def __init__(self, *args, forecaster_models: list[str] | None = None, **kwargs):
        models = forecaster_models or models_from_env()
        if "llms" not in kwargs:
            catalog = fetch_openrouter_models()
            models = resolve_models(models, catalog)
            kwargs["llms"] = helper_llms(models, catalog)
        super().__init__(*args, **kwargs)
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

    async def run_research(self, question) -> str:
        # A broken search must not cost us the whole question: forecast
        # without news instead (the first live run lost every question this way).
        try:
            return await super().run_research(question)
        except Exception as e:
            logger.warning(f"Research failed for {question.page_url}, forecasting without it: {e}")
            return "No research available. Rely on base rates and the question text."

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


RESEARCH_DIR = "research_reports"


def write_research_brief(reports: list, path: str) -> int:
    """Write one readable Markdown file for a human forecaster.

    Used for human contests (Metaculus Cup and similar) where bots cannot win:
    the bot does the research and gives a starting estimate; the human reads
    it, decides, and enters their own forecast by hand.
    """
    lines = [
        "# Research brief",
        "",
        f"Made {datetime.now().strftime('%Y-%m-%d %H:%M')} UTC. Nothing was posted to Metaculus.",
        "",
        "Contest rules allow AI help, but your forecast must be your own decision.",
        "Read the reasoning, check the sources, then enter your own number.",
        "",
    ]
    count = 0
    for report in reports:
        if isinstance(report, BaseException) or report is None:
            continue
        count += 1
        question = report.question
        try:
            estimate = report.make_readable_prediction(report.prediction)
        except Exception:
            estimate = str(report.prediction)
        lines += [
            "---",
            "",
            f"## {count}. {question.question_text}",
            "",
            f"- Link: {question.page_url}",
            f"- Bot panel estimate (median): **{estimate}**",
            "",
            report.explanation,
            "",
        ]
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    return count


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    parser = argparse.ArgumentParser(description="Run the ensemble forecasting bot")
    parser.add_argument(
        "--mode",
        choices=["tournament", "minibench", "test_questions", "cup_research", "research"],
        default="tournament",
        help=(
            "tournament = Fall FutureEval + MiniBench; minibench = MiniBench only; "
            "cup_research = research brief for open Metaculus Cup questions (never posts); "
            "research = research brief for the questions given with --urls (never posts)"
        ),
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="forecast but do not post to Metaculus"
    )
    parser.add_argument(
        "--urls", default="", help="research mode: Metaculus question links, space or comma separated"
    )
    parser.add_argument(
        "--max-questions", type=int, default=40, help="research modes: cap to control cost"
    )
    args = parser.parse_args()
    mode = args.mode
    research_only = mode in ("cup_research", "research")
    # Research modes are for human contests: never post a bot forecast there.
    publish = not args.dry_run and not research_only

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
        skip_previously_forecasted_questions=not research_only,
        extra_metadata_in_explanation=True,
    )

    client = MetaculusClient()
    reports = []
    if research_only:
        if mode == "research":
            links = [u for u in re.split(r"[\s,]+", args.urls) if u]
            if not links:
                raise SystemExit("research mode needs --urls with at least one question link")
            questions = [client.get_question_by_url(u) for u in links]
        else:
            questions = client.get_all_open_questions_from_tournament(
                client.CURRENT_METACULUS_CUP_ID
            )
        questions = questions[: args.max_questions]
        reports = asyncio.run(bot.forecast_questions(questions, return_exceptions=True))
        bot.log_report_summary(reports)
        path = os.path.join(RESEARCH_DIR, f"brief-{datetime.now().strftime('%Y%m%d-%H%M')}.md")
        n = write_research_brief(reports, path)
        print(f"\n📝  Research brief with {n} question(s) written to {path} (nothing posted).")
        return

    targets = {
        "tournament": [client.CURRENT_AI_COMPETITION_ID, client.CURRENT_MINIBENCH_ID],
        "minibench": [client.CURRENT_MINIBENCH_ID],
        "test_questions": ["bot-testing-area"],
    }[mode]
    if mode == "test_questions":
        bot.skip_previously_forecasted_questions = False

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
