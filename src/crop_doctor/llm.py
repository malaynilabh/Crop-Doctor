"""
LLM integration for Crop Doctor.

This module provides:
    - Gemini API client initialization (cached singleton)
    - Generic LLM text generation
    - Crop Doctor structured diagnosis explanation generation
    - A generator -> critic -> reviser self-review agent for that explanation
    - A stateful expert chat, built on the Interactions API's
      previous_interaction_id (server-side conversation memory)

The LLM layer is intentionally separated from Streamlit so that
it can later be reused by:
    - RAG
    - LangChain
    - LangGraph
    - Agent workflows
    - Evaluation pipelines

That means: no `import streamlit` here, and no session state. Anything
that needs to persist across turns (chat history, a diagnosis, a
previous_interaction_id) is passed in and handed back by the caller.
"""

import os
import time
from functools import lru_cache

from dotenv import load_dotenv
from google import genai

from .prompts import (
    build_chat_system_prompt,
    build_diagnosis_prompt,
    build_revision_prompt,
    build_review_prompt,
    build_system_prompt,
)
from .schemas import DiagnosisExplanation, ReviewVerdict


# ------------------------------------------------------------------
# Environment configuration
# ------------------------------------------------------------------

load_dotenv()


# ------------------------------------------------------------------
# Model configuration
# ------------------------------------------------------------------

DEFAULT_MODEL = "gemini-3.8-flash"

# gemini-3.8-flash reasons ("thinks") by default before answering, at a
# "medium" level. That's the right call for open-ended reasoning, but this
# app's tasks (structured extraction, a short chat reply) are well-specified
# enough that "low" thinking cuts latency and token cost without hurting
# quality — see https://ai.google.dev/gemini-api/docs/thinking. Note that
# max_output_tokens is a hard cap on thinking + output combined, so a low
# thinking_level also protects you from silent truncation of the JSON body.

DEFAULT_THINKING_LEVEL = "low"


class LLMConfigError(RuntimeError):
    """Raised when no Gemini API key is configured."""


# ------------------------------------------------------------------
# Gemini client (cached singleton — see "problem #2" in chat)
# ------------------------------------------------------------------

@lru_cache(maxsize=1)
def create_llm_client() -> genai.Client:
    """
    Create and return a Gemini API client.

    Cached so the whole process reuses one client (and its underlying
    HTTP connection pool) instead of paying setup cost on every call.
    """
    api_key = os.getenv("GEMINI_API_KEY")

    if not api_key:
        raise LLMConfigError(
            "GEMINI_API_KEY is not configured. "
            "Please add it to your .env file."
        )

    return genai.Client(api_key=api_key)


# ------------------------------------------------------------------
# Generic LLM generation
# ------------------------------------------------------------------

def generate_llm_response(
    system_prompt: str,
    user_prompt: str,
    model: str = DEFAULT_MODEL,
    temperature: float = 0.3,
    max_output_tokens: int = 1000,
    thinking_level: str = DEFAULT_THINKING_LEVEL,
) -> str:
    """
    Generate a plain-text response using Gemini's Interactions API.
    """

    client = create_llm_client()

    try:
        interaction = client.interactions.create(
            model=model,
            system_instruction=system_prompt,
            input=user_prompt,
            generation_config={
                "temperature": temperature,
                "max_output_tokens": max_output_tokens,
                "thinking_level": thinking_level,
            },
        )

    except Exception as exc:
        raise RuntimeError(f"LLM request failed: {exc}") from exc

    if not interaction.output_text:
        raise ValueError("LLM returned an empty response.")

    return interaction.output_text.strip()


def _generate_structured(
    system_prompt: str,
    user_prompt: str,
    schema_model,
    model: str = DEFAULT_MODEL,
    temperature: float = 0.3,
    max_output_tokens: int = 1500,
    thinking_level: str = DEFAULT_THINKING_LEVEL,
):
    """
    Shared helper: call the Interactions API with a Pydantic response schema
    and return a validated instance of that schema.
    """

    client = create_llm_client()

    try:
        interaction = client.interactions.create(
            model=model,
            system_instruction=system_prompt,
            input=user_prompt,
            response_format={
                "type": "text",
                "mime_type": "application/json",
                "schema": schema_model.model_json_schema(),
            },
            generation_config={
                "temperature": temperature,
                "max_output_tokens": max_output_tokens,
                "thinking_level": thinking_level,
            },
        )

    except Exception as exc:
        raise RuntimeError(f"LLM request failed: {exc}") from exc

    if not interaction.output_text:
        raise ValueError("LLM returned an empty response.")

    try:
        return schema_model.model_validate_json(interaction.output_text)
    except Exception as exc:
        raise ValueError("LLM returned an invalid structured response.") from exc


# ------------------------------------------------------------------
# Crop Doctor diagnosis explanation (+ self-review agent)
# ------------------------------------------------------------------

def generate_diagnosis_explanation(
    crop: str,
    disease: str,
    confidence: float,
    language: str = "English",
    model: str = DEFAULT_MODEL,
    self_review: bool = True,
    max_revisions: int = 1,
):
    """
    Generate a structured, farmer-friendly, language-aware explanation of a
    CNN disease prediction — one call producing every field the UI needs
    (previously this took two separate calls: summary + remedy).

    Returns
    -------
    (DiagnosisExplanation, review_meta: dict)
        review_meta = {"initial_passed": bool, "revised": bool, "feedback": str}
        lets the caller show what the self-review agent did.
    """

    if not 0.0 <= confidence <= 1.0:
        raise ValueError("confidence must be between 0.0 and 1.0")

    system_prompt = build_system_prompt(language)
    user_prompt = build_diagnosis_prompt(crop=crop, disease=disease, confidence=confidence)

    explanation = _generate_structured(
        system_prompt, user_prompt, DiagnosisExplanation, model=model, max_output_tokens=1500
    )

    review_meta = {"initial_passed": True, "revised": False, "feedback": "none"}

    if self_review:
        explanation, review_meta = _self_review_diagnosis(
            explanation, crop, disease, confidence, language, model=model, max_revisions=max_revisions
        )

    return explanation, review_meta


def _self_review_diagnosis(explanation, crop, disease, confidence, language, model, max_revisions=1):
    """The generator->critic->reviser loop, using structured output at every step."""

    review_system = (
        "You are a strict quality reviewer for agricultural AI explanations. "
        "Judge only against the criteria given."
    )

    def _review(exp) -> ReviewVerdict:
        prompt = build_review_prompt(exp, crop, disease, confidence, language)
        return _generate_structured(
            review_system, prompt, ReviewVerdict, model=model, max_output_tokens=250
        )

    verdict = _review(explanation)
    initial_passed = verdict.passed
    revised = False

    attempts = 0
    while not verdict.passed and attempts < max_revisions:
        revision_prompt = build_revision_prompt(explanation, verdict.feedback, crop, disease, confidence, language)
        explanation = _generate_structured(
            build_system_prompt(language),
            revision_prompt,
            DiagnosisExplanation,
            model=model,
            max_output_tokens=1500,
        )
        verdict = _review(explanation)
        revised = True
        attempts += 1

    review_meta = {"initial_passed": initial_passed, "revised": revised, "feedback": verdict.feedback}
    return explanation, review_meta


# ------------------------------------------------------------------
# Expert chat (stateful via previous_interaction_id)
# ------------------------------------------------------------------

def chat_with_expert(
    crop: str,
    disease: str,
    confidence: float,
    language: str,
    user_message: str,
    previous_interaction_id: str | None = None,
    model: str = DEFAULT_MODEL,
):
    """
    One chat turn, grounded in the current diagnosis.

    Stateless from this module's point of view: pass in the previous
    interaction's id (or None for the first turn) and you get back the new
    reply plus a fresh id to pass in next time. The Interactions API keeps
    the actual conversation history server-side — cheaper and simpler than
    resending the full transcript every turn.

    Returns
    -------
    (reply_text: str, interaction_id: str | None)
    """

    client = create_llm_client()
    system_prompt = build_chat_system_prompt(crop, disease, confidence, language)

    try:
        interaction = client.interactions.create(
            model=model,
            system_instruction=system_prompt,
            input=user_message,
            previous_interaction_id=previous_interaction_id,
            generation_config={
                "temperature": 0.4,
                "max_output_tokens": 600,
                "thinking_level": DEFAULT_THINKING_LEVEL,
            },
        )
    except Exception as exc:
        raise RuntimeError(f"LLM request failed: {exc}") from exc

    if not interaction.output_text:
        raise ValueError("LLM returned an empty response.")

    return interaction.output_text.strip(), getattr(interaction, "id", None)