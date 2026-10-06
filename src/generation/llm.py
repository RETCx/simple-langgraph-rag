import os
from functools import lru_cache
from typing import Any

from dotenv import load_dotenv
from langchain_core.messages import BaseMessage
from langchain_openai import ChatOpenAI

from ..errors import LLMServiceError


@lru_cache(maxsize=1)
def get_llm() -> ChatOpenAI:
    load_dotenv()

    api_key = os.getenv("OPENROUTER_API_KEY")
    base_url = os.getenv("OPENAI_BASE_URL")
    model_name = os.getenv("MODEL_NAME")

    if not api_key:
        raise LLMServiceError("OPENROUTER_API_KEY is missing. Check your .env file.")

    if not base_url:
        raise LLMServiceError("OPENAI_BASE_URL is missing. Check your .env file.")

    if not model_name:
        raise LLMServiceError("MODEL_NAME is missing. Check your .env file.")

    return ChatOpenAI(
        model=model_name,
        api_key=api_key,
        base_url=base_url,
        temperature=0,
    )


def invoke_llm(messages: list[BaseMessage], purpose: str) -> Any:
    """Invoke the configured LLM and convert provider failures to app errors."""
    try:
        return get_llm().invoke(messages)
    except LLMServiceError:
        raise
    except Exception as exc:
        raise LLMServiceError(
            f"The LLM service could not {purpose}. Check your connection and try again."
        ) from exc
