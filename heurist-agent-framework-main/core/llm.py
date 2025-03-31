import asyncio
import json
import logging
import re
import time
from types import SimpleNamespace
from typing import Dict, List, Union

import requests
from openai import AsyncOpenAI, OpenAI

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class LLMError(Exception):
    """Custom exception for LLM-related errors"""
    pass


def _format_messages(system_prompt: str = None, user_prompt: str = None, messages: List[Dict] = None) -> List[Dict]:
    """Convert between different message formats while maintaining backward compatibility"""
    if messages is not None:
        return messages
    if system_prompt and user_prompt:
        return [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}]
    raise ValueError("Either (system_prompt, user_prompt) or messages must be provided")


def call_llm(
    api_key: str,
    model_id: str,
    system_prompt: str = None,
    user_prompt: str = None,
    messages: List[Dict] = None,
    temperature: float = 0.7,
    max_tokens: int = 500,
    max_retries: int = 3,
    initial_retry_delay: int = 1,
) -> str:
    """
    Call LLM with retry mechanism.
    """
    client = OpenAI(api_key=api_key)
    formatted_messages = _format_messages(system_prompt, user_prompt, messages)
    retry_delay = initial_retry_delay

    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=model_id,
                messages=formatted_messages,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            return _handle_tool_response(response.choices[0].message)

        except (requests.exceptions.RequestException, KeyError, IndexError, json.JSONDecodeError, Exception) as e:
            logger.warning(f"{type(e).__name__} (attempt {attempt + 1}/{max_retries}): {str(e)}")
            if attempt < max_retries - 1:
                logger.info(f"Retrying in {retry_delay} seconds...")
                time.sleep(retry_delay)
                retry_delay *= 2

    raise LLMError("All retry attempts failed")


def call_llm_with_tools(*args, **kwargs):
    """Placeholder function for call_llm_with_tools."""
    logger.warning("call_llm_with_tools is missing. Using a placeholder function.")
    return {"content": "Tool function missing."}


async def call_llm_async(
    api_key: str,
    model_id: str,
    system_prompt: str = None,
    user_prompt: str = None,
    messages: List[Dict] = None,
    temperature: float = 0.7,
    max_tokens: int = 500,
    max_retries: int = 3,
    initial_retry_delay: int = 1,
) -> str:
    """
    Asynchronous version of `call_llm()` with retry mechanism.
    """
    client = AsyncOpenAI(api_key=api_key)
    formatted_messages = _format_messages(system_prompt, user_prompt, messages)
    retry_delay = initial_retry_delay

    for attempt in range(max_retries):
        try:
            result = await client.chat.completions.create(
                model=model_id,
                messages=formatted_messages,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            return result.choices[0].message.content

        except (requests.exceptions.RequestException, KeyError, IndexError, json.JSONDecodeError, Exception) as e:
            logger.warning(f"{type(e).__name__} (attempt {attempt + 1}/{max_retries}): {str(e)}")
            if attempt < max_retries - 1:
                logger.info(f"Retrying in {retry_delay} seconds...")
                await asyncio.sleep(retry_delay)
                retry_delay *= 2

    raise LLMError("All retry attempts failed")


def extract_function_calls_to_tool_calls(llm_text: str) -> SimpleNamespace:
    """
    Scan the LLM's text output for a <function=NAME>{...}</function> pattern,
    and convert to appropriate format for tool calls.
    """
    pattern = r"<function=([^>]+)>(.*?)(?:</function>|<function>|<function/>|></function>)"
    matches = re.findall(pattern, llm_text)

    if matches:
        function_name, args_json_str = matches[0]
        parsed_args = json.loads(args_json_str.strip())
        function_obj = SimpleNamespace(name=function_name, arguments=json.dumps(parsed_args))
        return SimpleNamespace(function=function_obj)

    return None


def _handle_tool_response(message):
    """
    Handle and format tool responses from the LLM.
    """
    if hasattr(message, "tool_calls") and message.tool_calls:
        return {"tool_calls": message.tool_calls[0], "content": message.content}
    if hasattr(message, "content") and message.content:
        text_response = message.content
        tool_calls = extract_function_calls_to_tool_calls(text_response)
        if tool_calls:
            logger.info("Found tool calls in response")
            return {"tool_calls": tool_calls, "content": ""}
        return {"content": text_response}
    return message
