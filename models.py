"""
models - One place that decides which model we talk to

Every lab imports get_model() from here, so no lab hard-codes a provider.
DeepSeek answers. If its key is missing, we use OpenAI instead.

You do not run this file. The labs import it.
"""

import os

from dotenv import load_dotenv

load_dotenv()

DEEPSEEK_MODEL = "deepseek-chat"
OPENAI_MODEL = "gpt-4o-mini"


def deepseek_key():
    key = os.getenv("DEEPSEEK_API_KEY", "")
    return key.strip()


def openai_key():
    key = os.getenv("OPENAI_API_KEY", "")
    return key.strip()


def which_provider():
    """Return the name of the chat provider we will actually use."""
    if deepseek_key() != "" and deepseek_key() != "your_deepseek_api_key_here":
        return "DeepSeek"
    if openai_key() != "":
        return "OpenAI"
    return "none"


def get_model(temperature=0.0, max_tokens=500):
    """Return a chat model. DeepSeek if we have its key, otherwise OpenAI."""
    provider = which_provider()
    if provider == "DeepSeek":
        from langchain_deepseek import ChatDeepSeek
        return ChatDeepSeek(model=DEEPSEEK_MODEL, temperature=temperature, max_tokens=max_tokens)
    if provider == "OpenAI":
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(model=OPENAI_MODEL, temperature=temperature, max_tokens=max_tokens)
    print("No model key found. Please run 00_setup_check.py first.")
    raise SystemExit()
