"""
llm_response.py - Production-ready LLM response generation for RAG chatbot.

Uses PromptBuilder from augmenting_prompt.py to build prompts and calls an LLM API.
Supports OpenAI ChatCompletion by default, with configurable model and parameters.
"""

import json
import logging
import os
import time
from functools import wraps
from pathlib import Path
from typing import List, Dict, Any, Optional, Union

import openai
from openai import OpenAI

from augmenting_prompt import PromptBuilder, get_default_builder, TemplateNotFoundError, InvalidInputError

# Configure module logger
logger = logging.getLogger(__name__)


def retry(max_retries=3, backoff_factor=2, exceptions=(
        openai.APIError,
        openai.APIConnectionError,
        openai.RateLimitError
)):
    """
    Decorator to retry a function on specified exceptions with exponential backoff.
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            for attempt in range(1, max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    if attempt == max_retries:
                        raise
                    wait = backoff_factor ** attempt
                    logger.warning(
                        f"API call failed (attempt {attempt}/{max_retries}): {e}. "
                        f"Retrying in {wait}s..."
                    )
                    time.sleep(wait)
            return None  # never reached
        return wrapper
    return decorator


class LLMResponder:
    """
    Handles LLM response generation using prompts built by PromptBuilder.
    """

    def __init__(
        self,
        builder: PromptBuilder,
        model: str = "gpt-3.5-turbo",
        temperature: float = 0.0,
        max_tokens: int = 500,
        api_key: Optional[str] = None,
        openai_client: Optional[OpenAI] = None,
    ):
        """
        Initialize the responder.

        Args:
            builder: PromptBuilder instance for constructing prompts/messages.
            model: Model name to use (e.g., "gpt-3.5-turbo", "gpt-4").
            temperature: Sampling temperature (0.0 = deterministic).
            max_tokens: Maximum tokens in response.
            api_key: OpenAI API key. If None, reads from OPENAI_API_KEY env var.
            openai_client: Optional pre-configured OpenAI client instance.
        """
        self.builder = builder
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens

        if openai_client:
            self.client = openai_client
        else:
            key = api_key or os.getenv("OPENAI_API_KEY")
            if not key:
                raise ValueError(
                    "OpenAI API key must be provided via api_key or "
                    "OPENAI_API_KEY environment variable."
                )
            self.client = OpenAI(api_key=key)

    @retry()
    def _call_llm(self, messages: List[Dict[str, str]]) -> str:
        """
        Internal method to call OpenAI ChatCompletion with retries.

        Args:
            messages: List of message dicts.

        Returns:
            Response content as string.
        """
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
            )
            content = response.choices[0].message.content
            if content is None:
                logger.warning("LLM returned empty content.")
                return ""
            return content.strip()
        except Exception as e:
            logger.exception("Unexpected error during LLM call")
            raise

    def respond(
        self,
        question: str,
        context_docs: List[Dict[str, Any]],
        history: Optional[List[Dict[str, str]]] = None,
        template_name: Optional[str] = None,
        **extra_kwargs
    ) -> str:
        """
        Generate a response to the user's question using the given context and history.

        Steps:
        1. Use builder.build_messages() to create the chat messages.
        2. Call the LLM API.
        3. Return the assistant's reply.

        Args:
            question: User's current question.
            context_docs: List of retrieved document chunks (as expected by PromptBuilder).
            history: Optional conversation history.
            template_name: Optional template file for user message.
            **extra_kwargs: Additional placeholders for the user message template.

        Returns:
            Assistant's response as a string.

        Raises:
            InvalidInputError: If inputs are invalid (propagated from builder).
            TemplateNotFoundError: If template missing.
            openai.APIError: If LLM API call fails after retries.
        """
        # Build messages using the PromptBuilder
        messages = self.builder.build_messages(
            question=question,
            context_docs=context_docs,
            history=history,
            template_name=template_name,
            **extra_kwargs
        )

        logger.debug(
            f"Sending {len(messages)} messages to LLM. "
            f"First 200 chars of last user message: {messages[-1]['content'][:200]}..."
        )

        # Call LLM
        response_content = self._call_llm(messages)

        logger.info(f"Generated response of length {len(response_content)} characters.")
        return response_content

    def respond_with_string_prompt(
        self,
        question: str,
        context_docs: List[Dict[str, Any]],
        history: Optional[List[Dict[str, str]]] = None,
        template_name: Optional[str] = None,
        **extra_kwargs
    ) -> str:
        """
        Alternative method using string prompt (build_prompt) and a simple completion.
        This is for models that don't support chat format. Uses the same LLM client
        but with the 'prompt' parameter instead of messages.

        Note: This uses the older completions endpoint, which may not be available for
        all models. For most modern chat models, use respond() instead.

        Args:
            Same as respond().

        Returns:
            Assistant's response as a string.
        """
        # Build string prompt
        prompt = self.builder.build_prompt(
            question=question,
            context_docs=context_docs,
            history=history,
            template_name=template_name,
            **extra_kwargs
        )

        logger.debug(f"Built string prompt of length {len(prompt)} characters.")

        # Use completions endpoint (requires model that supports it)
        try:
            response = self.client.completions.create(
                model=self.model,  # must be a completion model like "gpt-3.5-turbo-instruct"
                prompt=prompt,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
            )
            content = response.choices[0].text
            return content.strip() if content else ""
        except Exception as e:
            logger.exception("String completion failed")
            raise


# ---------------------------------------------------------------------
# Convenience function for quick one‑line usage
# ---------------------------------------------------------------------
def quick_respond(
    question: str,
    context_docs: Optional[List[Dict[str, Any]]] = None,
    history: Optional[List[Dict[str, str]]] = None,
    template_dir: Optional[str] = None,
    system_prompt: Optional[Union[str, Path]] = None,
    **llm_kwargs
) -> str:
    """
    Quick one-liner to get a response using default builder and responder.

    Args:
        question: User question.
        context_docs: List of context docs. If None, empty list is used.
        history: Optional chat history.
        template_dir: Override template directory (passed to get_default_builder).
        system_prompt: Override system prompt (passed to get_default_builder).
        **llm_kwargs: Passed to LLMResponder (model, temperature, etc.)

    Returns:
        Response string.
    """
    builder = get_default_builder(template_dir=template_dir, system_prompt=system_prompt)
    responder = LLMResponder(builder=builder, **llm_kwargs)
    return responder.respond(
        question=question,
        context_docs=context_docs or [],
        history=history
    )


# ---------------------------------------------------------------------
# Command-line interface for testing
# ---------------------------------------------------------------------
if __name__ == "__main__":
    import argparse
    import sys

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    parser = argparse.ArgumentParser(
        description="Generate LLM response using augmenting_prompt and OpenAI."
    )
    # Input arguments (mirror augmenting_prompt.py)
    parser.add_argument("question", nargs="?", type=str,
                        help="The user's question.")
    parser.add_argument("--context-file", type=str,
                        help="Path to a text file containing context documents "
                             "(separated by '---' lines).")
    parser.add_argument("--question", dest="question_flag", type=str,
                        help="Alternative way to provide the question.")
    parser.add_argument("--template", type=str, default=None,
                        help="Template file name.")
    parser.add_argument("--history-file", type=str,
                        help="Path to a JSON file containing chat history "
                             "(list of messages).")
    parser.add_argument("--n-results", type=int, default=3,
                        help="Number of documents to retrieve (only if no context-file).")
    parser.add_argument("--system-prompt", type=str,
                        help="Optional system prompt file or text.")
    # LLM arguments
    parser.add_argument("--model", type=str, default="gpt-3.5-turbo",
                        help="OpenAI model name.")
    parser.add_argument("--temperature", type=float, default=0.0,
                        help="Sampling temperature.")
    parser.add_argument("--max-tokens", type=int, default=500,
                        help="Maximum tokens in response.")
    parser.add_argument("--api-key", type=str,
                        help="OpenAI API key (or set OPENAI_API_KEY env var).")
    parser.add_argument("--no-retrieval", action="store_true",
                        help="Skip retrieval even if question given and no context file; "
                             "use empty context.")

    args = parser.parse_args()

    # Determine question (positional takes precedence)
    question = args.question if args.question else args.question_flag
    if not question and not args.context_file:
        print("Error: No question provided.", file=sys.stderr)
        sys.exit(1)

    # Load history if provided
    history = []
    if args.history_file:
        hist_path = Path(args.history_file)
        if hist_path.exists():
            with open(hist_path, "r", encoding="utf-8") as f:
                history = json.load(f)
            logger.info(f"Loaded history with {len(history)} messages.")
        else:
            logger.error(f"History file not found: {hist_path}")
            sys.exit(1)

    # Get context documents
    docs = []
    if args.context_file:
        try:
            from augmenting_prompt import load_context_from_file
            docs = load_context_from_file(args.context_file)
            logger.info(f"Loaded {len(docs)} document(s) from {args.context_file}")
        except Exception as e:
            logger.error(f"Failed to load context file: {e}")
            sys.exit(1)
    elif question and not args.no_retrieval:
        # Try to use the retrieval module
        try:
            from retrieve import retrieve_chunks
            logger.info(f"Retrieving top {args.n_results} documents for: {question}")
            doc_strings, distances, metadatas = retrieve_chunks(
                question, n_results=args.n_results
            )
            if doc_strings:
                docs = [{"text": doc} for doc in doc_strings]
                logger.info(f"Retrieved {len(docs)} document(s).")
            else:
                logger.warning("No documents retrieved.")
        except ImportError:
            logger.warning(
                "Retrieval module not available; proceeding with empty context."
            )
        except Exception as e:
            logger.error(f"Retrieval failed: {e}")
            sys.exit(1)

    # Build the responder using the default builder
    builder = get_default_builder(system_prompt=args.system_prompt)
    responder = LLMResponder(
        builder=builder,
        model=args.model,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        api_key=args.api_key
    )

    # Generate response
    try:
        response = responder.respond(
            question=question,
            context_docs=docs,
            history=history,
            template_name=args.template
        )
        print(response)
    except Exception as e:
        logger.exception("Failed to generate response")
        sys.exit(1)