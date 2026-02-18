#!/usr/bin/env python3
"""
llm_response.py - Generate answers using Ollama with DeepSeek-R1 model.

Features:
- Uses augmenting-prompt.PromptBuilder to build the prompt.
- Retrieves context from ChromaDB via retrieve.retrieve_chunks (if no context file provided).
- Sends the prompt to a local Ollama instance (supports DeepSeek-R1 and other models).
- Configurable via command-line arguments and environment variables.
- Comprehensive logging and error handling.
"""

import os
import sys
import json
import logging
import argparse
import time
from pathlib import Path
from typing import Optional, Dict, Any

import requests
from requests.exceptions import RequestException

# Add parent directory to path to import local modules
sys.path.insert(0, str(Path(__file__).parent))
from augmenting-prompt import PromptBuilder, get_default_builder, load_context_from_file
from retrieve import retrieve_chunks

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Default settings
DEFAULT_OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434")
DEFAULT_MODEL = os.environ.get("OLLAMA_MODEL", "deepseek-r1")
DEFAULT_MAX_TOKENS = 500
DEFAULT_TEMPERATURE = 0.7
DEFAULT_TOP_P = 0.9
DEFAULT_TIMEOUT = 60  # seconds

def generate_with_ollama(
    prompt: str,
    model: str = DEFAULT_MODEL,
    base_url: str = DEFAULT_OLLAMA_URL,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    temperature: float = DEFAULT_TEMPERATURE,
    top_p: float = DEFAULT_TOP_P,
    timeout: int = DEFAULT_TIMEOUT,
    retries: int = 3,
    retry_delay: float = 1.0
) -> str:
    """
    Send a prompt to Ollama and return the generated text.

    Args:
        prompt: The prompt string to send.
        model: Ollama model name (e.g., 'deepseek-r1').
        base_url: Base URL of Ollama server (e.g., http://localhost:11434).
        max_tokens: Maximum tokens to generate.
        temperature: Sampling temperature.
        top_p: Top-p sampling.
        timeout: Request timeout in seconds.
        retries: Number of retry attempts on failure.
        retry_delay: Initial delay between retries (exponential backoff).

    Returns:
        Generated text from the model.

    Raises:
        Exception: If all retries fail or client errors occur.
    """
    url = f"{base_url.rstrip('/')}/api/generate"
    headers = {"Content-Type": "application/json"}

    # Build request payload
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {
            "num_predict": max_tokens,
            "temperature": temperature,
            "top_p": top_p,
        }
    }

    for attempt in range(retries):
        try:
            logger.info(f"Invoking Ollama model {model} (attempt {attempt+1}/{retries})")
            response = requests.post(
                url,
                headers=headers,
                json=payload,
                timeout=timeout
            )
            response.raise_for_status()
            result = response.json()

            # Extract the generated text
            if "response" in result:
                return result["response"].strip()
            else:
                raise ValueError(f"Unexpected response format: {result}")

        except (RequestException, json.JSONDecodeError, KeyError, ValueError) as e:
            logger.warning(f"Ollama invocation attempt {attempt+1} failed: {e}")
            if attempt == retries - 1:
                logger.error("All retry attempts exhausted.")
                raise
            # Exponential backoff
            sleep_time = retry_delay * (2 ** attempt)
            logger.info(f"Retrying in {sleep_time:.1f} seconds...")
            time.sleep(sleep_time)

    # Should not reach here
    raise RuntimeError("Unexpected error in generate_with_ollama")

def main():
    parser = argparse.ArgumentParser(
        description="Generate an answer using RAG (retrieval + prompt builder) and Ollama."
    )
    # Question (positional)
    parser.add_argument("question", nargs="?", type=str, help="The user's question.")
    # Context options
    parser.add_argument("--context-file", type=str, help="Path to a text file containing context documents (separated by '---' lines).")
    parser.add_argument("--n-results", type=int, default=3, help="Number of documents to retrieve (only used with retrieval).")
    # Template options
    parser.add_argument("--template", type=str, default=None, help="Template file name (defaults to rag_prompt.txt).")
    # Ollama options
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL, help=f"Ollama model name (default: {DEFAULT_MODEL})")
    parser.add_argument("--url", type=str, default=DEFAULT_OLLAMA_URL, help=f"Ollama server URL (default: {DEFAULT_OLLAMA_URL})")
    parser.add_argument("--max-tokens", type=int, default=DEFAULT_MAX_TOKENS, help=f"Max tokens to generate (default: {DEFAULT_MAX_TOKENS})")
    parser.add_argument("--temperature", type=float, default=DEFAULT_TEMPERATURE, help=f"Temperature (default: {DEFAULT_TEMPERATURE})")
    parser.add_argument("--top-p", type=float, default=DEFAULT_TOP_P, help=f"Top-p (default: {DEFAULT_TOP_P})")
    parser.add_argument("--timeout", type=int, default=DEFAULT_TIMEOUT, help=f"Request timeout in seconds (default: {DEFAULT_TIMEOUT})")
    parser.add_argument("--retries", type=int, default=3, help="Number of retries for Ollama calls.")
    # Optional direct prompt input (bypass builder)
    parser.add_argument("--prompt", type=str, help="Direct prompt string (if provided, bypasses builder and retrieval).")
    # History (optional, but not implemented)
    parser.add_argument("--history-file", type=str, help="Path to JSON chat history (not yet implemented).")

    args = parser.parse_args()

    # If direct prompt is given, use it and skip everything else
    if args.prompt:
        logger.info("Using direct prompt (builder and retrieval bypassed).")
        final_prompt = args.prompt
        question_for_log = "direct prompt"
    else:
        # Determine question
        question = args.question
        if not question:
            logger.error("No question provided. Please provide a question as a positional argument.")
            sys.exit(1)

        # Initialize prompt builder
        template_dir = Path(__file__).parent / "templates"
        if not template_dir.exists():
            logger.error(f"Template directory not found: {template_dir}")
            sys.exit(1)
        builder = get_default_builder(template_dir)

        # Get context documents
        docs = None
        history = []  # Not implemented; could be loaded from file if needed

        if args.context_file:
            # Use file-based context
            try:
                docs = load_context_from_file(args.context_file)
                logger.info(f"Loaded {len(docs)} document(s) from {args.context_file}")
            except Exception as e:
                logger.error(f"Failed to load context file: {e}")
                sys.exit(1)
        else:
            # Use retrieval from ChromaDB
            try:
                logger.info(f"Retrieving top {args.n_results} documents for: {question}")
                doc_strings, distances, metadatas = retrieve_chunks(question, n_results=args.n_results)
                if not doc_strings:
                    logger.warning("No documents retrieved. Prompt will lack context.")
                    docs = []
                else:
                    docs = [{"text": doc} for doc in doc_strings]
                    logger.info(f"Retrieved {len(docs)} document(s).")
            except ImportError:
                logger.error("Retrieval module (retrieve.py) not available. Cannot fetch context. Provide --context-file or --prompt.")
                sys.exit(1)
            except Exception as e:
                logger.error(f"Retrieval failed: {e}")
                sys.exit(1)

        # Build the prompt
        try:
            final_prompt = builder.build_prompt(
                question=question,
                context_docs=docs,
                history=history,
                template_name=args.template
            )
        except Exception as e:
            logger.error(f"Prompt building failed: {e}")
            sys.exit(1)
        question_for_log = question

    # Log first 200 chars of prompt for debugging
    logger.debug(f"Final prompt (first 200 chars): {final_prompt[:200]}...")

    # Generate answer using Ollama
    try:
        answer = generate_with_ollama(
            prompt=final_prompt,
            model=args.model,
            base_url=args.url,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            timeout=args.timeout,
            retries=args.retries
        )
    except Exception as e:
        logger.error(f"Ollama generation failed: {e}")
        sys.exit(1)

    # Output the answer
    print("\n" + "="*50)
    print(f"Question: {question_for_log}")
    print("="*50)
    print(answer)
    print("="*50)

if __name__ == "__main__":
    main()