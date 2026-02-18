#!/usr/bin/env python3
"""
llm_generate.py - Production-ready script to generate answers using AWS Bedrock.

Features:
- Uses augmenting-prompt.PromptBuilder to build the prompt.
- Retrieves context from ChromaDB via retrieve.retrieve_chunks (if no context file provided).
- Sends the prompt to AWS Bedrock (supports Claude, Llama, Titan models).
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

import boto3
from botocore.exceptions import ClientError, BotoCoreError

# Add parent directory to path to import local modules
sys.path.insert(0, str(Path(__file__).parent))
from augmenting_prompt import PromptBuilder, get_default_builder, load_context_from_file
from retrieve import retrieve_chunks

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Default settings
DEFAULT_REGION = os.environ.get("AWS_REGION", "us-east-1")
DEFAULT_MODEL_ID = os.environ.get("BEDROCK_MODEL_ID", "anthropic.claude-v2")
DEFAULT_MAX_TOKENS = 500
DEFAULT_TEMPERATURE = 0.7
DEFAULT_TOP_P = 0.9

# Model-specific request body templates
MODEL_REQUEST_TEMPLATES = {
    # Anthropic Claude
    "anthropic.claude": lambda prompt, max_tokens, temperature, top_p: {
        "prompt": f"\n\nHuman: {prompt}\n\nAssistant:",
        "max_tokens_to_sample": max_tokens,
        "temperature": temperature,
        "top_p": top_p,
        "stop_sequences": ["\n\nHuman:"]
    },
    # Meta Llama 2
    "meta.llama2": lambda prompt, max_tokens, temperature, top_p: {
        "prompt": prompt,
        "max_gen_len": max_tokens,
        "temperature": temperature,
        "top_p": top_p
    },
    # Amazon Titan
    "amazon.titan": lambda prompt, max_tokens, temperature, top_p: {
        "inputText": prompt,
        "textGenerationConfig": {
            "maxTokenCount": max_tokens,
            "temperature": temperature,
            "topP": top_p,
            "stopSequences": []
        }
    },
    # AI21 Labs Jurassic (if needed)
    "ai21.j2": lambda prompt, max_tokens, temperature, top_p: {
        "prompt": prompt,
        "maxTokens": max_tokens,
        "temperature": temperature,
        "topP": top_p,
        "stopSequences": []
    }
}

def build_bedrock_request(model_id: str, prompt: str, max_tokens: int, temperature: float, top_p: float) -> Dict[str, Any]:
    """
    Build the request body for a given Bedrock model.

    Args:
        model_id: Full model ID (e.g., 'anthropic.claude-v2').
        prompt: The formatted prompt string.
        max_tokens: Maximum tokens to generate.
        temperature: Sampling temperature.
        top_p: Top-p sampling.

    Returns:
        Request body dictionary.

    Raises:
        ValueError: If model_id is not supported.
    """
    # Match model family (first part before dot or dash)
    for family_prefix, builder in MODEL_REQUEST_TEMPLATES.items():
        if model_id.startswith(family_prefix):
            return builder(prompt, max_tokens, temperature, top_p)
    raise ValueError(f"Unsupported model ID: {model_id}. Supported families: {list(MODEL_REQUEST_TEMPLATES.keys())}")

def generate_with_bedrock(
    prompt: str,
    model_id: str = DEFAULT_MODEL_ID,
    region: str = DEFAULT_REGION,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    temperature: float = DEFAULT_TEMPERATURE,
    top_p: float = DEFAULT_TOP_P,
    retries: int = 3,
    retry_delay: float = 1.0
) -> str:
    """
    Send a prompt to AWS Bedrock and return the generated text.

    Args:
        prompt: The prompt string to send.
        model_id: Bedrock model identifier.
        region: AWS region.
        max_tokens: Maximum tokens in response.
        temperature: Sampling temperature.
        top_p: Top-p sampling.
        retries: Number of retry attempts on failure.
        retry_delay: Initial delay between retries (exponential backoff).

    Returns:
        Generated text from the model.

    Raises:
        Exception: If all retries fail or client errors occur.
    """
    # Create Bedrock runtime client
    try:
        client = boto3.client("bedrock-runtime", region_name=region)
    except (BotoCoreError, ClientError) as e:
        logger.error(f"Failed to create Bedrock client: {e}")
        raise

    # Build request body
    try:
        body = build_bedrock_request(model_id, prompt, max_tokens, temperature, top_p)
        request_body = json.dumps(body)
    except Exception as e:
        logger.error(f"Error building request body: {e}")
        raise

    # Invoke model with retries
    for attempt in range(retries):
        try:
            logger.info(f"Invoking Bedrock model {model_id} (attempt {attempt+1}/{retries})")
            response = client.invoke_model(
                modelId=model_id,
                contentType="application/json",
                accept="application/json",
                body=request_body
            )
            response_body = json.loads(response["body"].read())
            logger.debug(f"Bedrock response: {response_body}")

            # Extract text based on model family
            if model_id.startswith("anthropic.claude"):
                return response_body["completion"].strip()
            elif model_id.startswith("meta.llama"):
                return response_body["generation"].strip()
            elif model_id.startswith("amazon.titan"):
                return response_body["results"][0]["outputText"].strip()
            elif model_id.startswith("ai21.j2"):
                return response_body["completions"][0]["data"]["text"].strip()
            else:
                # Fallback: try common keys
                if "completion" in response_body:
                    return response_body["completion"].strip()
                elif "generation" in response_body:
                    return response_body["generation"].strip()
                elif "outputText" in response_body:
                    return response_body["outputText"].strip()
                else:
                    raise ValueError(f"Unknown response format for model {model_id}: {response_body}")

        except (ClientError, BotoCoreError, json.JSONDecodeError, KeyError, ValueError) as e:
            logger.warning(f"Bedrock invocation attempt {attempt+1} failed: {e}")
            if attempt == retries - 1:
                logger.error("All retry attempts exhausted.")
                raise
            # Exponential backoff
            sleep_time = retry_delay * (2 ** attempt)
            logger.info(f"Retrying in {sleep_time:.1f} seconds...")
            time.sleep(sleep_time)

    # Should not reach here
    raise RuntimeError("Unexpected error in generate_with_bedrock")

def main():
    parser = argparse.ArgumentParser(
        description="Generate an answer using RAG (retrieval + prompt builder) and AWS Bedrock."
    )
    # Question (positional)
    parser.add_argument("question", nargs="?", type=str, help="The user's question.")
    # Context options
    parser.add_argument("--context-file", type=str, help="Path to a text file containing context documents (separated by '---' lines).")
    parser.add_argument("--n-results", type=int, default=3, help="Number of documents to retrieve (only used with retrieval).")
    # Template options
    parser.add_argument("--template", type=str, default=None, help="Template file name (defaults to rag_prompt.txt).")
    # Bedrock options
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL_ID, help=f"Bedrock model ID (default: {DEFAULT_MODEL_ID})")
    parser.add_argument("--region", type=str, default=DEFAULT_REGION, help=f"AWS region (default: {DEFAULT_REGION})")
    parser.add_argument("--max-tokens", type=int, default=DEFAULT_MAX_TOKENS, help=f"Max tokens to generate (default: {DEFAULT_MAX_TOKENS})")
    parser.add_argument("--temperature", type=float, default=DEFAULT_TEMPERATURE, help=f"Temperature (default: {DEFAULT_TEMPERATURE})")
    parser.add_argument("--top-p", type=float, default=DEFAULT_TOP_P, help=f"Top-p (default: {DEFAULT_TOP_P})")
    parser.add_argument("--retries", type=int, default=3, help="Number of retries for Bedrock calls.")
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

    # Generate answer using Bedrock
    try:
        answer = generate_with_bedrock(
            prompt=final_prompt,
            model_id=args.model,
            region=args.region,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            retries=args.retries
        )
    except Exception as e:
        logger.error(f"Bedrock generation failed: {e}")
        sys.exit(1)

    # Output the answer
    print("\n" + "="*50)
    print(f"Question: {question_for_log}")
    print("="*50)
    print(answer)
    print("="*50)

if __name__ == "__main__":
    main()