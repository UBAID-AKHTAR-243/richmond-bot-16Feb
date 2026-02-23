"""
llm_response.py - Production-ready LLM response generation for RAG chatbot using AWS Bedrock.

Uses PromptBuilder from augmenting_prompt.py to build prompts and calls Bedrock models.
Supports multiple model families (Titan, Claude, Llama2, Mistral) with automatic request/response formatting.
"""

import json
import logging
import os
import time
from functools import wraps
from pathlib import Path
from typing import List, Dict, Any, Optional, Union, Callable

import boto3
import botocore
from botocore.exceptions import ClientError, BotoCoreError

from augmenting_prompt import PromptBuilder, get_default_builder, TemplateNotFoundError, InvalidInputError

# Configure module logger
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------
# Retry decorator for transient Bedrock errors
# ---------------------------------------------------------------------
def retry(max_retries=3, backoff_factor=2, exceptions=(
        ClientError,
        BotoCoreError,
        botocore.exceptions.ConnectionError,
        botocore.exceptions.ReadTimeoutError
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
                    # Check if it's a throttling exception (AWS specific)
                    if isinstance(e, ClientError) and e.response['Error']['Code'] in ['ThrottlingException', 'TooManyRequestsException']:
                        if attempt == max_retries:
                            raise
                        wait = backoff_factor ** attempt
                        logger.warning(
                            f"Throttling (attempt {attempt}/{max_retries}): {e}. "
                            f"Retrying in {wait}s..."
                        )
                        time.sleep(wait)
                    elif attempt == max_retries:
                        raise
                    else:
                        wait = backoff_factor ** attempt
                        logger.warning(
                            f"Bedrock call failed (attempt {attempt}/{max_retries}): {e}. "
                            f"Retrying in {wait}s..."
                        )
                        time.sleep(wait)
            return None  # never reached
        return wrapper
    return decorator


# ---------------------------------------------------------------------
# Helper to extract response text from different Bedrock model outputs
# ---------------------------------------------------------------------
def extract_response_text(response_body: dict, model_id: str) -> str:
    """
    Extract the generated text from a Bedrock model response based on model family.

    Args:
        response_body: Parsed JSON response from Bedrock.
        model_id: The model ID used (e.g., 'amazon.titan-text-express-v1').

    Returns:
        Generated text as string.

    Raises:
        ValueError: If the response format is unknown or missing text.
    """
    model_lower = model_id.lower()

    # Amazon Titan
    if 'titan' in model_lower:
        try:
            return response_body['results'][0]['outputText']
        except (KeyError, IndexError) as e:
            raise ValueError(f"Unexpected Titan response format: {response_body}") from e

    # Anthropic Claude (v1, v2) - legacy prompt format
    elif 'claude' in model_lower and 'claude-3' not in model_lower:
        try:
            return response_body['completion']
        except KeyError as e:
            raise ValueError(f"Unexpected Claude (legacy) response format: {response_body}") from e

    # Anthropic Claude 3 (messages API)
    elif 'claude-3' in model_lower:
        try:
            # Claude 3 returns a list of content blocks; we take the first text block.
            content = response_body['content']
            for block in content:
                if block['type'] == 'text':
                    return block['text']
            raise ValueError("No text block found in Claude 3 response.")
        except (KeyError, IndexError) as e:
            raise ValueError(f"Unexpected Claude 3 response format: {response_body}") from e

    # Meta Llama 2
    elif 'llama2' in model_lower:
        try:
            return response_body['generation']
        except KeyError as e:
            raise ValueError(f"Unexpected Llama2 response format: {response_body}") from e

    # Mistral AI (mistral-7b, mixtral)
    elif 'mistral' in model_lower or 'mixtral' in model_lower:
        try:
            return response_body['outputs'][0]['text']
        except (KeyError, IndexError) as e:
            raise ValueError(f"Unexpected Mistral response format: {response_body}") from e

    # Default: try to find common patterns
    else:
        # Many models return a simple {"text": "..."} or {"generated_text": "..."}
        if 'text' in response_body:
            return response_body['text']
        elif 'generated_text' in response_body:
            return response_body['generated_text']
        elif 'output' in response_body and isinstance(response_body['output'], str):
            return response_body['output']
        else:
            raise ValueError(f"Unknown response format for model {model_id}: {response_body}")


# ---------------------------------------------------------------------
# Build request body for different model families
# ---------------------------------------------------------------------
def build_request_body(
    prompt: Optional[str] = None,
    messages: Optional[List[Dict[str, str]]] = None,
    model_id: str = "",
    **inference_params
) -> dict:
    """
    Construct the request payload for a Bedrock model based on model family.

    Args:
        prompt: String prompt (for non-chat models).
        messages: List of message dicts (for chat models).
        model_id: The Bedrock model ID.
        **inference_params: Additional inference parameters (temperature, max_tokens, etc.)

    Returns:
        Dictionary ready to be passed as the 'body' to invoke_model.

    Raises:
        ValueError: If neither prompt nor messages is provided, or if model family is unsupported.
    """
    model_lower = model_id.lower()

    # If both prompt and messages are given, prefer messages for chat models; otherwise prompt.
    # But we'll let the caller decide.

    # Amazon Titan
    if 'titan' in model_lower:
        if prompt is None:
            raise ValueError("Titan models require a string prompt.")
        body = {
            "inputText": prompt,
            "textGenerationConfig": {
                "maxTokenCount": inference_params.get("max_tokens", 512),
                "temperature": inference_params.get("temperature", 0.0),
                "topP": inference_params.get("top_p", 0.9),
                "stopSequences": inference_params.get("stop_sequences", []),
            }
        }
        return body

    # Anthropic Claude (legacy prompt format) - Claude 2 and older
    elif 'claude' in model_lower and 'claude-3' not in model_lower:
        if prompt is None:
            raise ValueError("Claude (legacy) models require a string prompt in Human/Assistant format.")
        body = {
            "prompt": prompt,
            "max_tokens_to_sample": inference_params.get("max_tokens", 500),
            "temperature": inference_params.get("temperature", 0.0),
            "top_p": inference_params.get("top_p", 0.9),
            "stop_sequences": inference_params.get("stop_sequences", ["\n\nHuman:"]),
        }
        return body

    # Anthropic Claude 3 (messages API)
    elif 'claude-3' in model_lower:
        if messages is None:
            raise ValueError("Claude 3 models require a messages list.")
        # Convert messages to Claude 3 format: each message has role and content (string)
        claude_messages = []
        for msg in messages:
            role = msg["role"]
            # Claude 3 uses "user" and "assistant" (same as our format)
            claude_messages.append({
                "role": role,
                "content": msg["content"]
            })
        body = {
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": inference_params.get("max_tokens", 500),
            "temperature": inference_params.get("temperature", 0.0),
            "top_p": inference_params.get("top_p", 0.9),
            "messages": claude_messages
        }
        return body

    # Meta Llama 2
    elif 'llama2' in model_lower:
        if prompt is None:
            raise ValueError("Llama2 models require a string prompt.")
        body = {
            "prompt": prompt,
            "max_gen_len": inference_params.get("max_tokens", 512),
            "temperature": inference_params.get("temperature", 0.0),
            "top_p": inference_params.get("top_p", 0.9),
        }
        return body

    # Mistral AI
    elif 'mistral' in model_lower or 'mixtral' in model_lower:
        if prompt is None:
            raise ValueError("Mistral models require a string prompt.")
        body = {
            "prompt": prompt,
            "max_tokens": inference_params.get("max_tokens", 500),
            "temperature": inference_params.get("temperature", 0.0),
            "top_p": inference_params.get("top_p", 0.9),
            "stop": inference_params.get("stop_sequences", []),
        }
        return body

    # Default fallback: assume the model accepts a simple prompt field
    else:
        if prompt is None:
            raise ValueError(f"Unknown model {model_id} and no prompt provided.")
        body = {
            "prompt": prompt,
            **inference_params
        }
        return body


# ---------------------------------------------------------------------
# Main LLMResponder class for AWS Bedrock
# ---------------------------------------------------------------------
class LLMResponder:
    """
    Generates LLM responses using AWS Bedrock, with prompts built by PromptBuilder.
    """

    def __init__(
        self,
        builder: PromptBuilder,
        model_id: str = "amazon.titan-text-express-v1",
        region: Optional[str] = None,
        profile_name: Optional[str] = None,
        boto3_session: Optional[boto3.Session] = None,
        inference_params: Optional[Dict[str, Any]] = None,
        max_retries: int = 3,
    ):
        """
        Initialize the responder.

        Args:
            builder: PromptBuilder instance.
            model_id: Bedrock model ID (e.g., 'anthropic.claude-v2').
            region: AWS region name (if not provided, uses boto3 default).
            profile_name: AWS profile name (if not provided, uses default).
            boto3_session: Pre-configured boto3 Session (overrides region/profile).
            inference_params: Default inference parameters (temperature, max_tokens, etc.).
            max_retries: Maximum retry attempts for Bedrock calls.
        """
        self.builder = builder
        self.model_id = model_id
        self.inference_params = inference_params or {}
        self.max_retries = max_retries

        # Create boto3 session and bedrock client
        if boto3_session:
            session = boto3_session
        else:
            session = boto3.Session(region_name=region, profile_name=profile_name)

        self.bedrock_runtime = session.client('bedrock-runtime')

    def _invoke_bedrock(self, body: dict) -> dict:
        """
        Invoke Bedrock model with retries.

        Args:
            body: Request body dictionary.

        Returns:
            Parsed JSON response.
        """
        # Convert body to JSON string
        body_json = json.dumps(body)

        # Use retry decorator on the actual boto3 call
        @retry(max_retries=self.max_retries)
        def _call():
            try:
                response = self.bedrock_runtime.invoke_model(
                    modelId=self.model_id,
                    contentType='application/json',
                    accept='application/json',
                    body=body_json
                )
                # Read and parse response body
                response_body = response['body'].read().decode('utf-8')
                return json.loads(response_body)
            except Exception as e:
                logger.error(f"Bedrock invoke_model failed: {e}")
                raise

        return _call()

    def respond(
        self,
        question: str,
        context_docs: List[Dict[str, Any]],
        history: Optional[List[Dict[str, str]]] = None,
        template_name: Optional[str] = None,
        use_messages: bool = True,
        **extra_kwargs
    ) -> str:
        """
        Generate a response using the appropriate input format (messages or prompt).

        Args:
            question: User's current question.
            context_docs: List of retrieved document chunks.
            history: Optional conversation history.
            template_name: Optional template file for user message.
            use_messages: If True, build messages via build_messages() and send as chat.
                          If False, build a string prompt via build_prompt().
            **extra_kwargs: Additional placeholders for the template.

        Returns:
            Assistant's response as a string.

        Raises:
            InvalidInputError, TemplateNotFoundError: Propagated from PromptBuilder.
            ValueError: If model family is incompatible with chosen input format.
            ClientError: If Bedrock call fails.
        """
        if use_messages:
            # Build messages (system, history, user)
            messages = self.builder.build_messages(
                question=question,
                context_docs=context_docs,
                history=history,
                template_name=template_name,
                **extra_kwargs
            )
            # Build request body for messages
            body = build_request_body(
                messages=messages,
                model_id=self.model_id,
                **self.inference_params
            )
            logger.debug(f"Sending messages to {self.model_id}: {messages[-1]['content'][:100]}...")
        else:
            # Build string prompt
            prompt = self.builder.build_prompt(
                question=question,
                context_docs=context_docs,
                history=history,
                template_name=template_name,
                **extra_kwargs
            )
            body = build_request_body(
                prompt=prompt,
                model_id=self.model_id,
                **self.inference_params
            )
            logger.debug(f"Sending prompt to {self.model_id} (first 200 chars): {prompt[:200]}...")

        # Invoke Bedrock
        response_body = self._invoke_bedrock(body)

        # Extract text
        answer = extract_response_text(response_body, self.model_id)
        logger.info(f"Generated response of length {len(answer)} characters.")
        return answer

    def respond_with_string_prompt(
        self,
        question: str,
        context_docs: List[Dict[str, Any]],
        history: Optional[List[Dict[str, str]]] = None,
        template_name: Optional[str] = None,
        **extra_kwargs
    ) -> str:
        """
        Convenience method that forces use_messages=False.
        """
        return self.respond(
            question=question,
            context_docs=context_docs,
            history=history,
            template_name=template_name,
            use_messages=False,
            **extra_kwargs
        )


# ---------------------------------------------------------------------
# Convenience function for quick one‑line usage
# ---------------------------------------------------------------------
def quick_respond(
    question: str,
    context_docs: Optional[List[Dict[str, Any]]] = None,
    history: Optional[List[Dict[str, str]]] = None,
    template_dir: Optional[str] = None,
    system_prompt: Optional[Union[str, Path]] = None,
    model_id: str = "amazon.titan-text-express-v1",
    region: Optional[str] = None,
    use_messages: bool = True,
    **inference_params
) -> str:
    """
    Quick one-liner to get a response using default builder and Bedrock responder.

    Args:
        question: User question.
        context_docs: List of context docs. If None, empty list is used.
        history: Optional chat history.
        template_dir: Override template directory (passed to get_default_builder).
        system_prompt: Override system prompt (passed to get_default_builder).
        model_id: Bedrock model ID.
        region: AWS region.
        use_messages: Whether to use messages format.
        **inference_params: Additional inference parameters (temperature, max_tokens, etc.)

    Returns:
        Response string.
    """
    builder = get_default_builder(template_dir=template_dir, system_prompt=system_prompt)
    responder = LLMResponder(
        builder=builder,
        model_id=model_id,
        region=region,
        inference_params=inference_params
    )
    return responder.respond(
        question=question,
        context_docs=context_docs or [],
        history=history,
        use_messages=use_messages
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
        description="Generate LLM response using augmenting_prompt and AWS Bedrock."
    )
    # Input arguments
    parser.add_argument("question", nargs="?", type=str, help="The user's question.")
    parser.add_argument("--context-file", type=str,
                        help="Path to a text file containing context documents (separated by '---' lines).")
    parser.add_argument("--question", dest="question_flag", type=str,
                        help="Alternative way to provide the question.")
    parser.add_argument("--template", type=str, default=None, help="Template file name.")
    parser.add_argument("--history-file", type=str,
                        help="Path to a JSON file containing chat history (list of messages).")
    parser.add_argument("--n-results", type=int, default=3,
                        help="Number of documents to retrieve (only if no context-file).")
    parser.add_argument("--system-prompt", type=str,
                        help="Optional system prompt file or text.")
    # AWS Bedrock arguments
    parser.add_argument("--model-id", type=str, default="amazon.titan-text-express-v1",
                        help="Bedrock model ID.")
    parser.add_argument("--region", type=str, default=None, help="AWS region.")
    parser.add_argument("--profile", type=str, default=None, help="AWS profile name.")
    parser.add_argument("--temperature", type=float, default=0.0,
                        help="Sampling temperature.")
    parser.add_argument("--max-tokens", type=int, default=500,
                        help="Maximum tokens in response.")
    parser.add_argument("--use-messages", action="store_true",
                        help="Use messages format (if supported by model).")
    parser.add_argument("--no-retrieval", action="store_true",
                        help="Skip retrieval even if question given and no context file; use empty context.")

    args = parser.parse_args()

    # Determine question
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
        # Try to use retrieval module
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
            logger.warning("Retrieval module not available; proceeding with empty context.")
        except Exception as e:
            logger.error(f"Retrieval failed: {e}")
            sys.exit(1)

    # Build the responder
    builder = get_default_builder(system_prompt=args.system_prompt)
    responder = LLMResponder(
        builder=builder,
        model_id=args.model_id,
        region=args.region,
        profile_name=args.profile,
        inference_params={
            "temperature": args.temperature,
            "max_tokens": args.max_tokens,
        }
    )

    # Generate response
    try:
        response = responder.respond(
            question=question,
            context_docs=docs,
            history=history,
            template_name=args.template,
            use_messages=args.use_messages
        )
        print(response)
    except Exception as e:
        logger.exception("Failed to generate response")
        sys.exit(1)