"""
llm_generate.py - Production-ready answer generation for RAG chatbot.

Features:
- Supports multiple LLM providers: OpenAI, Azure OpenAI, Anthropic, local Hugging Face.
- Configurable via environment variables or configuration dictionary.
- Robust error handling with automatic retries (using tenacity).
- Token counting and context window validation to prevent overflows.
- Logging for monitoring and debugging.
- Extensible design for adding new providers.
"""

import logging
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List, Union
import time

# Optional dependencies (soft imports)
try:
    import openai
    from openai import OpenAI, AzureOpenAI
    HAS_OPENAI = True
except ImportError:
    HAS_OPENAI = False
    openai = None

try:
    import anthropic
    HAS_ANTHROPIC = True
except ImportError:
    HAS_ANTHROPIC = False
    anthropic = None

try:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False

try:
    import tiktoken
    HAS_TIKTOKEN = True
except ImportError:
    HAS_TIKTOKEN = False

try:
    from tenacity import (
        retry,
        stop_after_attempt,
        wait_exponential,
        retry_if_exception_type,
        before_sleep_log
    )
    HAS_TENACITY = True
except ImportError:
    HAS_TENACITY = False
    # Dummy decorator if tenacity not installed (no retries)
    def retry(*args, **kwargs):
        def decorator(func):
            return func
        return decorator

# Configure module logger
logger = logging.getLogger(__name__)


# ========================== Exceptions ==========================
class LLMGenerationError(Exception):
    """Base exception for LLM generation errors."""
    pass


class ConfigurationError(LLMGenerationError):
    """Raised when required configuration is missing or invalid."""
    pass


class ContextWindowExceededError(LLMGenerationError):
    """Raised when the prompt exceeds the model's context window."""
    pass


class ProviderUnavailableError(LLMGenerationError):
    """Raised when the chosen provider is not available (missing library or API key)."""
    pass


# ========================== Data Classes ==========================
@dataclass
class GenerationConfig:
    """Configuration for a single generation call."""
    model: str
    temperature: float = 0.7
    max_tokens: int = 500
    top_p: float = 1.0
    frequency_penalty: float = 0.0
    presence_penalty: float = 0.0
    stop: Optional[List[str]] = None
    timeout: Optional[float] = 60.0
    # Additional provider-specific parameters
    extra_params: Dict[str, Any] = field(default_factory=dict)


@dataclass
class LLMProviderConfig:
    """Configuration for an LLM provider (API keys, endpoints, etc.)."""
    provider: str  # 'openai', 'azure_openai', 'anthropic', 'huggingface'
    api_key: Optional[str] = None
    api_base: Optional[str] = None
    api_version: Optional[str] = None  # for Azure
    deployment_name: Optional[str] = None  # for Azure
    model: str = "gpt-3.5-turbo"  # default model
    # For Hugging Face local
    model_path: Optional[str] = None
    device: Optional[str] = None  # 'cpu', 'cuda'
    max_context_length: int = 4096  # fallback context window size


# ========================== Token Counter ==========================
class TokenCounter:
    """Utility for counting tokens in prompts using tiktoken or fallback heuristics."""

    @staticmethod
    def count_tokens(text: str, model: str = "gpt-3.5-turbo") -> int:
        """
        Count tokens in text using tiktoken if available; otherwise approximate.

        Args:
            text: The text to count tokens for.
            model: The model name (used to select the correct encoding).

        Returns:
            Number of tokens (estimate if tiktoken not available).
        """
        if HAS_TIKTOKEN:
            try:
                encoding = tiktoken.encoding_for_model(model)
            except KeyError:
                # Fallback to cl100k_base (used by most modern OpenAI models)
                encoding = tiktoken.get_encoding("cl100k_base")
            return len(encoding.encode(text))
        else:
            # Simple approximation: 1 token ≈ 4 characters (rough)
            return len(text) // 4

    @staticmethod
    def truncate_to_context(text: str, max_tokens: int, model: str = "gpt-3.5-turbo") -> str:
        """
        Truncate text to fit within max_tokens.

        Args:
            text: The text to truncate.
            max_tokens: Maximum allowed tokens.
            model: Model name for token counting.

        Returns:
            Truncated text.
        """
        if HAS_TIKTOKEN:
            try:
                encoding = tiktoken.encoding_for_model(model)
            except KeyError:
                encoding = tiktoken.get_encoding("cl100k_base")
            tokens = encoding.encode(text)
            if len(tokens) <= max_tokens:
                return text
            truncated_tokens = tokens[:max_tokens]
            return encoding.decode(truncated_tokens)
        else:
            # Approximate truncation by characters
            approx_char_limit = max_tokens * 4
            if len(text) <= approx_char_limit:
                return text
            return text[:approx_char_limit]


# ========================== Base Generator ==========================
class LLMGenerator(ABC):
    """Abstract base class for LLM generators."""

    def __init__(self, config: LLMProviderConfig):
        self.config = config
        self._client = None
        self._setup_client()

    @abstractmethod
    def _setup_client(self):
        """Initialize the underlying client (API or local model)."""
        pass

    @abstractmethod
    def _generate_raw(self, prompt: str, gen_config: GenerationConfig) -> str:
        """
        Perform the actual generation call (without retries/validation).

        Args:
            prompt: The input prompt.
            gen_config: Generation parameters.

        Returns:
            Generated text.
        """
        pass

    def generate(self, prompt: str, gen_config: Optional[GenerationConfig] = None) -> str:
        """
        Public generation method with validation, logging, and retries.

        Args:
            prompt: The input prompt.
            gen_config: Generation parameters (uses defaults if None).

        Returns:
            Generated answer.

        Raises:
            LLMGenerationError: If generation fails after retries.
            ContextWindowExceededError: If prompt exceeds context window (and no truncation).
        """
        if gen_config is None:
            gen_config = GenerationConfig(model=self.config.model)

        # Log start
        logger.info(f"Generating with model {gen_config.model}, prompt length {len(prompt)} chars")

        # Token counting and context window check (if we have a counter)
        if hasattr(self, 'max_context_length') and self.max_context_length:
            prompt_tokens = TokenCounter.count_tokens(prompt, model=gen_config.model)
            if prompt_tokens > self.max_context_length:
                # Try to truncate? Here we raise error; you could add auto-truncate option.
                raise ContextWindowExceededError(
                    f"Prompt of {prompt_tokens} tokens exceeds model's max context length {self.max_context_length}"
                )
            logger.debug(f"Prompt tokens: {prompt_tokens}")

        # Apply retry decorator to the raw generation
        @retry(
            stop=stop_after_attempt(3),
            wait=wait_exponential(multiplier=1, min=2, max=10),
            retry=retry_if_exception_type((openai.APITimeoutError, openai.APIConnectionError, openai.InternalServerError)) if HAS_OPENAI else None,
            before_sleep=before_sleep_log(logger, logging.WARNING),
            reraise=True
        )
        def _generate_with_retry():
            return self._generate_raw(prompt, gen_config)

        try:
            start_time = time.time()
            answer = _generate_with_retry()
            elapsed = time.time() - start_time
            logger.info(f"Generation completed in {elapsed:.2f} seconds, answer length {len(answer)} chars")
            return answer
        except Exception as e:
            logger.exception("Generation failed after retries")
            raise LLMGenerationError(f"Generation failed: {e}") from e


# ========================== OpenAI Generator ==========================
class OpenAIGenerator(LLMGenerator):
    """Generator for OpenAI models (ChatCompletion API)."""

    def __init__(self, config: LLMProviderConfig):
        if not HAS_OPENAI:
            raise ProviderUnavailableError("OpenAI library not installed. Install with `pip install openai`")
        super().__init__(config)
        # Set max context length based on model (simplified; could have mapping)
        self.max_context_length = self._get_model_context_length(config.model)

    def _setup_client(self):
        api_key = self.config.api_key or os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ConfigurationError("OpenAI API key not provided and OPENAI_API_KEY not set")
        self._client = OpenAI(api_key=api_key, base_url=self.config.api_base)

    def _get_model_context_length(self, model: str) -> int:
        """Return max context length for known models, else default."""
        model_lower = model.lower()
        if "gpt-4" in model_lower:
            if "32k" in model_lower:
                return 32768
            return 8192
        elif "gpt-3.5-turbo" in model_lower:
            if "16k" in model_lower:
                return 16384
            return 4096
        else:
            # Default to provided config or 4096
            return self.config.max_context_length

    def _generate_raw(self, prompt: str, gen_config: GenerationConfig) -> str:
        messages = [{"role": "user", "content": prompt}]
        try:
            response = self._client.chat.completions.create(
                model=gen_config.model or self.config.model,
                messages=messages,
                temperature=gen_config.temperature,
                max_tokens=gen_config.max_tokens,
                top_p=gen_config.top_p,
                frequency_penalty=gen_config.frequency_penalty,
                presence_penalty=gen_config.presence_penalty,
                stop=gen_config.stop,
                timeout=gen_config.timeout,
                **gen_config.extra_params
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            logger.error(f"OpenAI API error: {e}")
            raise


# ========================== Azure OpenAI Generator ==========================
class AzureOpenAIGenerator(OpenAIGenerator):
    """Generator for Azure OpenAI models."""

    def _setup_client(self):
        api_key = self.config.api_key or os.getenv("AZURE_OPENAI_KEY")
        if not api_key:
            raise ConfigurationError("Azure OpenAI key not provided and AZURE_OPENAI_KEY not set")
        endpoint = self.config.api_base or os.getenv("AZURE_OPENAI_ENDPOINT")
        if not endpoint:
            raise ConfigurationError("Azure OpenAI endpoint not provided and AZURE_OPENAI_ENDPOINT not set")
        api_version = self.config.api_version or os.getenv("AZURE_OPENAI_VERSION") or "2023-07-01-preview"
        self._client = AzureOpenAI(
            api_key=api_key,
            azure_endpoint=endpoint,
            api_version=api_version
        )

    def _generate_raw(self, prompt: str, gen_config: GenerationConfig) -> str:
        # Azure requires deployment name (model) to be the deployment ID
        deployment = self.config.deployment_name or gen_config.model or self.config.model
        messages = [{"role": "user", "content": prompt}]
        try:
            response = self._client.chat.completions.create(
                model=deployment,
                messages=messages,
                temperature=gen_config.temperature,
                max_tokens=gen_config.max_tokens,
                top_p=gen_config.top_p,
                frequency_penalty=gen_config.frequency_penalty,
                presence_penalty=gen_config.presence_penalty,
                stop=gen_config.stop,
                timeout=gen_config.timeout,
                **gen_config.extra_params
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            logger.error(f"Azure OpenAI API error: {e}")
            raise


# ========================== Anthropic Generator ==========================
class AnthropicGenerator(LLMGenerator):
    """Generator for Anthropic Claude models."""

    def __init__(self, config: LLMProviderConfig):
        if not HAS_ANTHROPIC:
            raise ProviderUnavailableError("Anthropic library not installed. Install with `pip install anthropic`")
        super().__init__(config)
        # Anthropic context lengths (simplified)
        self.max_context_length = self._get_model_context_length(config.model)

    def _setup_client(self):
        api_key = self.config.api_key or os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            raise ConfigurationError("Anthropic API key not provided and ANTHROPIC_API_KEY not set")
        self._client = anthropic.Anthropic(api_key=api_key, base_url=self.config.api_base)

    def _get_model_context_length(self, model: str) -> int:
        model_lower = model.lower()
        if "claude-3-opus" in model_lower:
            return 200000
        elif "claude-3-sonnet" in model_lower:
            return 200000
        elif "claude-3-haiku" in model_lower:
            return 200000
        elif "claude-2.1" in model_lower or "claude-2" in model_lower:
            return 100000
        elif "claude-instant" in model_lower:
            return 100000
        else:
            return self.config.max_context_length

    def _generate_raw(self, prompt: str, gen_config: GenerationConfig) -> str:
        try:
            response = self._client.messages.create(
                model=gen_config.model or self.config.model,
                max_tokens=gen_config.max_tokens,
                temperature=gen_config.temperature,
                top_p=gen_config.top_p,
                messages=[{"role": "user", "content": prompt}],
                stop_sequences=gen_config.stop,
                **gen_config.extra_params
            )
            return response.content[0].text.strip()
        except Exception as e:
            logger.error(f"Anthropic API error: {e}")
            raise


# ========================== HuggingFace Local Generator ==========================
class HuggingFaceGenerator(LLMGenerator):
    """Generator for local Hugging Face models (transformers)."""

    def __init__(self, config: LLMProviderConfig):
        if not HAS_TRANSFORMERS:
            raise ProviderUnavailableError("Transformers library not installed. Install with `pip install transformers torch`")
        super().__init__(config)
        self.model = None
        self.tokenizer = None
        self.device = config.device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.max_context_length = config.max_context_length

    def _setup_client(self):
        """Load model and tokenizer from local path or HuggingFace hub."""
        model_name = self.config.model_path or self.config.model
        try:
            logger.info(f"Loading HuggingFace model: {model_name}")
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                device_map="auto" if self.device == "cuda" else None
            )
            if self.device == "cuda" and not hasattr(self.model, "device_map"):
                self.model.to(self.device)
            self.model.eval()
            logger.info("Model loaded successfully")
        except Exception as e:
            logger.exception("Failed to load HuggingFace model")
            raise ConfigurationError(f"Model loading failed: {e}") from e

    def _generate_raw(self, prompt: str, gen_config: GenerationConfig) -> str:
        try:
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=gen_config.max_tokens,
                    temperature=gen_config.temperature,
                    top_p=gen_config.top_p,
                    do_sample=gen_config.temperature > 0,
                    pad_token_id=self.tokenizer.eos_token_id,
                    **gen_config.extra_params
                )
            generated = self.tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
            return generated.strip()
        except Exception as e:
            logger.error(f"HuggingFace generation error: {e}")
            raise


# ========================== Factory ==========================
def get_llm_generator(provider_config: Union[LLMProviderConfig, Dict[str, Any]]) -> LLMGenerator:
    """
    Factory function to create an LLM generator from a configuration.

    Args:
        provider_config: Either an LLMProviderConfig instance or a dictionary
                         with required keys (at least 'provider').

    Returns:
        An instance of a concrete LLMGenerator.

    Raises:
        ConfigurationError: If provider is unsupported or config invalid.
    """
    if isinstance(provider_config, dict):
        provider_config = LLMProviderConfig(**provider_config)

    provider = provider_config.provider.lower()
    if provider == "openai":
        return OpenAIGenerator(provider_config)
    elif provider == "azure_openai":
        return AzureOpenAIGenerator(provider_config)
    elif provider == "anthropic":
        return AnthropicGenerator(provider_config)
    elif provider == "huggingface":
        return HuggingFaceGenerator(provider_config)
    else:
        raise ConfigurationError(f"Unsupported LLM provider: {provider}")


# ========================== Example Usage ==========================
if __name__ == "__main__":
    # Example: using OpenAI
    logging.basicConfig(level=logging.INFO)

    # Configuration (usually from environment or config file)
    config = {
        "provider": "openai",
        "model": "gpt-3.5-turbo",
        # api_key will be read from OPENAI_API_KEY env var
    }

    generator = get_llm_generator(config)
    prompt = "What is the capital of France?"
    gen_config = GenerationConfig(model="gpt-3.5-turbo", temperature=0.5, max_tokens=100)

    try:
        answer = generator.generate(prompt, gen_config)
        print(f"Answer: {answer}")
    except LLMGenerationError as e:
        print(f"Error: {e}")