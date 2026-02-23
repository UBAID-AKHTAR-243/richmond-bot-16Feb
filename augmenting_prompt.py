"""
augmenting-prompt.py - Production-ready prompt construction for RAG chatbot.

Features:
- Loads templates from external files (supports Jinja2 or plain text).
- Validates input parameters (context, question, history).
- Logs warnings for missing or short context.
- Provides flexible prompt building:
    * build_prompt() -> str (traditional single‑string prompt)
    * build_messages() -> List[Dict] (chat messages with system prompt)
- System prompt can be loaded from a file or set directly.
- Extensible for multiple prompt types (RAG, summarization, etc.).
"""
import logging
import json
import os
import argparse
import sys
from pathlib import Path
from typing import List, Dict, Any, Optional, Union
from string import Template

# For more advanced templating, install Jinja2: pip install jinja2
try:
    from jinja2 import Environment, FileSystemLoader, TemplateNotFound
    HAS_JINJA = True
except ImportError:
    HAS_JINJA = False
    TemplateNotFound = Exception  # dummy for compatibility

# Try to import the retrieval module (retrieve.py)
try:
    from retrieve import retrieve_chunks
    HAS_RETRIEVAL = True
except ImportError:
    HAS_RETRIEVAL = False
    # Define a dummy function to avoid crashes if retrieval is not available
    def retrieve_chunks(query, n_results=3):
        raise ImportError("retrieve module not found. Please ensure retrieve.py is in the same directory.")

# Configure module logger
logger = logging.getLogger(__name__)


class PromptBuilderError(Exception):
    """Base exception for prompt builder errors."""
    pass


class TemplateNotFoundError(PromptBuilderError):
    """Raised when a requested template file is missing."""
    pass


class InvalidInputError(PromptBuilderError):
    """Raised when input parameters are invalid (e.g., empty question)."""
    pass


class PromptBuilder:
    """
    Builds prompts for a RAG chatbot by loading templates and filling placeholders.

    Two output formats are supported:
      - Traditional string prompt (via build_prompt())
      - Chat messages list with system prompt (via build_messages())

    Templates can be stored as plain text files (with $placeholder syntax) or Jinja2
    templates (with {{ placeholder }} syntax). Jinja2 is used if available and the
    template file has a .j2 extension; otherwise, string.Template is used.

    Expected placeholders in the user message template:
        - $context / {{ context }} : The retrieved document text.
        - $question / {{ question }} : The user's current question.
        - $history / {{ history }} : (Optional) Formatted chat history – used only
          in the string prompt method; for chat messages, history is handled separately.
    """

    def __init__(
        self,
        template_dir: Union[str, Path],
        default_template: str = "rag_prompt.txt",
        system_prompt: Optional[Union[str, Path]] = None
    ):
        """
        Initialize the PromptBuilder.

        Args:
            template_dir: Directory containing prompt template files.
            default_template: Name of the default template file (without path).
            system_prompt: Optional system prompt. Can be:
                - a string (used directly)
                - a Path to a text file (content will be read)
                If None, a default professional instruction is used.

        Raises:
            FileNotFoundError: If template_dir does not exist, or if system_prompt
                               is a file path and the file does not exist.
        """
        self.template_dir = Path(template_dir)
        if not self.template_dir.exists():
            raise FileNotFoundError(f"Template directory not found: {self.template_dir}")

        self.default_template = default_template

        if HAS_JINJA:
            self.jinja_env = Environment(
                loader=FileSystemLoader(self.template_dir),
                trim_blocks=True,
                lstrip_blocks=True,
                autoescape=False  # Prompts are plain text, no HTML escaping needed
            )
        else:
            self.jinja_env = None
            logger.info("Jinja2 not installed; using string.Template (supports $placeholder only).")

        # Load system prompt
        self._system_prompt_content = self._load_system_prompt(system_prompt)

    def _load_system_prompt(self, system_prompt: Optional[Union[str, Path]]) -> str:
        """Load system prompt from string or file. Return default if None."""
        if system_prompt is None:
            return (
                "You are a helpful assistant. "
                "Use the provided Context to answer the User Query. "
                "If you don't know the answer, say you don't know. "
                "Keep the tone professional."
            )
        if isinstance(system_prompt, (str, Path)) and Path(system_prompt).is_file():
            # It's a file path
            with open(system_prompt, 'r', encoding='utf-8') as f:
                return f.read().strip()
        # Assume it's a string
        return str(system_prompt).strip()

    def set_system_prompt(self, system_prompt: Union[str, Path]) -> None:
        """Update the system prompt dynamically."""
        self._system_prompt_content = self._load_system_prompt(system_prompt)

    def _load_template(self, template_name: Optional[str] = None) -> Union[Template, Any]:
        """
        Load a template from file. Returns a string.Template or Jinja2 template object.

        Args:
            template_name: Name of template file (e.g., 'rag_prompt.txt'). If None, default is used.

        Returns:
            A template object ready for substitution.

        Raises:
            TemplateNotFoundError: If the template file does not exist.
        """
        name = template_name or self.default_template
        template_path = self.template_dir / name

        if not template_path.exists():
            raise TemplateNotFoundError(f"Template file not found: {template_path}")

        # Use Jinja2 for .j2 files, otherwise string.Template
        if self.jinja_env and name.endswith('.j2'):
            try:
                return self.jinja_env.get_template(name)
            except TemplateNotFound as e:
                raise TemplateNotFoundError(f"Jinja template not found: {name}") from e
        else:
            with open(template_path, 'r', encoding='utf-8') as f:
                content = f.read()
            return Template(content)

    def _validate_inputs(self, question: str, context_docs: List[Dict[str, Any]]) -> str:
        """
        Validate and preprocess inputs before building prompt.

        Args:
            question: User's query string.
            context_docs: List of retrieved document chunks (each a dict with a 'text' key).

        Returns:
            Combined context text.

        Raises:
            InvalidInputError: If question is empty.
        """
        if not question or not question.strip():
            raise InvalidInputError("Question cannot be empty.")

        if not context_docs:
            logger.warning("No context documents provided. Prompt may lack grounding.")
            context_text = "No relevant documents found."
        else:
            # Extract text from each document; handle missing 'text' key gracefully
            texts = []
            for i, doc in enumerate(context_docs):
                text = doc.get('text') or doc.get('content') or doc.get('page_content')
                if text is None:
                    logger.warning(f"Document {i} missing 'text' key; skipping.")
                    continue
                texts.append(text.strip())

            if not texts:
                logger.warning("All context documents had no readable text.")
                context_text = "No valid document content available."
            else:
                context_text = "\n\n---\n\n".join(texts)
                logger.info(f"Combined context length: {len(context_text)} characters.")

        return context_text

    def _format_history(self, history: Optional[List[Dict[str, str]]]) -> str:
        """
        Convert chat history into a readable string for the prompt (used by build_prompt).

        Args:
            history: List of messages, e.g., [{"role": "user", "content": "..."}, ...]

        Returns:
            Formatted history string, or empty string if no history.
        """
        if not history:
            return ""

        lines = []
        for msg in history:
            role = msg.get("role", "unknown").capitalize()
            content = msg.get("content", "").strip()
            if content:
                lines.append(f"{role}: {content}")
        return "\n".join(lines)

    def build_prompt(
        self,
        question: str,
        context_docs: List[Dict[str, Any]],
        history: Optional[List[Dict[str, str]]] = None,
        template_name: Optional[str] = None,
        **extra_kwargs
    ) -> str:
        """
        Construct a traditional single‑string prompt by filling the template.

        The template should contain placeholders for $context, $question, and optionally
        $history (which will be replaced with the formatted history string).

        Args:
            question: The user's current question.
            context_docs: List of retrieved document chunks (each dict must contain text/content).
            history: Optional list of previous conversation messages.
            template_name: Optional specific template file to use (otherwise default).
            **extra_kwargs: Additional placeholders to pass to the template.

        Returns:
            Fully formatted prompt string.

        Raises:
            InvalidInputError: If inputs are invalid.
            TemplateNotFoundError: If template file is missing.
        """
        # 1. Validate and combine context
        context_text = self._validate_inputs(question, context_docs)

        # 2. Format history if provided
        history_text = self._format_history(history) if history else ""

        # 3. Load the template
        template = self._load_template(template_name)

        # 4. Prepare substitution dictionary
        data = {
            "context": context_text,
            "question": question.strip(),
            "history": history_text,
            **extra_kwargs
        }

        # 5. Render
        try:
            if isinstance(template, Template):
                # string.Template substitution
                prompt = template.safe_substitute(**data)
            else:
                # Jinja2 template
                prompt = template.render(**data)
        except KeyError as e:
            logger.error(f"Missing placeholder in template: {e}")
            raise PromptBuilderError(f"Template missing required key: {e}") from e
        except Exception as e:
            logger.exception("Unexpected error during prompt rendering")
            raise PromptBuilderError(f"Prompt rendering failed: {e}") from e

        logger.debug(f"Built prompt (first 200 chars): {prompt[:200]}...")
        return prompt

    def build_messages(
        self,
        question: str,
        context_docs: List[Dict[str, Any]],
        history: Optional[List[Dict[str, str]]] = None,
        template_name: Optional[str] = None,
        **extra_kwargs
    ) -> List[Dict[str, str]]:
        """
        Construct a list of chat messages (suitable for OpenAI's chat API).

        The list always starts with a system message, followed by any history messages,
        and ends with a user message containing the context and current question.

        The user message content is generated by rendering the specified template
        (the same template used by build_prompt) but with the $history placeholder
        intentionally left empty – history is instead injected as separate messages.

        Args:
            question: The user's current question.
            context_docs: List of retrieved document chunks.
            history: Optional list of previous conversation messages. Each message
                     must have "role" and "content" keys. Roles should be "user" or "assistant".
            template_name: Optional template file for the user message content.
            **extra_kwargs: Additional placeholders for the user message template.

        Returns:
            List of message dicts: [{"role": "system", "content": ...},
                                    ... history messages ...,
                                    {"role": "user", "content": ...}]

        Raises:
            InvalidInputError: If inputs are invalid.
            TemplateNotFoundError: If template file is missing.
        """
        # 1. Validate and combine context (same as build_prompt)
        context_text = self._validate_inputs(question, context_docs)

        # 2. Build the user message content using the template (ignore history)
        template = self._load_template(template_name)
        data = {
            "context": context_text,
            "question": question.strip(),
            "history": "",  # history is handled via separate messages, so leave empty
            **extra_kwargs
        }
        try:
            if isinstance(template, Template):
                user_content = template.safe_substitute(**data)
            else:
                user_content = template.render(**data)
        except KeyError as e:
            logger.error(f"Missing placeholder in user template: {e}")
            raise PromptBuilderError(f"User template missing required key: {e}") from e
        except Exception as e:
            logger.exception("Unexpected error during user message rendering")
            raise PromptBuilderError(f"User message rendering failed: {e}") from e

        # 3. Construct messages list
        messages = [{"role": "system", "content": self._system_prompt_content}]

        # 4. Add history messages (if any)
        if history:
            for msg in history:
                role = msg.get("role")
                content = msg.get("content", "").strip()
                if role not in ("user", "assistant"):
                    logger.warning(f"Skipping message with invalid role: {role}")
                    continue
                if content:
                    messages.append({"role": role, "content": content})

        # 5. Add current user message
        messages.append({"role": "user", "content": user_content})

        logger.debug(f"Built messages list with {len(messages)} messages")
        return messages


# Optional: Convenience function for quick use with default builder
_default_builder: Optional[PromptBuilder] = None


def get_default_builder(
    template_dir: Optional[Union[str, Path]] = None,
    system_prompt: Optional[Union[str, Path]] = None
) -> PromptBuilder:
    """
    Get or create a default PromptBuilder instance (singleton-like).

    Args:
        template_dir: Override the default template directory. If not provided,
                      uses the 'prompts' folder relative to this file.
        system_prompt: Optional system prompt (string or file path).

    Returns:
        PromptBuilder instance.
    """
    global _default_builder
    if _default_builder is None:
        if template_dir is None:
            # Assume prompts are in a 'prompts' subdirectory next to this file
            base_dir = Path(__file__).parent / "prompts"
            # If that doesn't exist, try current working directory/prompts
            if not base_dir.exists():
                base_dir = Path.cwd() / "prompts"
            template_dir = base_dir
        _default_builder = PromptBuilder(template_dir, system_prompt=system_prompt)
    return _default_builder


def load_context_from_file(file_path: Union[str, Path]) -> List[Dict[str, str]]:
    """
    Load context documents from a text file.

    The file can contain multiple documents separated by a line containing exactly '---'.
    Each document is converted to a dict with a 'text' key.
    If no separator is found, the entire file is treated as a single document.

    Args:
        file_path: Path to the context file.

    Returns:
        List of dictionaries, each with a 'text' key.
    """
    file_path = Path(file_path)
    if not file_path.exists():
        raise FileNotFoundError(f"Context file not found: {file_path}")

    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()

    # Split by separator lines (e.g., "---" on its own line)
    documents = []
    current_doc_lines = []
    for line in content.splitlines():
        if line.strip() == "---":
            if current_doc_lines:
                documents.append("\n".join(current_doc_lines).strip())
                current_doc_lines = []
        else:
            current_doc_lines.append(line)
    if current_doc_lines:
        documents.append("\n".join(current_doc_lines).strip())

    # If no separator was found, treat the whole content as one document
    if not documents:
        documents = [content.strip()]

    # Convert to required dict format
    return [{"text": doc} for doc in documents if doc]


# Example usage (for testing; not used in production)
if __name__ == "__main__":
    # Setup logging to see warnings
    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser(
        description="Build a prompt from context (file or retrieval) and question. "
                    "You can provide the question as a positional argument or with --question."
    )
    # Positional question (optional)
    parser.add_argument("question", nargs="?", type=str, help="The user's question. If provided without --context-file, retrieval is used.")
    # Optional flags
    parser.add_argument("--context-file", type=str, help="Path to a text file containing context documents (separated by '---' lines).")
    parser.add_argument("--question", dest="question_flag", type=str, help="Alternative way to provide the question.")
    parser.add_argument("--template", type=str, default=None, help="Template file name (optional, defaults to rag_prompt.txt).")
    parser.add_argument("--history-file", type=str, help="Path to a JSON file containing chat history (optional, not implemented in this example).")
    parser.add_argument("--n-results", type=int, default=3, help="Number of documents to retrieve (only used with retrieval).")
    parser.add_argument("--system-prompt", type=str, help="Optional system prompt file or text (if file path, must exist).")
    parser.add_argument("--chat", action="store_true", help="Output chat messages (list) instead of a single string.")

    args = parser.parse_args()

    # Determine template directory (same as before)
    template_dir = Path(__file__).parent / "templates"
    builder = get_default_builder(template_dir, system_prompt=args.system_prompt)

    # Determine the question: positional takes precedence, else use --question flag
    question = args.question if args.question else args.question_flag

    # Decide how to get context documents
    docs = None
    history = []  # Placeholder for now

    if args.context_file:
        # Use file-based context
        if not question:
            logger.error("When using --context-file, you must also provide a question (either positional or --question).")
            sys.exit(1)
        try:
            docs = load_context_from_file(args.context_file)
            logger.info(f"Loaded {len(docs)} document(s) from {args.context_file}")
        except Exception as e:
            logger.error(f"Failed to load context file: {e}")
            sys.exit(1)

    elif question:
        # Use retrieval from ChromaDB
        if not HAS_RETRIEVAL:
            logger.error("Retrieval module (retrieve.py) not available. Cannot fetch context.")
            sys.exit(1)
        try:
            logger.info(f"Retrieving top {args.n_results} documents for: {question}")
            # Note: retrieve_chunks is a dummy if import fails; adjust as needed.
            doc_strings, distances, metadatas = retrieve_chunks(question, n_results=args.n_results)
            if not doc_strings:
                logger.warning("No documents retrieved. Prompt will lack context.")
                docs = []
            else:
                docs = [{"text": doc} for doc in doc_strings]
                logger.info(f"Retrieved {len(docs)} document(s).")
        except Exception as e:
            logger.error(f"Retrieval failed: {e}")
            sys.exit(1)

    else:
        # No arguments: fall back to example data for demonstration
        logger.info("No question provided; using example data.")
        docs = [
            {"text": "Pakistan has a population of over 240 million people."},
            {"text": "The capital of Pakistan is Islamabad."}
        ]
        question = "What is its capital?"
        history = [
            {"role": "user", "content": "What is the population of Pakistan?"},
            {"role": "assistant", "content": "Pakistan has over 240 million people."}
        ]

    # Build the prompt (either string or messages)
    if args.chat:
        messages = builder.build_messages(
            question=question,
            context_docs=docs,
            history=history,
            template_name=args.template
        )
        print(json.dumps(messages, indent=2))  # pretty print for demonstration
    else:
        prompt = builder.build_prompt(
            question=question,
            context_docs=docs,
            history=history,
            template_name=args.template
        )
        print(prompt)