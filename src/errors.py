"""User-facing errors for expected RAG pipeline failures."""


class RAGApplicationError(Exception):
    """Base class for errors that the CLI can present without a traceback."""


class InputDataError(RAGApplicationError):
    """The source PDF input is unavailable or invalid."""


class ParsedDataError(RAGApplicationError):
    """The structured JSON output is unavailable or malformed."""


class KnowledgeBaseError(RAGApplicationError):
    """The Chroma knowledge base cannot be queried safely."""


class LLMServiceError(RAGApplicationError):
    """The configured LLM is unavailable or could not complete a request."""
