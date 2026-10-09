"""Exception hierarchy. The CLI maps these to friendly messages and exit codes."""


class CullerError(Exception):
    """Base class for all expected, user-facing errors."""


class ConfigError(CullerError):
    """Configuration file or override is invalid."""


class ImageLoadError(CullerError):
    """An image could not be decoded or violates a safety limit."""


class AnalyzerUnavailable(CullerError):
    """An analyzer cannot run in this environment (missing dependency or model)."""


class ReportError(CullerError):
    """A report could not be read, parsed or written."""
