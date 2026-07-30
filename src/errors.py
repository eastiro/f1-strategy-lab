from __future__ import annotations


class F1LabError(Exception):
    """Base error for project-level failures that need a clear user message."""


class CaseDataUnavailableError(F1LabError, FileNotFoundError):
    """Raised when a bundled race case is missing one or more required files."""
