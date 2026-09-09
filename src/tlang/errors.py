# -------------------------------------------------------------
# @file          errors.py
# @author        Priyangkar Ghosh
# @created       2026-09-07
# @description   Diagnostic types carrying source location, so
#                errors point at the .tlang line that caused them
# @license       MIT
# -------------------------------------------------------------

from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True, frozen=True)
class SourceLocation:
    """Points at a line in an original .tlang file (1-based)."""
    module: str | None = None
    line: int | None = None

    def __str__(self) -> str:
        if self.module is None: return '<unknown>'
        if self.line is None: return self.module
        return f'{self.module}:{self.line}'


class TlangError(Exception):
    """Base class for every error raised by tlang.

    Carries an optional source location so messages can point back at the
    .tlang line responsible instead of at preprocessor internals.
    """

    def __init__(self, message: str, location: SourceLocation | None = None) -> None:
        self.message = message
        self.location = location
        super().__init__(str(self))

    def __str__(self) -> str:
        if self.location is None: return self.message
        return f'{self.location}: {self.message}'


class TlangSyntaxError(TlangError):
    """Malformed attribute, unbalanced brackets, or an unterminated function body."""


class TlangAttributeError(TlangError):
    """An attribute is unknown, misapplied, or given bad arguments."""


class TlangDependencyError(TlangError):
    """A missing module, a circular [include], or a duplicate module name."""


class TlangBindingError(TlangError):
    """SSBO binding points conflict or are exhausted."""


class TlangCompileError(TlangError):
    """A generated GLSL stage failed to compile.

    ``source`` holds the exact GLSL handed to the driver so that the #line
    directives in it can be matched against the driver's message.
    """

    def __init__(
        self,
        message: str,
        location: SourceLocation | None = None,
        *,
        stage: str | None = None,
        entry_point: str | None = None,
        source: str | None = None,
    ) -> None:
        self.stage = stage
        self.entry_point = entry_point
        self.source = source
        super().__init__(message, location)


class TlangLinkError(TlangError):
    """A [program(...)] could not be linked into a GL program."""


__all__ = [
    'SourceLocation',
    'TlangError',
    'TlangSyntaxError',
    'TlangAttributeError',
    'TlangDependencyError',
    'TlangBindingError',
    'TlangCompileError',
    'TlangLinkError',
]
