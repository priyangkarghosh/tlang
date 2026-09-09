# -------------------------------------------------------------
# @file          attribute.py
# @author        Priyangkar Ghosh
# @created       2025-06-14
# @description   Attribute dataclass
# @license       MIT
# -------------------------------------------------------------

from dataclasses import dataclass

from tlang.errors import SourceLocation


@dataclass(slots=True)
class Attribute:
    name: str
    raw_args: str
    args: list[str]
    kwargs: dict[str, str]
    # Where in the .tlang source this was written; set by AttributeManager.
    location: SourceLocation | None = None