# -------------------------------------------------------------
# @file          attribute.py
# @author        Priyangkar Ghosh
# @created       2025-06-14
# @description   Attribute dataclass
# @license       MIT
# -------------------------------------------------------------

from dataclasses import dataclass


@dataclass(slots=True)
class Attribute:
    name: str
    raw_args: str
    args: list[str]
    kwargs: dict[str, str]