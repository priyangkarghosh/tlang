from dataclasses import dataclass


@dataclass(slots=True)
class Attribute:
    name: str
    raw_args: str
    args: list[str]
    kwargs: dict[str, str]