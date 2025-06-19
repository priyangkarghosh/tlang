from dataclasses import dataclass


@dataclass(slots=True)
class Attribute:
    name: str
    args: list[str]
    kwargs: dict[str, str]