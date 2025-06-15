from dataclasses import dataclass


@dataclass(slots=True)
class ShaderSourceLine:
    vctx: str
    data: str