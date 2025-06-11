from dataclasses import dataclass


@dataclass
class ShaderSourceLine:
    vctx: str
    data: str