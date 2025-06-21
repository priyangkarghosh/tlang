from enum import Enum


class ShaderStage(str, Enum):
    VERT = 'vert'
    FRAG = 'frag'
    GEOM = 'geom'
    COMP = 'comp'
    TESC = 'tesc'
    TESE = 'tese'

    @classmethod
    def from_token(cls, token: str) -> 'ShaderStage':
        try: return _SHADER_STAGE_ALIASES[token.lower()]
        except KeyError as exc:
            valid = ", ".join(sorted(_SHADER_STAGE_ALIASES))
            raise ValueError(
                f"Unknown shader stage '{token}'. Expected one of: {valid}"
            ) from exc
    
    @classmethod
    def gather_stages(cls, kwargs: dict) -> dict['ShaderStage', str | None]:
        stages: dict[ShaderStage, str | None] = {}
        for token, stage in _SHADER_STAGE_ALIASES.items():
            if (name := kwargs.get(token)) is not None:
                stages[stage] = name
        return stages

_SHADER_STAGE_ALIASES: dict[str, ShaderStage] = {
    # vertex
    "vert":        ShaderStage.VERT,
    "vertex":      ShaderStage.VERT,
    # fragment
    "frag":        ShaderStage.FRAG,
    "fragment":    ShaderStage.FRAG,
    # geometry
    "geom":        ShaderStage.GEOM,
    "geometry":    ShaderStage.GEOM,
    # compute
    "comp":        ShaderStage.COMP,
    "compute":     ShaderStage.COMP,
    # tessellation control
    "tesc":        ShaderStage.TESC,
    "tess_control":ShaderStage.TESC,
    # tessellation evaluation
    "tese":        ShaderStage.TESE,
    "tess_eval":   ShaderStage.TESE,
}