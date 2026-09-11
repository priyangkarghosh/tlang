# -------------------------------------------------------------
# @file          shader_source_line.py
# @author        Priyangkar Ghosh
# @created       2025-06-10
# @description   Essentially just one line of the shader
# @license       MIT
# -------------------------------------------------------------

from dataclasses import dataclass

from tlang.errors import SourceLocation


@dataclass(slots=True)
class ShaderSourceLine:
    vctx: str # virtual context
    data: str

    def location(self, line: int) -> SourceLocation:
        return SourceLocation(self.vctx, line)