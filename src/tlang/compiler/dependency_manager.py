# -------------------------------------------------------------
# @file          dependency_manager.py
# @author        Priyangkar Ghosh
# @created       2025-07-13
# @description   Resolves module dependencies and renders Jinja-templated GLSL text.
# @license       MIT
# -------------------------------------------------------------

import logging
logger = logging.getLogger(__name__)

from jinja2 import DictLoader, Environment, StrictUndefined, TemplateSyntaxError, UndefinedError
from tlang.errors import SourceLocation, TlangDependencyError
from tlang.compiler.shader import Shader
from tlang.compiler.shader_processor import ShaderProcessor

class DependencyManager:
    def __init__(self, constants) -> None:
        self.modules: dict[str, str] = {}
        self.dps_graph: dict[str, set[str]] = {}
        self.constants = constants
        self._env: Environment | None = None
        self._resolved: dict[str, list[str]] = {}  # memoized resolve_dependencies() results, keyed by module name

    def register(self, sp: ShaderProcessor) -> None:
        if sp.name in self.modules:
            raise TlangDependencyError(f"Shader processor '{sp.name}' already exists", SourceLocation(module=sp.name))
        self.modules[sp.name] = Shader.build_map(sp.module)
        self.dps_graph[sp.name] = sp.dps.copy()

    def build_all(self):
        return {
            sp: self._build(sp)
            for sp in self.modules.keys()
        }

    def resolve_dependencies(self, name: str) -> list[str]:
        """Return `name`'s full transitive dependency list (name included) in topological order.

        Raises `TlangDependencyError` on a circular or missing dependency. Results are memoized per module name.
        """
        if (cached := self._resolved.get(name)) is not None: return cached

        ret, visited, tree = [], set(), set()
        def dfs(node: str):
            if node in visited: return
            if node in tree: raise TlangDependencyError(f"Circular dependency detected at '{node}'", SourceLocation(module=node))

            tree.add(node)
            for dep in self.dps_graph.get(node, set()):
                if dep not in self.modules:
                    raise TlangDependencyError(f"Missing dependency: '{dep}'", SourceLocation(module=node))
                dfs(dep)

            tree.remove(node)
            visited.add(node)
            ret.append(node)

        dfs(name)
        self._resolved[name] = ret
        return ret

    def _create_environment(self) -> None:
        self._env = Environment(
            loader=DictLoader(self.modules),
            autoescape=False,
            trim_blocks=False,
            lstrip_blocks=False,
            # render() runs per source line; stripping the trailing newline would merge lines and
            # break line-anchored directives (e.g. #pragma, #define).
            keep_trailing_newline=True,
            # An unknown {{ NAME }} raises immediately instead of rendering as the literal name.
            undefined=StrictUndefined,
        )

    def render(self, text: str, module: str, line: int | None = None) -> str:
        """Render one snippet of GLSL text against the project constants, substituting `{{ NAME }}`.

        Raises `TlangDependencyError` (naming `module`/`line`) instead of letting a
        `jinja2.UndefinedError`/`TemplateSyntaxError` escape.

        Note: raw GLSL is rendered directly through Jinja, so GLSL that itself contains `{{ }}`/`{% %}`
        syntax (e.g. `mat2({{1.0, 0.0}, {0.0, 1.0}})`) will fail to parse here.
        """
        if self._env is None: self._create_environment()
        assert self._env is not None

        try:
            tmpl = self._env.from_string(text)
            return tmpl.render(**self.constants)
        except TemplateSyntaxError as e:
            raise TlangDependencyError(
                f"Template syntax error while rendering '{module}': {e.message}", SourceLocation(module, line or e.lineno)
            ) from e
        except UndefinedError as e:
            raise TlangDependencyError(
                f"Undefined template constant in '{module}': {e.message}", SourceLocation(module, line)
            ) from e

    def _build(self, name: str) -> str:
        deps = self.resolve_dependencies(name)
        includes = [f"{{% include '{dep}' %}}" for dep in deps if dep != name]
        return self.render('\n'.join(includes + [self.modules[name]]), name)
