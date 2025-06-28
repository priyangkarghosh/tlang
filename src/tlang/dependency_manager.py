from collections import deque
import logging
logger = logging.getLogger(__name__)

from jinja2 import DictLoader, Environment, TemplateNotFound, Undefined
from tlang.shader import Shader
from tlang.shader_processor import ShaderProcessor

class LogUndefined(Undefined):
    def __str__(self):
        logger.warning(f"Missing constant: '{self._undefined_name}'")
        return self._undefined_name

    def __repr__(self):
        return str(self)

class DependencyManager:
    def __init__(self, constants) -> None:
        self.modules: dict[str, str] = {}
        self.dps_graph: dict[str, set[str]] = {}
        self.constants = constants
        self._env: Environment | None = None

    def register(self, sp: ShaderProcessor) -> None:
        if sp.name in self.modules:
            raise ValueError(f"Shader processor '{sp.name}' already exists")
        self.modules[sp.name] = Shader.build_map(sp.module)
        self.dps_graph[sp.name] = sp.dps.copy()
    
    def build_all(self):
        return { 
            sp: self._build(sp) 
            for sp in self.modules.keys() 
        }

    def _create_environment(self) -> None:
        self._env = Environment(
            loader=DictLoader(self.modules),
            autoescape=False,
            trim_blocks=False,
            lstrip_blocks=False,
            undefined=LogUndefined,
        )

    def _build(self, name: str) -> str:
        if self._env is None: self._create_environment()
        assert self._env is not None
        
        def resolve_dependencies(name: str) -> list[str]:
            ret, visited, tree = [], set(), set()
            def dfs(node: str):
                # stop recursing if this node has already been visited
                if node in visited: return

                # if this node has already been added to the link tree, its a circular import
                if node in tree: raise ValueError(f"Circular dependency detected at '{node}'")

                # continue dfs through each dps this node has
                tree.add(node)
                for dep in self.dps_graph.get(node, set()):
                    if dep not in self.modules:
                        raise ValueError(f"Missing dependency: '{dep}'")
                    dfs(dep)

                # update structures
                tree.remove(node)
                visited.add(node)
                ret.append(node)

            # start dfs at the input node
            dfs(name)
            return ret

        # create dependency list
        deps = resolve_dependencies(name)
        includes = [f"{{% include '{dep}' %}}" for dep in deps if dep != name]

        # create fresh template using the module as a base
        tmpl = self._env.from_string('\n'.join(includes + [self.modules[name]]))
        return tmpl.render(**self.constants)