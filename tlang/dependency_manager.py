import logging
logger = logging.getLogger(__name__)

from jinja2 import DictLoader, Environment, TemplateNotFound
from tlang.shader import Shader
from tlang.shader_processor import ShaderProcessor


class DependencyManager:
    def __init__(self, constants) -> None:
        self.modules: dict[str, str] = {}
        self.constants = constants
        self._env: Environment | None = None

    def register(self, sp: ShaderProcessor) -> None:
        if sp.name in self.modules:
            raise ValueError(f"Shader processor '{sp.name}' already exists")
        self.modules[sp.name] = Shader.build_map(sp.module)
    
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
        )

    def _build(self, name: str) -> str:
        if self._env is None: self._create_environment()
        assert self._env is not None

        try:
            tmpl = self._env.get_template(name)
            return tmpl.render(**self.constants)
        except TemplateNotFound as exc:
            raise FileNotFoundError(
                f"Shader file '{name}' not found in template environment."
            ) from exc
