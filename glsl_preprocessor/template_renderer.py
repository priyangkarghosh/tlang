from jinja2 import DictLoader, Environment, TemplateNotFound
import logging

logger = logging.getLogger(__name__)


class TemplateManager:
    def __init__(self, sources: dict[str, str], constants: dict[str, object]):
        self._env = Environment(
            loader=DictLoader(sources),
            autoescape=False,
            trim_blocks=False,
            lstrip_blocks=False
        )
        self.constants = constants

    def render(self, shader_name: str) -> str:
        logger.debug("Rendering template %s", shader_name)
        try:
            template = self._env.get_template(shader_name)
            return template.render(**self.constants)
        except TemplateNotFound:
            raise FileNotFoundError(
                f"Shader file '{shader_name}' not found in template environment."
            )
