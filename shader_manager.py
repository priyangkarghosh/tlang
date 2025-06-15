from glob import glob
import os
from pathlib import Path

from moderngl import Context
from dependency_manager import DependencyManager
from shader import Shader
from shader_processor import ShaderProcessor

FILE_EXT = '.tlang'
class ShaderManager:
    def __init__(self, ctx: Context, version: str, dir: str, constants: dict = {}) -> None:
        self.dm = DependencyManager(constants)

        # process each file in the 
        processors: dict[str, ShaderProcessor] = {}
        pattern = os.path.join(dir, f'**/*{FILE_EXT}')
        for file_path in glob(pattern, recursive=True):
            # create the new processor
            processors[name] = process = ShaderProcessor(
                name := os.path.relpath(file_path, dir).replace('\\', '/').replace(FILE_EXT, ''), 
                Path(file_path).read_text()
            )

            self.dm.register(process)     

        # build each common src
        commons = self.dm.build_all()

        # create a shader obj for each shader
        self.shaders: dict[str, Shader] = {}
        for name, common in commons.items():
            process = processors[name]

            # consolidate all extensions (including extensions for dependencies)
            for dp in process.dps: 
                process.ext.update(processors[dp].ext)

            # create the shader obj
            self.shaders[name] = Shader(
                ctx, version, common, process
            )

