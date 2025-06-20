import logging
logger = logging.getLogger(__name__)

import os
import time
import inspect
from glob import glob
from pathlib import Path

from moderngl import Context
from tlang.dependency_manager import DependencyManager
from tlang.shader import Shader
from tlang.shader_processor import ShaderProcessor


FILE_EXT = '.tlang'
class ShaderManager:
    def __init__(self, ctx: Context, version: str, dir: str, constants: dict = {}) -> None:
        # start time for shader manager
        t0 = time.perf_counter()

        # check if the file path given is absolute
        # -> if not, convert it to an absolute dir   
        path = Path(dir)
        if not path.is_absolute():
            frame = inspect.stack()[1]
            path = (Path(frame.filename).resolve().parent / dir).resolve()

        # process each file in the dir
        dm = DependencyManager(constants)
        processors: dict[str, ShaderProcessor] = {}
        pattern = os.path.join(dir, f'**/*{FILE_EXT}')
        for fp in glob(pattern, recursive=True):
            # create the new processor
            processors[name] = process = ShaderProcessor(
                name := (fp := Path(fp)).relative_to(dir).with_suffix('').as_posix(), 
                fp.read_text()
            )

            dm.register(process)     

        # build each common src
        commons = dm.build_all()

        # create a shader obj for each shader
        self._shaders: dict[str, Shader] = {}
        for name, common in commons.items():
            process = processors[name]

            # consolidate all extensions (including extensions for dependencies)
            for dp in process.dps: 
                process.ext.update(processors[dp].ext)

            # create the shader obj
            self._shaders[name] = Shader(
                ctx, name, version, common, process
            )
        
        # elapsed time for shader manager for logging
        t1 = time.perf_counter()
        logging.info(
            'Built and compiled %d shaders in %.2f seconds', 
            len(self._shaders), t1 - t0
        )
    
    def __getitem__(self, name: str) -> Shader | None:
        return self.get_shader(name)

    def __contains__(self, value: str) -> bool:
        return value in self._shaders
    
    def get_shader(self, name: str) -> Shader | None:
        return self._shaders.get(name)

