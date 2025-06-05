from glob import glob
import os
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

from moderngl import Context

from glsl_preprocessor.entry_point_manager import EntryPointManager
from glsl_preprocessor.generics_manager import GenericsManager
from glsl_preprocessor.include_manager import IncludeManager
from glsl_preprocessor.memory_manager import MemoryManager
from glsl_preprocessor.program_manager import ProgramManager
from glsl_preprocessor.template_renderer import TemplateManager

FILE_EXTENSION = '.tlang'

class Processor():
    def __init__(self, base_dir: str, constants: dict[str, object] | None = None) -> None:
        logger.info("Initializing processor with %s", base_dir)
        # load sources using the base directory
        self._base_dir = base_dir
        self._constants = constants or {}
        self._sources = self.load_sources() # raw shader files

        # create data dicts
        self._entries: dict[str, list[dict]] = {}
        self._programs: dict[str, dict[str, dict]] = {}
        self._processed_sources: dict[str, str] = {} # processed shader files

        # process raw sources
        self.process_sources()
        logger.info("Processed %d sources", len(self._sources))

    # load sources
    # -> also process any "includes" while reading
    def load_sources(self) -> dict[str, str]:
        logger.debug("Loading shader sources")
        sources: dict[str, str] = {}
        pattern = os.path.join(self._base_dir, f'**/*{FILE_EXTENSION}')
        for file_path in glob(pattern, recursive=True):
            rel_name = os.path.relpath(file_path, self._base_dir).replace('\\', '/')
            raw_src = Path(file_path).read_text()
            refac_src = IncludeManager.refactor(raw_src)
            sources[rel_name] = refac_src
            logger.debug("Read %s", rel_name)
        return sources
    
    def process_sources(self):
        for src_name, src in self._sources.items():
            logger.debug("Processing %s", src_name)
            # render shader for "includes" and other constants
            src = TemplateManager(self._sources, self._constants).render(src_name)

            # find programs in the shader and strip the declarations
            src, self._programs[src_name] = ProgramManager.find_programs(src)
            
            # refactor shader buffer, shared, and atomic counter declarations
            src = MemoryManager.refactor(src)

            # generics
            src = GenericsManager.process(src)

            # entry points
            src, self._entries[src_name] = EntryPointManager.extract(src)
            self._processed_sources[src_name] = src
            logger.debug("Finished %s", src_name)
