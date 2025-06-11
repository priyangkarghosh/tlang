from glob import glob
import os
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

from glsl_preprocessor.processor import FILE_EXTENSION
from glsl_preprocessor.include_manager import IncludeManager


class Loader:
    @staticmethod
    def load_sources(base_dir: str) -> dict[str, str]:
        logger.info("Loading sources from %s", base_dir)
        sources: dict[str, str] = {}
        pattern = os.path.join(base_dir, f'**/*{FILE_EXTENSION}')
        for file_path in glob(pattern, recursive=True):
            rel_name = os.path.relpath(file_path, base_dir).replace('\\', '/')
            raw_text = Path(file_path).read_text()
            expanded = IncludeManager.refactor(raw_text)
            sources[rel_name] = expanded
            logger.debug("Loaded %s", rel_name)
        return sources
