import re
import logging

logger = logging.getLogger(__name__)

# matches lines in the form `#include <filename>`
INCLUDE_PATTERN = re.compile(
    r'^(?P<prefix>\s*)#include\s+<(?P<filename>.+?)>\s*\n?',
    re.MULTILINE
)

class IncludeManager:
    @staticmethod
    def refactor(src: str) -> str:
        logger.debug("Processing includes")

        def repl(m: re.Match) -> str:
            # extract filename from the match
            filename = m.group('filename')
            logger.debug("Found include: %s", filename)
            # replace it with the jinja equivalent
            return f"{{% include \"{filename}\" %}}\n"

        # sub every pattern in the original string
        return INCLUDE_PATTERN.sub(repl, src)
