import re

# matches lines in the form `#include <filename>`
INCLUDE_PATTERN = re.compile(
    r'^(?P<prefix>\s*)#include\s+<(?P<filename>.+?)>\s*\n?',
    re.MULTILINE
)

class IncludeManager:
    @staticmethod
    def refactor(src: str) -> str:
        def repl(m: re.Match) -> str:
            # extract filename from the match
            filename = m.group('filename')
            # replace it with the jinja equivalent
            return f"{{% include \"{filename}\" %}}\n"

        # sub every pattern in the original string
        return INCLUDE_PATTERN.sub(repl, src)
