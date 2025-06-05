import re
from collections import defaultdict
import logging

logger = logging.getLogger(__name__)

# matches program declaration in the form: [prog(name) : @vert=func_name @frag=func_name]
PROG_PATTERN = re.compile(
    r"""
    ^\s*
    \[ \s*prog\(\s*'(?P<name>\w+)'\s*\)\s*        # “[prog('test')]”
    :                                             # “:” separator
    (?P<stages>.+?)                               # the “@vert='vs_main' @frag='fs_main' …”
    \]\s*
    """,
    re.VERBOSE | re.MULTILINE
)

# matches stages in the program declaration
STAGE_PATTERN = re.compile(
    r"""@(?P<stage>\w+)\s*=\s*'(?P<func>\w+)'""",
    re.VERBOSE
)

class ProgramManager:
    @staticmethod
    def find_programs(src: str) -> tuple[str, dict[str, dict]]:
        logger.debug("Searching for programs")
        programs = defaultdict(dict)

        # find all prog matches
        for m in PROG_PATTERN.finditer(src):
            name = m.group('name')
            stages = m.group('stages')

            stages_dict: dict[str, str] = {}
            for sm in STAGE_PATTERN.finditer(stages):
                stage = sm.group('stage').lower()
                func = sm.group('func')
                stages_dict[stage] = func

            programs[name] = stages_dict
            logger.debug("Found program %s", name)

        # remove the program matches and return the stripped src and the program list
        return PROG_PATTERN.sub('', src), programs
