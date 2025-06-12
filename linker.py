import re

# matches lines in the form `#include <filename>`
INCLUDE_PATTERN = re.compile(
    r'^(?P<prefix>\s*)#include\s+<(?P<filenames>.+?)>\s*\n?',
    re.MULTILINE
)

class Linker:
    @staticmethod
    def _process_includes(src: str):
        pass

    @staticmethod
    def _process_extensions(src: str, src_map: dict[str, str]):
        for index, line in src_map.items():
            out_line = ''
            last_match = 0
            for match in EXTEND_PATTERN.finditer(line):
                required = match.group(1) == '!'
                content = match.group(2)

                # multiple extensions separated by commas
                extensions = [s.strip() for s in content.split(',') if s.strip()]
                resolved_lines = []

                for ext in extensions:
                    if ext in EXTENSION_ALIASES:
                        for real_ext in EXTENSION_ALIASES[ext]:
                            mode = EXTENSION_DEFAULTS.get(real_ext, 'enable')
                            resolved_lines.append(f'#extension {real_ext} : {mode}')
                    else:
                        mode = 'require' if required else EXTENSION_DEFAULTS.get(ext, 'enable')
                        resolved_lines.append(f'#extension {ext} : {mode}')

                out_line += line[last_match:match.start()] + '\n'.join(resolved_lines) + '\n'
                last_match = match.end()

            # restore rest of line if anything changed
            if last_match:
                out_line += line[last_match:] + '\n'
                src_map[index] = out_line
