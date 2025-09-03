# -------------------------------------------------------------
# @file          binding_registry.py
# @author        Priyangkar Ghosh
# @created       2025-07-13
# @description   Goes through the .tlang source file and finds 
#                buffer declarations. Assigns buffer bindings
#                based on usage, when NOT user assigned.
# @license       MIT
# -------------------------------------------------------------

from collections import defaultdict
from bitarray import bitarray
from moderngl import Context
import regex as re

LAYOUT_PATTERN = re.compile(
    r"layout\s*\(\s*([^)]*?)\s*\)\s*"
    r"(?:readonly|coherent|volatile|restrict\s+)*buffer\s+(\w+)\s*{",
    re.MULTILINE,
)
BINDING_PATTERN = re.compile(r"\bbinding\s*=\s*(\d+)\b")

class BindingRegistry():
    @staticmethod
    def inject_bindings(ctx: Context, modules: dict[str, str]):
        # build a bit‑set of free binding points
        free_bindings = bitarray(max_bindings := ctx.info['GL_MAX_SHADER_STORAGE_BUFFER_BINDINGS'])
        free_bindings.setall(True)

        canon: dict[str, int] = defaultdict()      # final block -> binding map
        usage: dict[str, int] = defaultdict(int)   # popularity counter per block

        # scan modules and keep track of explicit bindings
        for src in modules.values():
            seen: set[str] = set()
            for layout_args, block in LAYOUT_PATTERN.findall(src):
                # count the block only once per module
                if block not in seen:
                    usage[block] += 1
                    seen.add(block)

                # record bindings
                if (m := BINDING_PATTERN.search(layout_args)):
                    bind = int(m.group(1))
                    if bind >= max_bindings: raise ValueError(f"Binding {bind} exceeds GL limit ({max_bindings})")

                    prev = canon.setdefault(block, bind)
                    if prev != bind: raise ValueError(f"Block '{block}' has conflicting bindings ({prev} vs {bind})")
                    free_bindings[bind] = False # reserve the slot

        # assign bindings from most popular to least popular
        billboard = sorted(
            (b for b in usage if b not in canon),
            key=lambda b: (usage[b], b),  # primary: usage desc, secondary: name
        )

        while billboard and any(free_bindings):
            if not (b := billboard.pop()): break
            if (bind := free_bindings.index(True)) == -1: break
            free_bindings[bind] = False
            canon[b] = bind

        # patch sources
        inject_pattern = re.compile(
            r"layout\s*\(\s*([^)]*?)\s*\)\s*((?:readonly|coherent|volatile|restrict)\s+)?buffer\s+(\w+)\s*{",
            re.MULTILINE,
        )

        binded_modules: dict[str, str] = {}
        for name, src in modules.items():
            # copy so we can track which indices are still unused inside this module
            local_free = free_bindings.copy()
            local_free.setall(True)

            def replacer(match: re.Match) -> str:
                layout_args, qualifier, block_name = match.group(1), match.group(2) or "", match.group(3)

                # if the layout already has an explicit binding, leave it untouched
                if BINDING_PATTERN.search(layout_args): return match.group(0)

                # try the canonical binding first; otherwise pick the next free one
                binding = canon.get(block_name)
                if binding is None or not local_free[binding]:
                    try: binding = local_free.index(True)
                    except ValueError: raise RuntimeError("Out of SSBO bindings!")

                local_free[binding] = False
                new_args = f"binding = {binding}, {layout_args}".strip().strip(',')
                return f"layout({new_args}) {qualifier}buffer {block_name} {{"

            binded_modules[name] = inject_pattern.sub(replacer, src)
        return binded_modules
    
    @staticmethod
    def remove_unused_buffers(src: str) -> str:
        # Matches complete layout buffer declarations
        buffer_pattern = re.compile(
            r'''(
                layout\s*\([^)]*\)\s*      # layout(...)
                buffer\s+\w+\s*            # buffer BlockName
                \{                         # {
                (?P<fields>.*?)            # capture buffer fields
                \}\s*;                     # };
            )''',
            re.DOTALL | re.VERBOSE
        )

        matches = list(buffer_pattern.finditer(src))
        to_remove = []

        for match in matches:
            full_decl = match.group(1)
            fields = match.group("fields")

            # extract variable names from the field block
            var_names = []
            for line in fields.splitlines():
                line = line.strip()
                if not line or line.startswith("//") or ";" not in line:
                    continue
                try:
                    type_, var_str = line.split(None, 1)
                    var_str = var_str.split(";")[0]
                    for var in var_str.split(","):
                        var = var.strip()
                        var_name = re.sub(r'\[.*?\]', '', var)  # remove [] if present
                        if var_name:
                            var_names.append(var_name)
                except ValueError:
                    continue  # malformed line

            # check if any variable is used outside this block
            rest_of_code = src[:match.start()] + src[match.end():]
            used = False
            for name in var_names:
                # match whole words only
                if re.search(r'\b' + re.escape(name) + r'\b', rest_of_code):
                    used = True
                    break

            if not used:
                to_remove.append(full_decl)

        # safely remove unused buffers
        for block in to_remove:
            src = src.replace(block, '')
        return src