import re
import logging

logger = logging.getLogger(__name__)


ENTRY_HEADER = re.compile(
    r'''
    ^\s*
    \[                                  # literal “[”
        (?P<stage>\w+)                  # stage keyword: compute, vert, frag, geom, tesc, tese, etc.
        (?:\(\s*(?P<args>[\s\S]*?)\s*\))?  # allow ANY chars (including newlines) inside “(...)”
    \]\s*
    (?P<ret_type>\w[\w\s\*]*)\s+        # return type (e.g. “void”)
    (?P<name>\w+)\s*                    # function name
    \((?P<params>[^\)]*)\)\s*           # parameter list (ignored here)
    \{                                  # opening brace of function body
    ''',
    re.MULTILINE | re.VERBOSE
)

class EntryPointManager:
    @classmethod
    def _parse_io_pairs(cls, text: str) -> list[tuple[str, str]]:
        """
        Parse a comma-separated list of “name:type” pairs (e.g. “pos:vec3, normal:vec3”).
        Returns a list of (name, type) tuples, trimming whitespace.
        """
        out = []
        for pair in text.split(','):
            pair = pair.strip()
            if not pair or ':' not in pair:
                continue
            name, typ = pair.split(':', 1)
            out.append((name.strip(), typ.strip()))
        return out

    @classmethod
    def extract(cls, src: str) -> tuple[str, list[dict[str, str]]]:
        logger.debug("Extracting entry points")
        entry_points: list[dict[str, str]] = []
        removals: list[tuple[int, int]] = []

        for m in ENTRY_HEADER.finditer(src):
            info = m.groupdict()
            stage = info['stage'].lower()
            raw_body = None
            start_idx, body_start = m.start(), m.end()
            brace_count = 1
            i = body_start
            while i < len(src) and brace_count > 0:
                if src[i] == '{': brace_count += 1
                elif src[i] == '}': brace_count -= 1
                i += 1
            raw_body = src[body_start : i - 1].strip()

            decl_lines: list[str] = []
            args = info.get('args') or ""

            if stage in ('vert', 'vertex'):
                parts = [p.strip() for p in args.split(';') if p.strip()]
                inputs_text = outputs_text = ""
                for part in parts:
                    if re.match(r'^inputs\s*=', part, re.IGNORECASE):
                        inputs_text = re.sub(r'^inputs\s*=\s*', '', part, flags=re.IGNORECASE)
                    elif re.match(r'^outputs\s*=', part, re.IGNORECASE):
                        outputs_text = re.sub(r'^outputs\s*=\s*', '', part, flags=re.IGNORECASE)

                inputs = cls._parse_io_pairs(inputs_text)
                outputs = cls._parse_io_pairs(outputs_text)

                for name, typ in inputs:
                    base, *ann = typ.split('@')
                    if ann and ann[0].startswith('location'):
                        loc = ann[0].split('=', 1)[1]
                        decl_lines.append(f"layout(location = {loc}) in {base} {name};")
                    else:
                        decl_lines.append(f"in {base} {name};")

                for name, typ in outputs:
                    base, *ann = typ.split('@')
                    if ann and ann[0].startswith('location'):
                        loc = ann[0].split('=', 1)[1]
                        decl_lines.append(f"layout(location = {loc}) out {base} {name};")
                    else:
                        decl_lines.append(f"out {base} {name};")

            elif stage in ('frag', 'fragment'):
                parts = [p.strip() for p in args.split(';') if p.strip()]
                inputs_text = outputs_text = ""
                for part in parts:
                    if re.match(r'^inputs\s*=', part, re.IGNORECASE):
                        inputs_text = re.sub(r'^inputs\s*=\s*', '', part, flags=re.IGNORECASE)
                    elif re.match(r'^outputs\s*=', part, re.IGNORECASE):
                        outputs_text = re.sub(r'^outputs\s*=\s*', '', part, flags=re.IGNORECASE)

                inputs = cls._parse_io_pairs(inputs_text)
                outputs = cls._parse_io_pairs(outputs_text)

                for name, typ in inputs:
                    base, *ann = typ.split('@')
                    decl_lines.append(f"in {base} {name};")
                for name, typ in outputs:
                    base, *ann = typ.split('@')
                    decl_lines.append(f"out {base} {name};")

            elif stage in ('geom', 'geometry'):
                # split on semicolons
                parts = [p.strip() for p in args.split(';') if p.strip()]

                # collect three buckets
                inputs_text = ""
                in_prim    = None
                out_prim   = None
                maxverts   = None

                for part in parts:
                    low = part.lower()
                    if low.startswith('inputs='):
                        # grab everything after the '='
                        inputs_text = part.split('=',1)[1].strip()
                    elif low.startswith('in='):
                        in_prim = part.split('=',1)[1].strip()
                    elif low.startswith('out='):
                        # out might carry the @maxverts suffix
                        val = part.split('=',1)[1].strip()
                        if ':' in val:
                            prim, attr = val.split(':',1)
                            out_prim = prim.strip()
                            if '@maxverts' in attr:
                                maxverts = attr.split('=',1)[1].strip()
                        else:
                            out_prim = val
                    elif low.startswith('@maxverts'):
                        maxverts = part.split('=',1)[1].strip()

                # now emit declarations:
                # 1) per-vertex inputs become arrays
                if inputs_text:
                    for name, typ in cls._parse_io_pairs(inputs_text):
                        # e.g. "in vec3@binding=0 vert[];"
                        decl_lines.append(f"in {typ} {name}[];")

                # 2) input primitive
                if in_prim:
                    decl_lines.append(f"layout({in_prim}) in;")

                # 3) single output + max_vertices
                if out_prim:
                    if maxverts:
                        decl_lines.append(
                            f"layout({out_prim}, max_vertices = {maxverts}) out;"
                        )
                    else:
                        decl_lines.append(f"layout({out_prim}) out;")

            elif stage in ('tesc', 'tesscontrol', 'tesscontrolshader'):
                verts_match = re.search(r'vertices\s*=\s*(\d+)', args, re.IGNORECASE)
                if verts_match:
                    vcount = verts_match.group(1)
                    decl_lines.append(f"layout (vertices = {vcount}) out;")

            elif stage in ('tese', 'tesseval', 'tessevaluationshader'):
                qualifiers = [q.strip() for q in args.split(',') if q.strip()]
                if qualifiers:
                    qstr = ", ".join(qualifiers)
                    decl_lines.append(f"layout({qstr}) in;")

            else:
                pass

            # wrap everything in “void main() { … }”
            decl_block = ("\n".join(decl_lines) + "\n") if decl_lines else ""
            orig_fn = f"{info['ret_type']} {info['name']}() {{\n{raw_body}\n}}"
            full_fn = f"{decl_block}{orig_fn}\n\nvoid main() {{\n    {info['name']}();\n}}"

            info['body'] = raw_body
            info['full_function'] = full_fn
            entry_points.append(info)
            removals.append((start_idx, i))

        # strip out entry point sections
        stripped_src = src
        for start, end in sorted(removals, reverse=True):
            stripped_src = stripped_src[:start] + stripped_src[end:]

        logger.debug(f"Extracted {len(entry_points)} entry points")
        return stripped_src, entry_points