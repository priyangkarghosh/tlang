import re


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
        entry_points: list[dict[str, str]] = []
        removals: list[tuple[int, int]] = []

        for m in ENTRY_HEADER.finditer(src):
            start_idx = m.start()
            body_start = m.end()
            brace_count = 1
            i = body_start

            while i < len(src) and brace_count > 0:
                if src[i] == '{':
                    brace_count += 1
                elif src[i] == '}':
                    brace_count -= 1
                i += 1

            info = m.groupdict()
            raw_body = src[body_start : i - 1].strip()
            stage = info['stage'].lower()
            args = info.get('args') or ""

            decl_lines: list[str] = []

            if stage == 'compute':
                nt_match = re.match(
                    r'\s*numthreads\s*\(\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*\)\s*',
                    args
                )
                if nt_match:
                    x, y, z = nt_match.groups()
                    decl_lines.append(
                        f"layout (local_size_x = {x}, local_size_y = {y}, local_size_z = {z}) in;"
                    )

            elif stage in ('vert', 'vertex'):
                parts = [p.strip() for p in args.split(';') if p.strip()]
                inputs_text = ""
                outputs_text = ""
                for part in parts:
                    if re.match(r'^\s*inputs\s*=', part, re.IGNORECASE):
                        inputs_text = re.sub(r'^\s*inputs\s*=\s*', '', part, flags=re.IGNORECASE).strip()
                    elif re.match(r'^\s*outputs\s*=', part, re.IGNORECASE):
                        outputs_text = re.sub(r'^\s*outputs\s*=\s*', '', part, flags=re.IGNORECASE).strip()

                inputs = cls._parse_io_pairs(inputs_text)
                outputs = cls._parse_io_pairs(outputs_text)
                for iname, itype in inputs:
                    decl_lines.append(f"in {itype} {iname};")
                for oname, otype in outputs:
                    decl_lines.append(f"out {otype} {oname};")

            elif stage in ('frag', 'fragment'):
                parts = [p.strip() for p in args.split(';') if p.strip()]
                inputs_text = ""
                outputs_text = ""
                for part in parts:
                    if re.match(r'^\s*inputs\s*=', part, re.IGNORECASE):
                        inputs_text = re.sub(r'^\s*inputs\s*=\s*', '', part, flags=re.IGNORECASE).strip()
                    elif re.match(r'^\s*outputs\s*=', part, re.IGNORECASE):
                        outputs_text = re.sub(r'^\s*outputs\s*=\s*', '', part, flags=re.IGNORECASE).strip()

                inputs = cls._parse_io_pairs(inputs_text)
                outputs = cls._parse_io_pairs(outputs_text)
                for iname, itype in inputs:
                    decl_lines.append(f"in {itype} {iname};")
                for oname, otype in outputs:
                    decl_lines.append(f"out {otype} {oname};")

            elif stage in ('geom', 'geometry'):
                parts = [token.strip() for token in args.split(',') if token.strip()]
                layout_in = ""
                layout_out = ""
                maxverts = ""
                for token in parts:
                    kv = token.split('=')
                    if len(kv) != 2:
                        continue
                    key = kv[0].strip().lower()
                    val = kv[1].strip()
                    if key == 'in':
                        layout_in = f"layout({val}) in;"
                    elif key == 'out':
                        layout_out = f"layout({val}) out;"
                    elif key in ('maxverts', 'max_vertices'):
                        maxverts = f"layout(max_vertices = {val}) out;"
                for line in (layout_in, layout_out, maxverts):
                    if line:
                        decl_lines.append(line)

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
            full_fn = f"{decl_block}void main() {{\n{raw_body}\n}}"

            info['body'] = raw_body
            info['full_function'] = full_fn
            entry_points.append(info)
            removals.append((start_idx, i))

        # remove all matched entry-point blocks from the source
        stripped_src = src
        for start, end in sorted(removals, reverse=True):
            stripped_src = stripped_src[:start] + stripped_src[end:]

        return stripped_src, entry_points