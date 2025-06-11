import re
import logging

logger = logging.getLogger(__name__)

# i.e. T, V, X, whatever
GENERIC_TYPE_PATTERN = re.compile(r'^[A-Z]\w*$')
GENERIC_STRUCT_PATTERN = re.compile(
    r'struct\s+(?P<name>\w+)<(?P<params>[^>]+)>\s*\{(?P<body>[\s\S]*?)\};',
    re.DOTALL
)
GENERIC_FUNC_PATTERN = re.compile(
    r'generic<(?P<params>[^>]+)>\s*'
    r'(?P<ret>\w+)\s+'
    r'(?P<name>\w+)\s*\((?P<args>[^)]*)\)\s*\{(?P<body>[\s\S]*?)\}',
    re.DOTALL
)

# matches something like Base< T1, T2 > or Base<Outer<A,B>,C>
SPECIALIZATION_PATTERN = re.compile(
    r'(?P<base>\w+)<(?P<types>(?:[^<>]|<[^<>]*>)+)>'
)

IMPLICIT_GENERIC_FUNC = re.compile(
    r'^(?P<ret>[A-Z]\w*)\s+'      # return type starts with uppercase
    r'(?P<name>\w+)\s*\((?P<args>[^)]*)\)\s*\{',
    re.MULTILINE
)

SIMPLE_CALL = re.compile(r'(?P<name>\w+)\s*\((?P<args>[^)]*)\)')

_BUILTINS = {
    'void','bool','int','uint','float','double',
    'vec2','vec3','vec4','ivec2','ivec3','ivec4',
    'uvec2','uvec3','uvec4','dvec2','dvec3','dvec4',
}

class GenericsManager:
    @staticmethod
    def normalize_type_string(t: str) -> str:
        s = re.sub(r"[<>,\s]+", "_", t)
        s = re.sub(r"_+", "_", s)
        return s.strip("_")

    @staticmethod
    def split_types(type_list_str: str) -> list[str]:
        out, current, stack = [], [], []
        for ch in type_list_str:
            if ch == '<':
                stack.append('<'); current.append(ch)
            elif ch == '>':
                if stack: stack.pop(); current.append(ch)
            elif ch == ',' and not stack:
                out.append(''.join(current).strip()); current = []
            else:
                current.append(ch)
        if current: out.append(''.join(current).strip())
        return out

    @staticmethod
    def build_param_map(param_list: str, concrete_types: tuple[str, ...]) -> dict[str, str]:
        names = [p.strip() for p in param_list.split(',')]
        return dict(zip(names, concrete_types))

    @staticmethod
    def _parse_specialization(type_str: str):
        m = re.match(r'(\w+)<', type_str)
        if not m: return None
        base = m.group(1)
        start = m.end(); depth = 1
        for i, c in enumerate(type_str[start:], start):
            if c == '<': depth += 1
            elif c == '>': depth -= 1
            if depth == 0:
                inner = type_str[start:i]
                return base, inner
        return None

    @staticmethod
    def _replace_generics(src: str, base: str, concrete_name: str) -> str:
        pattern = f"{base}<"
        i, result = 0, ""
        while True:
            idx = src.find(pattern, i)
            if idx == -1:
                result += src[i:]; break
            result += src[i:idx]
            j = idx + len(pattern); depth = 1
            while j < len(src) and depth > 0:
                if src[j] == '<': depth += 1
                elif src[j] == '>': depth -= 1
                j += 1
            result += concrete_name
            i = j
        return result

    @staticmethod
    def inject_wrappers(src: str) -> str:
        def wrap(m: re.Match) -> str:
            ret_t, args_str = m.group('ret'), m.group('args')
            types = [ret_t] + [w.strip().split(None,1)[0] for w in args_str.split(',') if w.strip()]
            params, seen = [], set()
            for t in types:
                if t in seen: continue
                seen.add(t)
                if t not in _BUILTINS and GENERIC_TYPE_PATTERN.match(t):
                    params.append(t)
            if not params: return m.group(0)
            return f"generic<{','.join(params)}> {m.group(0)}"
        return IMPLICIT_GENERIC_FUNC.sub(wrap, src)

    @classmethod
    def process(cls, src: str) -> str:
        # 1) Find declarations
        structs = {m.group('name'):m for m in GENERIC_STRUCT_PATTERN.finditer(src)}
        funcs   = {m.group('name'):m for m in GENERIC_FUNC_PATTERN.finditer(src)}

        # 2) Gather specs
        specs = set((b, tuple(GenericsManager.split_types(t))) for b,t in SPECIALIZATION_PATTERN.findall(src))
        final_specs = []
        for base, types in specs:
            param_names = None
            if base in structs:
                param_names = [p.strip() for p in structs[base].group('params').split(',')]
            elif base in funcs:
                param_names = [p.strip() for p in funcs[base].group('params').split(',')]
            if param_names and set(types)==set(param_names): continue
            final_specs.append((base, types))

        # 3) Process structs and funcs
        src2, struct_chunks = cls.process_structs(src, structs, final_specs)
        src3, func_chunks   = cls.process_funcs(src2, funcs, final_specs)

        # 4) Strip definitions
        src_clean = GENERIC_STRUCT_PATTERN.sub('', src3)
        src_clean = GENERIC_FUNC_PATTERN.sub('', src_clean)
        src_clean = re.sub(r'\n\s*\n+', '\n\n', src_clean).strip()

        return '\n'.join(struct_chunks + func_chunks) + '\n\n' + src_clean + '\n'

    @classmethod
    def process_structs(
        cls,
        src: str,
        structs: dict[str, re.Match],
        final_specs: list[tuple[str, tuple[str, ...]]]
    ) -> tuple[str, list[str]]:
        # 1) Build lookup for specs
        spec_set = set(final_specs)

        # 2) Determine dependencies using manual parse for nested generics
        deps: dict[tuple[str, tuple[str, ...]], list[tuple[str, tuple[str, ...]]]] = {}
        for spec in final_specs:
            base, types = spec
            dep_list: list[tuple[str, tuple[str, ...]]] = []
            for t in types:
                parsed = cls._parse_specialization(t.strip())
                if not parsed:
                    continue
                dep_base, raw_inner = parsed
                inner_types = tuple(GenericsManager.split_types(raw_inner))
                dep_spec = (dep_base, inner_types)
                if dep_spec in spec_set:
                    dep_list.append(dep_spec)
            deps[spec] = dep_list

        # 3) Topological sort (unchanged)
        sorted_specs, visited = [], {}
        def dfs(node):
            state = visited.get(node, 0)
            if state == 1:
                raise RuntimeError(f"Cyclic dependency at {node}")
            if state == 2:
                return
            visited[node] = 1
            for d in deps[node]: dfs(d)
            visited[node] = 2
            sorted_specs.append(node)
        for spec in final_specs:
            if visited.get(spec, 0) == 0:
                dfs(spec)

        working_src = GENERIC_STRUCT_PATTERN.sub('', src)
        generated_chunks = []

        # 4) Emit structs and methods
        for base, types in sorted_specs:
            if base not in structs:
                continue
            sm = structs[base]
            raw_body = sm.group('body')
            
            # Map generic parameters to concrete types
            raw_mapping = cls.build_param_map(sm.group('params'), types)
            mapping = {p: GenericsManager.normalize_type_string(t) for p, t in raw_mapping.items()}

            # 4b) Extract any constructors from raw_body
            constructors: list[dict[str, str]] = []
            ctr_pattern = re.compile(
                rf'\b{base}\s*\((?P<args>[^)]*)\)\s*\{{(?P<body>[\s\S]*?)\}}',
                re.DOTALL
            )
            def extract_ctor(m: re.Match) -> str:
                constructors.append({'args': m.group('args'), 'body': m.group('body')})
                return ''
            fields_no_ctors = ctr_pattern.sub(extract_ctor, raw_body)

            # 4c) Extract any member methods from raw_body (after removing ctors)
            methods: list[dict[str, str]] = []
            mem_fn_pattern = re.compile(
                r'(?P<ret>\w+)\s+(?P<name>\w+)\s*\((?P<args>[^)]*)\)\s*\{(?P<body>[\s\S]*?)\}',
                re.DOTALL
            )
            def extract_method(m: re.Match) -> str:
                methods.append({
                    'ret': m.group('ret'),
                    'name': m.group('name'),
                    'args': m.group('args'),
                    'body': m.group('body')
                })
                return ''
            fields_only = mem_fn_pattern.sub(extract_method, fields_no_ctors).strip()

            # 4d) Replace every generic parameter in the remaining field‐definitions with its concrete name
            for p, t in mapping.items():
                fields_only = re.sub(rf'\b{p}\b', t, fields_only)

            # Build concrete name
            normalized_types = [GenericsManager.normalize_type_string(t) for t in types]
            concrete_name = f"{base}_{'_'.join(normalized_types)}"

            # 4f) Indent each line in fields_only by 4 spaces
            field_lines = [line.strip() for line in fields_only.splitlines() if line.strip()]
            indent = "    "
            struct_body = "\n".join(indent + line for line in field_lines)

            # 4g) Emit the actual `struct CONCRETE_NAME { … };`
            struct_chunk = (
                f"struct {concrete_name} {{\n"
                f"{struct_body}\n"
                f"}};"
            )
            generated_chunks.append(struct_chunk)

            # 4h) Emit each member‐method as a standalone free function, replacing “this.” with “s.”
            for meth in methods:
                ret, name, args, body = meth['ret'], meth['name'], meth['args'], meth['body']
                for p, t in mapping.items():
                    ret = re.sub(rf'\b{p}\b', t, ret)
                    args = re.sub(rf'\b{p}\b', t, args)
                    body = re.sub(rf'\b{p}\b', t, body)
                body = re.sub(r'\bthis\.', 's.', body).strip()

                # Build signature
                if args.strip():
                    fn_sig = f"{ret} {concrete_name}_{name}(inout {concrete_name} s, {args})"
                else:
                    fn_sig = f"{ret} {concrete_name}_{name}(inout {concrete_name} s)"

                # Indent the body lines
                body_lines = [line.rstrip() for line in body.splitlines() if line.strip()]
                method_body = "\n".join(indent + line for line in body_lines)

                method_chunk = (
                    f"{fn_sig} {{\n"
                    f"{method_body}\n"
                    f"}}"
                )
                generated_chunks.append(method_chunk)

                # Replace any “instance.method(” → “concreteName_method(instance,” in the working source
                member_call = rf'(?P<inst>\b\w+)\.{name}\s*\('
                replacement = rf'{concrete_name}_{name}(\g<inst>,'
                working_src = re.sub(member_call, replacement, working_src)

            # 4i) Emit each constructor as a standalone function
            for ctor in constructors:
                args, body = ctor['args'], ctor['body']
                for p, t in mapping.items():
                    args = re.sub(rf'\b{p}\b', t, args)
                    body = re.sub(rf'\b{p}\b', t, body)
                body = re.sub(r'\bthis\.', 's.', body).strip()

                if args.strip():
                    ctor_sig = f"{concrete_name} {concrete_name}_ctor({args})"
                else:
                    ctor_sig = f"{concrete_name} {concrete_name}_ctor()"

                ctor_body_lines = [line.rstrip() for line in body.splitlines() if line.strip()]
                ctor_full_body = [f"{concrete_name} s;"] + ctor_body_lines + ["return s;"]
                ctor_body = "\n".join(indent + line for line in ctor_full_body)

                ctor_chunk = (
                    f"{ctor_sig} {{\n"
                    f"{ctor_body}\n"
                    f"}}"
                )
                generated_chunks.append(ctor_chunk)

            # 4j) Replace all “Base<…>” references in the working_src with the new concrete name
            escaped_types = [re.escape(t) for t in types]
            sep = r'\s*,\s*'.join(escaped_types)
            # After generating chunks, replace nested generics in source
            working_src = cls._replace_generics(working_src, base, concrete_name)
            # Replace constructor calls
            working_src = re.sub(
                rf"\b{re.escape(concrete_name)}\s*\(",
                f"{concrete_name}_ctor(",
                working_src
            )

        return working_src, generated_chunks

    @classmethod
    def process_funcs(cls, src: str, funcs: dict[str, re.Match], final_specs: list[tuple[str, tuple[str, ...]]]) -> tuple[str, list[str]]:
        generated_chunks, working_src = [], src
        indent = '    '

        # Sort to honor dependencies if needed
        for base, types in sorted(final_specs):
            if base not in funcs: continue
            fm = funcs[base]
            raw_mapping = cls.build_param_map(fm.group('params'), types)
            mapping = {p: GenericsManager.normalize_type_string(t) for p,t in raw_mapping.items()}

            # Create concrete function name
            normalized_types = [GenericsManager.normalize_type_string(t) for t in types]
            concrete_name = f"{base}_{'_'.join(normalized_types)}"

            # Substitute body
            ret, args, body = fm.group('ret'), fm.group('args'), fm.group('body')
            for p,t in mapping.items():
                ret  = re.sub(rf"\b{p}\b", t, ret)
                args = re.sub(rf"\b{p}\b", t, args)
                body = re.sub(rf"\b{p}\b", t, body)

            # Emit specialized function
            lines = [line.strip() for line in body.splitlines() if line.strip()]
            fn_body = '\n'.join(indent + line for line in lines)
            fn_sig  = f"{ret} {concrete_name}({args})"
            generated_chunks.append(f"{fn_sig} {{\n{fn_body}\n}}")

            # Replace all nested calls in working_src
            # e.g., Base<T1, Base<T2>>(...)
            working_src = cls._replace_generics(working_src, base, concrete_name)
            # Also strip remaining angle instantiations without args
            angle_pat = rf"\b{base}<[^>]+>"
            working_src = re.sub(angle_pat, concrete_name, working_src)

        return working_src, generated_chunks