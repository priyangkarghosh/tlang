import re

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
        # Turn e.g. "test_struct_2<uint, float>" → "test_struct_2_uint_float"
        s = re.sub(r"[<>,\s]+", "_", t)   # every '<', '>', ',' or whitespace → '_'
        s = re.sub(r"_+", "_", s)         # collapse any "__" → "_"
        return s.strip("_")
    
    @staticmethod
    def split_types(type_list_str: str) -> list[str]:
        out: list[str] = []
        current: list[str] = []
        stack: list[str] = []

        for ch in type_list_str:
            match ch:
                case '<':
                    stack.append('<')
                    current.append(ch)
                case '>':
                    if stack: stack.pop()
                    current.append(ch)
                case ',' if not stack:
                    out.append(''.join(current).strip())
                    current = []
                case _:
                    current.append(ch)
        
        if current: out.append(''.join(current).strip())
        return out

    @staticmethod
    def inject_wrappers(src: str) -> str:
        def wrap(m: re.Match) -> str:
            ret_t = m.group('ret')
            args_str = m.group('args')

            # collect return type + each arg's type (first token before whitespace).
            types = [ret_t]
            if args_str.strip():
                for w in args_str.split(','):
                    # split once on whitespace to grab the type token
                    t = w.strip().split(None, 1)[0]
                    types.append(t)

            params = []
            seen = set()
            for t in types:
                if t in seen: continue

                # only keep t if it’s not a built in and matches uppercase-pattern
                seen.add(t)
                if t not in _BUILTINS and GENERIC_TYPE_PATTERN.match(t):
                    params.append(t)

            # check if there were any unique params
            if not params: return m.group(0)
            return f"generic<{','.join(params)}> {m.group(0)}"

        # return the replaced generic functions
        return IMPLICIT_GENERIC_FUNC.sub(wrap, src)

    @staticmethod
    def build_param_map(param_list: str, concrete_types: tuple[str, ...]) -> dict[str, str]:
        names = [p.strip() for p in param_list.split(',')]
        return dict(zip(names, concrete_types))

    @classmethod
    def process(cls, src: str) -> str:
        src = cls.inject_wrappers(src)

        structs = {m.group('name'): m for m in GENERIC_STRUCT_PATTERN.finditer(src)}
        funcs   = {m.group('name'): m for m in GENERIC_FUNC_PATTERN.finditer(src)}

        specs: set[tuple[str, tuple[str, ...]]] = set()

        for base, tlist in SPECIALIZATION_PATTERN.findall(src):
            raw_types = GenericsManager.split_types(tlist)
            specs.add((base, tuple(raw_types)))

        for m in SIMPLE_CALL.finditer(src):
            fname = m.group('name')
            if fname not in funcs:
                continue
            arg_list = [a.strip() for a in m.group('args').split(',') if a.strip()]
            param_count = len(funcs[fname].group('params').split(','))
            if len(arg_list) != param_count:
                continue
            deduced: list[str] = []
            for a in arg_list:
                if re.match(r'\d+\.\d*', a):
                    deduced.append('float')
                elif re.match(r'\d+', a):
                    deduced.append('int')
                else:
                    deduced.append('uint')
            specs.add((fname, tuple(deduced)))

        final_specs: list[tuple[str, tuple[str, ...]]] = []
        for base, types in specs:
            if base in structs:
                param_names = [p.strip() for p in structs[base].group('params').split(',')]
            elif base in funcs:
                param_names = [p.strip() for p in funcs[base].group('params').split(',')]
            else:
                param_names = None
            if param_names and set(types) == set(param_names):
                continue
            final_specs.append((base, types))

        src, struct_chunks = cls.process_structs(src, structs, final_specs)

        src, func_chunks = cls.process_funcs(src, funcs, final_specs)

        src = GENERIC_STRUCT_PATTERN.sub('', src)
        src = GENERIC_FUNC_PATTERN.sub('', src)
        src = re.sub(r'\n\s*\n\s*\n*', '\n\n', src).strip()

        all_chunks = struct_chunks + func_chunks
        return '\n'.join(all_chunks) + '\n\n' + src + '\n'

    @classmethod
    def process_structs(
        cls,
        src: str,
        structs: dict[str, re.Match],
        final_specs: list[tuple[str, tuple[str, ...]]]
    ) -> tuple[str, list[str]]:
        generated_chunks: list[str] = []

        for base, types in sorted(final_specs):
            if base not in structs:
                continue

            sm = structs[base]
            raw_body = sm.group('body')
            mapping = cls.build_param_map(sm.group('params'), types)

            constructors: list[dict[str, str]] = []
            ctr_pattern = re.compile(
                rf'\b{base}\s*\((?P<args>[^)]*)\)\s*\{{(?P<body>[\s\S]*?)\}}',
                re.DOTALL
            )
            def extract_ctor(m: re.Match) -> str:
                constructors.append({'args': m.group('args'), 'body': m.group('body')})
                return ''
            fields_no_ctors = ctr_pattern.sub(extract_ctor, raw_body)

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

            for p, t in mapping.items():
                fields_only = re.sub(rf'\b{p}\b', t, fields_only)

            concrete_name = f"{base}_{'_'.join(GenericsManager.normalize_type_string(t) for t in types)}"
            generated_chunks.append(f"struct {concrete_name} {{ {fields_only} }};")

            for meth in methods:
                ret, name, args, body = meth['ret'], meth['name'], meth['args'], meth['body']
                for p, t in mapping.items():
                    ret  = re.sub(rf'\b{p}\b', t, ret)
                    args = re.sub(rf'\b{p}\b', t, args)
                    body = re.sub(rf'\b{p}\b', t, body)
                body = re.sub(r'\bthis\.', 's.', body)

                if args.strip():
                    fn_sig = f"{ret} {concrete_name}_{name}(inout {concrete_name} s, {args})"
                else:
                    fn_sig = f"{ret} {concrete_name}_{name}(inout {concrete_name} s)"
                generated_chunks.append(f"{fn_sig} {{ {body} }}")

                member_call = rf'(?P<inst>\b\w+)\.{name}\s*\('
                replacement = rf'{concrete_name}_{name}(\g<inst>,'
                src = re.sub(member_call, replacement, src)

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
                full_body = f"{concrete_name} s; {body} return s;"
                generated_chunks.append(f"{ctor_sig} {{ {full_body} }}")

            escaped_types = [re.escape(t) for t in types]
            sep = r'\s*,\s*'.join(escaped_types)
            angle_pattern = rf'\b{re.escape(base)}<\s*{sep}\s*>'
            src = re.sub(angle_pattern, concrete_name, src)

            ctor_call_pattern = rf'\b{concrete_name}\s*\('
            src = re.sub(ctor_call_pattern, f"{concrete_name}_ctor(", src)

        return src, generated_chunks

    @classmethod
    def process_funcs(
        cls,
        src: str,
        funcs: dict[str, re.Match],
        final_specs: list[tuple[str, tuple[str, ...]]]
    ) -> tuple[str, list[str]]:
        generated_chunks: list[str] = []

        for base, types in sorted(final_specs):
            if base not in funcs:
                continue

            fm = funcs[base]
            ret, args, body = fm.group('ret'), fm.group('args'), fm.group('body')
            mapping = cls.build_param_map(fm.group('params'), types)

            for p, t in mapping.items():
                ret  = re.sub(rf'\b{p}\b', t, ret)
                args = re.sub(rf'\b{p}\b', t, args)
                body = re.sub(rf'\b{p}\b', t, body)

            concrete_name = f"{base}_{'_'.join(GenericsManager.normalize_type_string(t) for t in types)}"
            generated_chunks.append(f"{ret} {concrete_name}({args}) {{ {body} }}")

            escaped_types = [re.escape(t) for t in types]
            sep = r'\s*,\s*'.join(escaped_types)
            angle_pattern = rf'\b{re.escape(base)}<\s*{sep}\s*>'
            fn_call_pattern = angle_pattern + r'\s*\('
            src = re.sub(fn_call_pattern, f"{concrete_name}(", src)
            src = re.sub(angle_pattern, concrete_name, src)

        return src, generated_chunks