# -------------------------------------------------------------
# @file          attribute_manager.py
# @author        Priyangkar Ghosh
# @created       2025-06-10
# @description   Extracts and dispatches `[attr(...)]`/`#name<args>` attributes from .tlang source.
# @license       MIT
# -------------------------------------------------------------

import logging
logger = logging.getLogger(__name__)

import regex as re
import bisect

from tlang.frontend.attribute import Attribute
from tlang.frontend.attribute_handlers import REGISTRY
from tlang.frontend.attribute_registry import AttrCtx, Diagnostics, Scope, bind_params
from tlang.errors import SourceLocation, TlangSyntaxError
from tlang.frontend.function_manager import FunctionList
from tlang.shader_source_line import ShaderSourceLine
from tlang.shader_utils import mask_comments_and_strings


BLOCK_PATTERN = re.compile(r'\[((?:[^\[\]]|\[[^\[\]]*\])*)\]',re.DOTALL) # support for one nested bracket
ATTR_PATTERN = re.compile(
    r'''
    (?P<name>\w+!?)          # attribute name: shader / include / resourceblock / …
    (?:                      # argument list is OPTIONAL -- bare attributes
                              # like `export` or `unroll` have none at all
        \(                       # opening '('
            (?P<args>            # everything up to the matching ')', with unlimited nesting
                (?:              # non-capturing group
                    [^()]++      #   a run of non-paren chars (possessive: no backtracking into it)
                    |            #   …or…
                    \( (?&args) \)  #   a balanced nested parenthetical, referring only to
                                    #   this group (not the whole pattern, unlike (?R))
                )*+              #   possessive: once a branch is taken it is never revisited,
                                  #   which is what makes this linear instead of exponential
            )
        \)                       # closing ')'
    )?
    ''',
    re.VERBOSE | re.DOTALL
)
ALT_ATTR_PATTERN = re.compile(r'#(?P<name>\w+!?)\s*<\s*(?P<args>.*?)\s*>')
# Only matches within a single line.

# The old ATTR_PATTERN used `(?R)` (recurse the whole name+args pattern) inside the args
# body. Since `\w+!?` can consume 1..k chars of any word that `[^()]` could also consume one
# char of at a time, this made every word in the argument list ambiguous between the two
# alternatives -- a word of length k has ~2^(k-1) parses. That's free as long as the overall
# match succeeds (the first parse found wins), but the moment the argument text contains a
# parenthesis `(?R)` can't close (e.g. a `(` in a comment with no partner), the engine has to
# enumerate the whole exponential space before giving up, hanging the preprocessor.
#
# `(?&args)` instead recurses only the self-contained balanced-parens group, and both
# alternatives are possessive (`++` / `*+`), so a character or a balanced parenthetical, once
# consumed, is never reconsidered. This makes matching (and failing) linear in input length.

# A generous ceiling on how long attribute-regex matching may run. Even with the pattern
# above being linear, this is the backstop that turns "some other pathological input we
# didn't think of hangs the preprocessor forever" into "the preprocessor raises a clear
# error" -- matching real attribute text takes well under a millisecond, so seconds of
# budget is never spent in practice.
ATTR_MATCH_TIMEOUT_SECONDS = 5.0


ARG_PATTERN = re.compile(r"""
    \s*
    (?:@?(?P<key>\w+)\s*=\s*)?
    (?P<value>
        '[^']*'
        |"[^"]*"
        |[^,]+
    )
    \s*(?:,|$)
""", re.VERBOSE)


# class to extract anything in the form []
class AttributeManager:
    @classmethod
    def match_attr(cls, attr_str: str, location: SourceLocation | None = None) -> Attribute | None:
        text = attr_str.strip()
        try:
            m = ATTR_PATTERN.fullmatch(text, timeout=ATTR_MATCH_TIMEOUT_SECONDS)
        except TimeoutError:
            name = nm.group(0) if (nm := re.match(r'\w+!?', text)) else text
            raise TlangSyntaxError(
                f"Attribute '{name}' took longer than {ATTR_MATCH_TIMEOUT_SECONDS:g}s to parse "
                f"-- aborting instead of hanging. This usually means unbalanced parentheses "
                f"somewhere in its argument list.", location,
            )
        if m:
            # a bare attribute (no parens at all) has raw_args='', args=[], kwargs={}
            raw_args = m.group('args') or ''
            return Attribute(m.group('name'), raw_args, *cls.parse_args(raw_args))
        return None

    @staticmethod
    def parse_args(arg_str: str) -> tuple[list[str], dict[str, str]]:
        args: list[str] = []
        kwargs: dict[str, str] = {}
        # Comments are masked to spaces (not stripped from arg_str itself -- callers that
        # want the raw text, e.g. resourceblock, read attr.raw_args directly) so a comment's
        # commas can't mis-split the argument list. String literals are left untouched so
        # quoted commas/parens keep parsing exactly as before.
        masked = mask_comments_and_strings(arg_str, mask_strings=False)
        for m in ARG_PATTERN.finditer(masked):
            val = m.group("value").strip()
            if len(val) >= 2 and val[0] == val[-1] and val[0] in "'\"":
                val = val[1:-1].strip()

            if len(val):
                key = m.group("key")
                if key: kwargs[key] = val
                else: args.append(val)
        return args, kwargs

    # Splits "[shader('compute'), numthreads(64, 1, 1)]" into individual Attributes.
    @classmethod
    def split_attr_block(
        cls, block_str: str, location: SourceLocation | None = None,
        diagnostics: Diagnostics | None = None,
    ) -> list[Attribute]:
        buffer: list[str] = []
        attrs: list[Attribute] = []
        stack_depth: int = 0

        # Paren depth and top-level commas are decided on a masked copy, so a comment inside
        # one attribute's argument list can't desync the split (a stray '(' or a ',' in a
        # `//` or `/* */` comment no longer fools the depth counter). The actual characters
        # -- comment text included -- still come from block_str itself via `buffer`.
        masked = mask_comments_and_strings(block_str)

        # Without a collector (direct calls, tests) this raises; with one it honours `strict`.
        def fail(message: str) -> None:
            if diagnostics is None: raise TlangSyntaxError(message, location)
            diagnostics.fail(message, location, TlangSyntaxError)

        def flush_buffer():
            text = ''.join(buffer).strip()
            buffer.clear()
            if not text: return
            if (attr := cls.match_attr(text, location)) is None:
                fail(f"Malformed attribute '{text}' in [{block_str.strip()}]. "
                     f"Expected 'name' or 'name(args)'.")
                return
            attrs.append(attr)

        for c, mc in zip(block_str, masked):
            if mc == '(': stack_depth += 1
            elif mc == ')': stack_depth -= 1

            if mc == ',' and not stack_depth: flush_buffer()
            else: buffer.append(c)

        if stack_depth != 0:
            fail(f"Unbalanced parentheses in attribute block [{block_str.strip()}]"
                 f" ({'missing' if stack_depth > 0 else 'unexpected'} ')')")
            return attrs
        flush_buffer()
        return attrs

    @classmethod
    def process_attrs(
        cls, shader_name: str, src_map: dict[int, ShaderSourceLine], funcs: FunctionList, strict: bool = True,
    ) -> list[Attribute]:
        diagnostics = Diagnostics(strict=strict)
        cls._attach_func_ctx_attrs(shader_name, funcs, diagnostics)
        return cls._attach_glob_ctx_attrs(shader_name, src_map, funcs, diagnostics)

    # ----- dispatch -----
    # Looks the attribute up in attribute_handlers.REGISTRY and either runs it immediately, or
    # (if deferred) queues it on the target function's `.attrs` for ShaderProcessor to resolve
    # once that function's stage/StageConfig are known.

    @classmethod
    def _dispatch_global(
        cls, attr: Attribute, shader_name: str, funcs: FunctionList, index: int,
        glob_attachments: list[Attribute], diagnostics: Diagnostics,
        src_map: dict[int, ShaderSourceLine] | None = None, end_index: int | None = None,
    ) -> str:
        if (rows := REGISTRY.resolve_scope(attr.name, Scope.GLOBAL, attr.location, diagnostics)) is None:
            return ''  # unknown name / wrong scope -- diagnosed already (dropped only when strict=False)
        spec = rows[0]  # every row sharing a name agrees on `deferred` (enforced at registry construction)

        if spec.deferred:
            # Resolved later, in ShaderProcessor, once this function's stage is known.
            if fn := funcs.find_next(index): fn.attrs.append(attr)
            return f"//<<ATTR '{attr.name}'>>//\n"

        assert spec.handler is not None, f"attribute '{attr.name}' is non-deferred but has no handler"
        bound = bind_params(spec.params, attr, REGISTRY.canonical, spec.variadic)
        ctx = AttrCtx(
            shader_name=shader_name, diagnostics=diagnostics, attr=attr,
            funcs=funcs, index=index, glob_attachments=glob_attachments,
            src_map=src_map, end_index=end_index,
        )
        spec.handler(ctx, bound)
        return f"//<<ATTR '{attr.name}'>>//\n"

    @classmethod
    def _dispatch_funcbody(cls, attr: Attribute, shader_name: str, diagnostics: Diagnostics) -> str:
        if (rows := REGISTRY.resolve_scope(attr.name, Scope.FUNCBODY, attr.location, diagnostics)) is None:
            return ''
        spec = rows[0]
        if spec.literal is not None: return spec.literal
        if spec.handler is not None:
            bound = bind_params(spec.params, attr, REGISTRY.canonical, spec.variadic)
            spec.handler(AttrCtx(shader_name=shader_name, diagnostics=diagnostics, attr=attr), bound)
            return f"//<<ATTR '{attr.name}'>>//\n"
        return ''

    @classmethod
    def _attach_func_ctx_attrs(cls, shader_name: str, funcs: FunctionList, diagnostics: Diagnostics):
        for func in funcs.items:
            line_indices, i = sorted(func.line_body), 0
            while i < len(line_indices):
                idx_out, repl = cls._process_attr_line(
                    shader_name, line_indices[i], func.line_body, Scope.FUNCBODY, diagnostics
                )
                func.line_body[idx_out].data = repl
                i = bisect.bisect_right(line_indices, idx_out)

    @classmethod
    def _attach_glob_ctx_attrs(
        cls, shader_name: str, src_map: dict[int, ShaderSourceLine], funcs: FunctionList, diagnostics: Diagnostics,
    ) -> list[Attribute]:
        glob_attachments: list[Attribute] = []
        line_indices, i = sorted(src_map), 0
        while i < len(line_indices):
            idx_out, repl = cls._process_attr_line(
                shader_name, line_indices[i], src_map, Scope.GLOBAL, diagnostics,
                funcs=funcs, glob_attachments=glob_attachments
            )
            src_map[idx_out].data = repl
            i = bisect.bisect_right(line_indices, idx_out)
        return glob_attachments

    @classmethod
    def _process_attr_line(
        cls,
        shader_name: str,
        index: int,
        map: dict[int, ShaderSourceLine],
        scope: Scope,
        diagnostics: Diagnostics,
        **kwargs
    ) -> tuple[int, str]:
        # Dispatch on the first non-whitespace char: '[attr(...)]' blocks and '#name<args>'
        # directives are unrelated syntaxes and must not both hit the bracket scanner.
        init = map[index].data
        stripped = init.lstrip()
        if stripped.startswith('['):
            return cls._process_block_attr_line(shader_name, index, map, scope, diagnostics, **kwargs)
        if stripped.startswith('#'):
            return cls._process_alt_attr_line(shader_name, index, map, scope, diagnostics, init, **kwargs)
        return index, init

    @classmethod
    def _process_block_attr_line(
        cls,
        shader_name: str,
        index: int,
        map: dict[int, ShaderSourceLine],
        scope: Scope,
        diagnostics: Diagnostics,
        **kwargs
    ) -> tuple[int, str]:
        # Accumulate the (possibly multi-line) '[...]' block without mutating `map` yet, so a
        # raised error leaves the source map untouched. Bracket depth is measured on a masked
        # copy of each line so a stray '['/']' in a comment or string can't desync the span.
        start_index = index
        line_str, depth = '', 0
        while True:
            line_str += (line := map[index].data)
            masked = mask_comments_and_strings(line)
            depth += masked.count('[') - masked.count(']')
            if depth <= 0 or (index + 1) not in map: break
            index += 1
        end_index = index

        def handle_attr(attr: Attribute) -> str:
            if scope is Scope.GLOBAL:
                return cls._dispatch_global(
                    attr, shader_name, kwargs['funcs'], start_index, kwargs['glob_attachments'],
                    diagnostics, src_map=map, end_index=end_index,
                )
            return cls._dispatch_funcbody(attr, shader_name, diagnostics)

        out_line, last_match = '', 0
        for match in BLOCK_PATTERN.finditer(line_str):
            for attr in cls.split_attr_block(match.group(1).strip(), SourceLocation(shader_name, start_index), diagnostics):
                attr.location = SourceLocation(shader_name, start_index)
                out_line += handle_attr(attr)
            last_match = match.end()

        # Keep any trailing text after the last attribute (e.g. a comment), including the '\n'.
        out_line += line_str[last_match:]

        # Blank the interior lines only now; the caller writes the accumulated final line itself.
        for i in range(start_index, end_index): map[i].data = '\n'
        return end_index, out_line

    @classmethod
    def _process_alt_attr_line(
        cls,
        shader_name: str,
        index: int,
        map: dict[int, ShaderSourceLine],
        scope: Scope,
        diagnostics: Diagnostics,
        line: str,
        **kwargs
    ) -> tuple[int, str]:
        # If ALT_ATTR_PATTERN doesn't match, this is an ordinary preprocessor directive
        # (#define/#version/#extension/...) and must be left completely untouched.
        if not (match := ALT_ATTR_PATTERN.search(line)):
            return index, line

        raw_args = match.group('args') or ''
        attr = Attribute(match.group('name'), raw_args, *cls.parse_args(raw_args), location=SourceLocation(shader_name, index))
        if scope is Scope.GLOBAL:
            replacement = cls._dispatch_global(
                attr, shader_name, kwargs['funcs'], index, kwargs['glob_attachments'],
                diagnostics, src_map=map, end_index=index,
            )
        else:
            replacement = cls._dispatch_funcbody(attr, shader_name, diagnostics)
        return index, replacement + line[match.end():]
