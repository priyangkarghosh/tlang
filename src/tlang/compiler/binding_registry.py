# -------------------------------------------------------------
# @file          binding_registry.py
# @author        Priyangkar Ghosh
# @created       2025-07-13
# @description   Scans generated GLSL for buffer/uniform block declarations and assigns
#                bindings for any not pinned explicitly.
# @license       MIT
# -------------------------------------------------------------

import logging
from collections import defaultdict
from moderngl import ComputeShader, Context, Program, StorageBlock, UniformBlock
import regex as re

from tlang.errors import SourceLocation, TlangBindingError
from tlang.shader_stages import ShaderStage

logger = logging.getLogger(__name__)

# Caps how many distinct storage blocks one stage may declare. This, not the
# much larger binding-index ceiling, is what gates linking.
STAGE_LIMIT_KEY: dict[ShaderStage, str] = {
    ShaderStage.COMP: 'GL_MAX_COMPUTE_SHADER_STORAGE_BLOCKS',
    ShaderStage.VERT: 'GL_MAX_VERTEX_SHADER_STORAGE_BLOCKS',
    ShaderStage.FRAG: 'GL_MAX_FRAGMENT_SHADER_STORAGE_BLOCKS',
    ShaderStage.GEOM: 'GL_MAX_GEOMETRY_SHADER_STORAGE_BLOCKS',
    ShaderStage.TESC: 'GL_MAX_TESS_CONTROL_SHADER_STORAGE_BLOCKS',
    ShaderStage.TESE: 'GL_MAX_TESS_EVALUATION_SHADER_STORAGE_BLOCKS',
}

# Same, for uniform blocks -- a separate GL pool, typically ~14 per stage.
UNIFORM_STAGE_LIMIT_KEY: dict[ShaderStage, str] = {
    ShaderStage.COMP: 'GL_MAX_COMPUTE_UNIFORM_BLOCKS',
    ShaderStage.VERT: 'GL_MAX_VERTEX_UNIFORM_BLOCKS',
    ShaderStage.FRAG: 'GL_MAX_FRAGMENT_UNIFORM_BLOCKS',
    ShaderStage.GEOM: 'GL_MAX_GEOMETRY_UNIFORM_BLOCKS',
    ShaderStage.TESC: 'GL_MAX_TESS_CONTROL_UNIFORM_BLOCKS',
    ShaderStage.TESE: 'GL_MAX_TESS_EVALUATION_UNIFORM_BLOCKS',
}

# Per-stage limit map by block keyword, so `_stage_limit` serves both pools.
STAGE_LIMIT_KEYS: dict[str, dict[ShaderStage, str]] = {
    'buffer': STAGE_LIMIT_KEY,
    'uniform': UNIFORM_STAGE_LIMIT_KEY,
}

# Context-wide binding-index ceiling per pool; the two are distinct namespaces.
MAX_POOL_KEY: dict[str, str] = {
    'buffer': 'GL_MAX_SHADER_STORAGE_BUFFER_BINDINGS',
    'uniform': 'GL_MAX_UNIFORM_BUFFER_BINDINGS',
}

# Short noun used in diagnostics for each pool, e.g. "SSBO block" / "uniform block".
BLOCK_NOUN: dict[str, str] = {'buffer': 'SSBO', 'uniform': 'uniform'}

# Conservative guess used only when the driver reports no per-stage limit.
# `_stage_limit` logs whenever it applies, so it is never silent.
FALLBACK_STAGE_LIMIT = 8

# Fallback binding-index ceiling for either pool. Pins beyond it are rejected.
FALLBACK_MAX_BINDINGS = 8

# Memory qualifiers allowed between `layout(...)` and the block keyword. Each
# consumes its own trailing whitespace so consecutive qualifiers stay separated.
BUFFER_QUALIFIER = r"(?:readonly|writeonly|coherent|volatile|restrict)"


def _block_layout_pattern(keyword: str) -> re.Pattern:
    """Layout-block regex for `keyword`, capturing (layout args, qualifiers, name).

    Callers read `match.end() - 1` as the opening '{', so the pattern must keep
    ending in that literal brace.
    """
    return re.compile(
        rf"layout\s*\(\s*([^)]*?)\s*\)\s*((?:{BUFFER_QUALIFIER}\s+)*){keyword}\s+(\w+)\s*{{",
        re.MULTILINE,
    )

BLOCK_PATTERN: dict[str, re.Pattern] = {
    'buffer': _block_layout_pattern('buffer'),
    'uniform': _block_layout_pattern('uniform'),
}

BINDING_PATTERN = re.compile(r"\bbinding\s*=\s*(\d+)\b")

# Optional instance name (and array suffix) between a block's '}' and its ';'.
INSTANCE_TAIL_PATTERN = re.compile(r"\s*(\w+)?(?:\s*\[[^\]]*\])?\s*;")

# Qualifier words that prefix a field declarator without naming it.
FIELD_QUALIFIER_WORDS = frozenset({
    "highp", "mediump", "lowp",
    "readonly", "writeonly", "coherent", "volatile", "restrict", "const",
})

# Comments, strings and preprocessor lines, blanked before scanning so none of
# them count as a use. Blanked character-for-character to preserve offsets.
MASK_PATTERN = re.compile(
    r'//[^\n]*|/\*.*?\*/|"(?:\\.|[^"\\])*"|^[ \t]*#[^\n]*',
    re.DOTALL | re.MULTILINE,
)

class BindingRegistry():
    @staticmethod
    def compute_usage(modules: dict[str, str]) -> dict[str, int]:
        """How many distinct modules declare each SSBO block name, project-wide.

        Computed once, pre-DCE, so a block gets the same preferred binding in
        every artifact that uses it.
        """
        usage: dict[str, int] = defaultdict(int)
        for name, src in modules.items():
            seen: set[str] = set()
            masked = BindingRegistry._mask(src)
            for _layout_args, _qualifiers, block in BLOCK_PATTERN['buffer'].findall(masked):
                if block not in seen:
                    usage[block] += 1
                    seen.add(block)
        return dict(usage)

    @staticmethod
    def preference_rank(usage: dict[str, int]) -> dict[str, int]:
        """Block name -> rank, most cross-module-popular first, ties broken
        lexicographically. Every artifact consults this same map, which is what
        keeps a shared block on the same binding across artifacts.
        """
        ordered = sorted(usage, key=lambda b: (-usage[b], b))
        return {b: i for i, b in enumerate(ordered)}

    @staticmethod
    def _stage_limit(ctx: Context, stage: ShaderStage, keyword: str = 'buffer') -> tuple[int, str]:
        """The max number of distinct `keyword` blocks `stage` may declare at once."""
        key = STAGE_LIMIT_KEYS[keyword][stage]
        if (val := ctx.info.get(key)) is not None:
            return int(val), key
        logger.warning(
            "%s not reported by this driver/context -- falling back to %d for "
            "stage '%s'. This is a conservative guess, not a verified hardware "
            "limit; if builds mysteriously fail to link, check the real value.",
            key, FALLBACK_STAGE_LIMIT, stage.value,
        )
        return FALLBACK_STAGE_LIMIT, f"{key} (not reported, using fallback)"

    @staticmethod
    def _max_pool(ctx: Context, keyword: str) -> tuple[int, str]:
        """Binding-index ceiling for `keyword`'s pool."""
        key = MAX_POOL_KEY[keyword]
        if (val := ctx.info.get(key)) is not None:
            return int(val), key
        logger.warning(
            "%s not reported by this driver/context -- falling back to %d.",
            key, FALLBACK_MAX_BINDINGS,
        )
        return FALLBACK_MAX_BINDINGS, f"{key} (not reported, using fallback)"

    @staticmethod
    def _allocate_canon(
        ctx: Context, artifact: str, stage_sources: dict[ShaderStage, str],
        pref_rank: dict[str, int], keyword: str,
    ) -> dict[str, int]:
        """Assign bindings for every live `keyword` block across `stage_sources`, from one pool.

        Explicit `layout(binding = N)` pins are honoured first and never moved; the rest are
        auto-assigned in `pref_rank` order to the lowest free index. Raises `TlangBindingError`
        for the same conditions `allocate_artifact` documents, scoped to this one pool.
        """
        pattern = BLOCK_PATTERN[keyword]
        noun = BLOCK_NOUN[keyword]
        max_pool, pool_source = BindingRegistry._max_pool(ctx, keyword)

        # per-stage live block lists (for the budget check below) + explicit pins
        per_stage_blocks: dict[ShaderStage, set[str]] = {}
        explicit: dict[str, int] = {}
        for stage, src in stage_sources.items():
            masked = BindingRegistry._mask(src)
            names: set[str] = set()
            for layout_args, _qualifiers, block in pattern.findall(masked):
                names.add(block)
                if not (m := BINDING_PATTERN.search(layout_args)): continue

                bind = int(m.group(1))
                if bind >= max_pool:
                    raise TlangBindingError(
                        f"Artifact '{artifact}': binding {bind} on {noun} block '{block}' exceeds "
                        f"the driver's binding-index ceiling ({max_pool}, from {pool_source})",
                        SourceLocation(artifact),
                    )

                prev = explicit.setdefault(block, bind)
                if prev != bind:
                    raise TlangBindingError(
                        f"Artifact '{artifact}': {noun} block '{block}' has conflicting explicit "
                        f"bindings ({prev} vs {bind}) across its stages",
                        SourceLocation(artifact),
                    )
            per_stage_blocks[stage] = names

        # Two different blocks pinned to the same index would silently alias onto one GL
        # binding point, so refuse it outright rather than ship it.
        owner_of: dict[int, str] = {}
        for block, bind in explicit.items():
            if (owner := owner_of.get(bind)) is not None and owner != block:
                raise TlangBindingError(
                    f"Artifact '{artifact}': {noun} blocks '{owner}' and '{block}' are both "
                    f"explicitly bound to {bind}",
                    SourceLocation(artifact),
                )
            owner_of[bind] = block

        # Auto-assign every live, unpinned block in `pref_rank` order to the lowest free index.
        live = {b for names in per_stage_blocks.values() for b in names}
        canon: dict[str, int] = dict(explicit)
        reserved = set(explicit.values())
        remaining = sorted((b for b in live if b not in canon), key=lambda b: (pref_rank.get(b, len(pref_rank)), b))

        slot = 0
        for block in remaining:
            while slot in reserved: slot += 1
            if slot >= max_pool:
                raise TlangBindingError(
                    f"Artifact '{artifact}': out of {noun} bindings while assigning '{block}' "
                    f"(binding-index ceiling {max_pool}, from {pool_source})",
                    SourceLocation(artifact),
                )
            canon[block] = slot
            reserved.add(slot)
            slot += 1

        # The limit that actually gates linking is the number of distinct blocks live in one
        # stage, not the (much larger) binding-index ceiling checked above.
        for stage, names in per_stage_blocks.items():
            if not names: continue
            limit, limit_source = BindingRegistry._stage_limit(ctx, stage, keyword)
            if len(names) > limit:
                raise TlangBindingError(
                    f"Artifact '{artifact}': {stage.value} stage references {len(names)} {noun} "
                    f"blocks {sorted(names)} but the driver allows only {limit} ({limit_source})",
                    SourceLocation(artifact),
                )

        return canon

    @staticmethod
    def _patch_bindings(src: str, keyword: str, canon: dict[str, int]) -> str:
        """Inject `binding = N` into every `keyword` declarator in `src`
        that doesn't already pin one explicitly, using `canon`."""
        pattern = BLOCK_PATTERN[keyword]

        def replacer(match: re.Match) -> str:
            layout_args, qualifier, block_name = match.group(1), match.group(2), match.group(3)
            # Allocation scans masked source, this scans raw, so a declaration inside a
            # comment reaches here with no binding assigned. Leave it exactly as written.
            if block_name not in canon: return match.group(0)
            if BINDING_PATTERN.search(layout_args): return match.group(0)
            new_args = f"binding = {canon[block_name]}, {layout_args}".strip().strip(',')
            return f"layout({new_args}) {qualifier}{keyword} {block_name} {{"

        return pattern.sub(replacer, src)

    @staticmethod
    def allocate_artifact(
        ctx: Context, artifact: str, stage_sources: dict[ShaderStage, str], pref_rank: dict[str, int],
    ) -> tuple[dict[ShaderStage, str], dict[str, int], dict[str, int]]:
        """Assign bindings for one artifact: a compute entry point, or every stage of one
        `[program(...)]` together, so a shared block keeps one binding.

        `stage_sources` must already be DCE'd. SSBO and uniform blocks are separate GL
        namespaces, so each gets its own pool; returns `(patched_stage_sources, ssbo_canon,
        uniform_canon)` since `Kernel.bind_ssbo` consumes only the SSBO one.

        Raises `TlangBindingError` on a pin conflict, a pin past the index ceiling, a stage
        over its per-kind block limit, or an exhausted pool.
        """
        ssbo_canon = BindingRegistry._allocate_canon(ctx, artifact, stage_sources, pref_rank, 'buffer')
        uniform_canon = BindingRegistry._allocate_canon(ctx, artifact, stage_sources, pref_rank, 'uniform')

        patched: dict[ShaderStage, str] = {}
        for stage, src in stage_sources.items():
            src = BindingRegistry._patch_bindings(src, 'buffer', ssbo_canon)
            src = BindingRegistry._patch_bindings(src, 'uniform', uniform_canon)
            patched[stage] = src

        return patched, ssbo_canon, uniform_canon

    @staticmethod
    def verify_link(
        linked: ComputeShader | Program, canon: dict[str, int], artifact: str,
        uniform_canon: dict[str, int] | None = None,
    ) -> None:
        """Reflect the linked program's blocks and assert they match the canons.

        GL silently accepts two blocks aliased onto one binding and just returns
        wrong data, so this turns any gap in the textual scan into a build failure.
        Blocks the driver eliminated are skipped.
        """
        for kind_canon, block_type in ((canon, StorageBlock), (uniform_canon or {}, UniformBlock)):
            for name, expected in kind_canon.items():
                block = linked.get(name, None)
                if not isinstance(block, block_type): continue
                if (actual := block.binding) != expected:
                    raise TlangBindingError(
                        f"Binding mismatch for block '{name}' in artifact '{artifact}': "
                        f"tlang assigned {expected} but the linked program reports {actual}. "
                        f"This should be impossible -- treat it as a tlang bug.",
                        SourceLocation(artifact),
                    )

    @staticmethod
    def _mask(src: str) -> str:
        """Blank comments, string literals, and preprocessor lines in `src`.

        Same length, same newline positions as `src` -- offsets computed
        against the result stay valid against the original.
        """
        blank = lambda m: ''.join(c if c == '\n' else ' ' for c in m.group(0))
        return MASK_PATTERN.sub(blank, src)

    @staticmethod
    def _match_brace(text: str, open_pos: int) -> int | None:
        """Index of the '}' matching the '{' at `open_pos` in `text`, or None if unbalanced."""
        depth = 0
        for i in range(open_pos, len(text)):
            if text[i] == '{': depth += 1
            elif text[i] == '}':
                depth -= 1
                if depth == 0: return i
        return None

    @staticmethod
    def _split_top_level(body: str) -> list[str]:
        """Split a block body into field statements on top-level ';' only.

        A nested struct's own members (inside its `{ }`) sit at brace
        depth > 0, so e.g. `struct S { vec3 a; float b; } s[];` comes back
        as one statement, not three -- the inner ';'s don't split it.
        """
        stmts, depth, start = [], 0, 0
        for i, c in enumerate(body):
            if c == '{': depth += 1
            elif c == '}': depth -= 1
            elif c == ';' and depth == 0:
                stmts.append(body[start:i])
                start = i + 1
        return stmts

    @staticmethod
    def _declarator_names(decl_list: str) -> list[str]:
        """Trailing identifier of each comma-separated declarator in `decl_list`.

        Strips array suffixes, initializers, and leading qualifier/precision words -- so
        `highp uint counts[]` yields `counts`, and `uint a, b[4], c` yields `a`, `b`, `c`.
        """
        names = []
        for segment in decl_list.split(','):
            segment = re.sub(r'\[[^\]]*\]', '', segment).split('=')[0]
            words = [w for w in segment.split() if w not in FIELD_QUALIFIER_WORDS]
            if (ids := re.findall(r'[A-Za-z_]\w*', ' '.join(words))):
                names.append(ids[-1])
        return names

    @staticmethod
    def _statement_field_names(stmt: str) -> list[str]:
        """Field identifier(s) declared by one top-level block statement.

        A plain statement is just a declarator list. A nested-struct statement
        (`struct S { ... } s[]`) skips the type body -- only the declarator list after its
        closing brace names an actual block field.
        """
        stmt = stmt.strip()
        if not stmt: return []
        if (brace := stmt.find('{')) == -1:
            return BindingRegistry._declarator_names(stmt)
        close = BindingRegistry._match_brace(stmt, brace)
        return BindingRegistry._declarator_names(stmt[close + 1:]) if close is not None else []

    @staticmethod
    def remove_dead_blocks(src: str, keywords: tuple[str, ...] = ('buffer', 'uniform')) -> str:
        """Strip blocks of each kind in `keywords` whose fields are never referenced elsewhere in `src`.

        A best-effort identifier scan with no notion of scope, biased toward keeping when
        uncertain (a wrongly-kept block costs a binding slot; a wrongly-removed one is a
        compile error). An instance-named block is searched for by its instance name. Must
        run before `allocate_artifact` so bindings are only spent on survivors.
        """
        masked = BindingRegistry._mask(src)
        removable: list[tuple[int, int]] = []

        for keyword in keywords:
            pattern = BLOCK_PATTERN[keyword]
            for lm in pattern.finditer(masked):
                block_name = lm.group(3)
                open_brace = lm.end() - 1

                if (close_brace := BindingRegistry._match_brace(masked, open_brace)) is None:
                    logger.warning("DCE: unbalanced braces in %s block '%s' -- keeping.", keyword, block_name)
                    continue

                if not (tail := INSTANCE_TAIL_PATTERN.match(masked, close_brace + 1)):
                    logger.warning("DCE: %s block '%s' has no terminating ';' -- keeping.", keyword, block_name)
                    continue

                instance_name, block_end = tail.group(1), tail.end()
                body = masked[open_brace + 1:close_brace]
                field_names = [n for stmt in BindingRegistry._split_top_level(body)
                               for n in BindingRegistry._statement_field_names(stmt)]

                # an instance-named block is only ever referenced through the
                # instance (`inst.field`), never the bare field names
                search_names = [instance_name] if instance_name else field_names
                if not search_names:
                    logger.warning("DCE: %s block '%s' has no field/instance name to check -- keeping.", keyword, block_name)
                    continue

                rest = masked[:lm.start()] + masked[block_end:]
                if any(re.search(r'\b' + re.escape(n) + r'\b', rest) for n in search_names):
                    continue  # at least one name is referenced elsewhere -- keep

                removable.append((lm.start(), block_end))

        if not removable: return src
        removable.sort()

        # Replace each removed span with blank lines (not a deletion) so #line-based
        # diagnostics further down the module stay line-accurate.
        out, cursor = [], 0
        for start, end in removable:
            out.append(src[cursor:start])
            out.append('\n' * src.count('\n', start, end))
            cursor = end
        out.append(src[cursor:])
        return ''.join(out)

    @staticmethod
    def remove_unused_buffers(src: str) -> str:
        """Back-compat name for `remove_dead_blocks` -- strips dead `buffer`
        AND `uniform` blocks alike. Kept because `Shader._build` and
        existing tests call it under this name."""
        return BindingRegistry.remove_dead_blocks(src)
