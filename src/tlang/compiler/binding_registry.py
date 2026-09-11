# -------------------------------------------------------------
# @file          binding_registry.py
# @author        Priyangkar Ghosh
# @created       2025-07-13
# @description   Scans generated GLSL for buffer/uniform block declarations and assigns
#                bindings for any not pinned explicitly.
# @license       MIT
# -------------------------------------------------------------

import ctypes
import logging
from collections import defaultdict
from moderngl import ComputeShader, Context, Program, StorageBlock, Uniform, UniformBlock
from OpenGL.GL import (
    GL_ACTIVE_RESOURCES, GL_ATOMIC_COUNTER_BUFFER, GL_BUFFER_BINDING,
    glGetProgramInterfaceiv, glGetProgramResourceiv,
)
import regex as re

from tlang.errors import SourceLocation, TlangBindingError
from tlang.frontend.function_manager import CONTROL_KEYWORDS, FUNC_PATTERN
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

# Same, for atomic counter BUFFERS (not counters themselves -- GL packs many counters into
# one buffer binding, so this is the cap on distinct *bindings* a stage may reference, exactly
# like the SSBO/UBO limits above are caps on distinct *blocks*). GL_MAX_COMPUTE_ATOMIC_COUNTER_
# BUFFERS=8 IS reported on the reference machine (RTX 3090 / GL 4.6 / moderngl 5.12) -- unlike
# GL_MAX_ATOMIC_COUNTER_BUFFER_BINDINGS below, which is not.
ATOMIC_COUNTER_STAGE_LIMIT_KEY: dict[ShaderStage, str] = {
    ShaderStage.COMP: 'GL_MAX_COMPUTE_ATOMIC_COUNTER_BUFFERS',
    ShaderStage.VERT: 'GL_MAX_VERTEX_ATOMIC_COUNTER_BUFFERS',
    ShaderStage.FRAG: 'GL_MAX_FRAGMENT_ATOMIC_COUNTER_BUFFERS',
    ShaderStage.GEOM: 'GL_MAX_GEOMETRY_ATOMIC_COUNTER_BUFFERS',
    ShaderStage.TESC: 'GL_MAX_TESS_CONTROL_ATOMIC_COUNTER_BUFFERS',
    ShaderStage.TESE: 'GL_MAX_TESS_EVALUATION_ATOMIC_COUNTER_BUFFERS',
}

# Per-stage limit map by block keyword, so `_stage_limit` serves all three pools.
STAGE_LIMIT_KEYS: dict[str, dict[ShaderStage, str]] = {
    'buffer': STAGE_LIMIT_KEY,
    'uniform': UNIFORM_STAGE_LIMIT_KEY,
    'counter': ATOMIC_COUNTER_STAGE_LIMIT_KEY,
}

# Context-wide binding-index ceiling per pool; the two are distinct namespaces.
MAX_POOL_KEY: dict[str, str] = {
    'buffer': 'GL_MAX_SHADER_STORAGE_BUFFER_BINDINGS',
    'uniform': 'GL_MAX_UNIFORM_BUFFER_BINDINGS',
    # texture and image units are their own separate pools too (verified on an RTX 3090 / GL
    # 4.6 / moderngl 5.12: GL_MAX_TEXTURE_IMAGE_UNITS=32; GL_MAX_IMAGE_UNITS is not reported by
    # `ctx.info` at all on that driver, so it always takes the fallback path below).
    'texture': 'GL_MAX_TEXTURE_IMAGE_UNITS',
    'image': 'GL_MAX_IMAGE_UNITS',
    # atomic counter BUFFER binding-index ceiling -- NOT reported by `ctx.info` at all on the
    # reference machine (RTX 3090 / GL 4.6 / moderngl 5.12), so this pool always takes the
    # `_max_pool` fallback path below, exactly like 'image' does there.
    'counter': 'GL_MAX_ATOMIC_COUNTER_BUFFER_BINDINGS',
}

# Short noun used in diagnostics for each pool, e.g. "SSBO block" / "uniform block".
BLOCK_NOUN: dict[str, str] = {
    'buffer': 'SSBO', 'uniform': 'uniform', 'texture': 'sampler', 'image': 'image',
    'counter': 'atomic counter',
}

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

# -----------------------------------------------------------------
# sampler / image uniform discovery -- same "scan, not new syntax" deal as the block pattern
# above: raw GLSL (`uniform sampler2D tex;`, `layout(rgba8, binding = N) uniform image2D img;`)
# keeps working exactly as today. Unlike buffer/uniform blocks, GL reflects both sampler and
# image uniforms as a plain `Uniform` with a writable `.value` that IS the unit index, so this
# pool is never patched into the GLSL text -- see `assign_opaque_units`, which assigns purely
# through post-link reflection instead of `_patch_bindings`.
# -----------------------------------------------------------------

_OPAQUE_DIMS = ('1D', '2D', '3D', 'Cube', '1DArray', '2DArray', 'CubeArray', '2DMS', '2DMSArray', 'Buffer', '2DRect')
_SHADOW_DIMS = ('1D', '2D', 'Cube', '1DArray', '2DArray', 'CubeArray', '2DRect')


def _type_alternation(*names: str) -> str:
    # Longest-first is only a minor efficiency nicety here, not a correctness requirement: the
    # pattern below demands a `\s+` boundary right after the type name, so e.g. "sampler2D"
    # can never falsely consume the front of "sampler2DArray" and stop there.
    return '|'.join(sorted(set(names), key=len, reverse=True))


SAMPLER_TYPE_ALT = _type_alternation(
    *(f'{prefix}sampler{dim}' for prefix in ('', 'i', 'u') for dim in _OPAQUE_DIMS),
    *(f'sampler{dim}Shadow' for dim in _SHADOW_DIMS),
)
IMAGE_TYPE_ALT = _type_alternation(
    *(f'{prefix}image{dim}' for prefix in ('', 'i', 'u') for dim in _OPAQUE_DIMS),
)

# kind -> alternation of every GLSL opaque type name that pool covers.
OPAQUE_TYPE_ALT: dict[str, str] = {'texture': SAMPLER_TYPE_ALT, 'image': IMAGE_TYPE_ALT}


def _opaque_uniform_pattern(type_alt: str) -> re.Pattern:
    """Declaration regex for one opaque-uniform pool, capturing (layout args, type, name).

    `layout(...)` is optional (most declarations have none -- every sampler/image defaults to
    unit 0 until this module assigns one). Memory qualifiers (`readonly`, `coherent`, ...) may
    appear on either side of `uniform` -- GLSL accepts both orderings for images -- so both are
    matched, harmlessly permissive for samplers (which never carry one). No block braces here,
    unlike `_block_layout_pattern`: these are plain statement declarations.
    """
    return re.compile(
        rf"\buniform\s+(?:{BUFFER_QUALIFIER}\s+)*({type_alt})\s+(\w+)\s*(?:\[[^\]]*\])?\s*;",
        re.MULTILINE,
    )


# An optional `layout(...)` (plus any memory qualifiers) sitting immediately before a
# declaration. Applied only at the few offsets `OPAQUE_PATTERN` already matched, never
# scanned across the whole unit: as an unanchored prefix it made the engine retry a
# nondeterministic optional group at every character, costing 54 ms per findall.
OPAQUE_PREFIX_PATTERN = re.compile(
    rf"layout\s*\(\s*([^)]*)\s*\)\s*(?:{BUFFER_QUALIFIER}\s+)*$"
)


OPAQUE_PATTERN: dict[str, re.Pattern] = {
    kind: _opaque_uniform_pattern(type_alt) for kind, type_alt in OPAQUE_TYPE_ALT.items()
}

# -----------------------------------------------------------------
# atomic_uint discovery -- same "scan, not new syntax" deal, but a THIRD shape from either pool
# above. Two counters can share one binding at different byte offsets
# (`layout(binding=0, offset=0) uniform atomic_uint a;` / `(binding=0, offset=4) ... b;`), so
# allocation here is 2-D (binding, offset), not 1-D. And unlike samplers/images, an atomic
# counter is COMPLETELY INVISIBLE to moderngl reflection (`program.get('name')` is always
# `None`, verified against a live GL 4.6 context) -- there is no `.value` to assign post-link,
# so an unpinned declaration's binding/offset MUST be patched into the GLSL text, the same way
# `_patch_bindings` does for SSBO/UBO blocks. Anchored on the required `\buniform` literal
# exactly like `OPAQUE_PATTERN`, for the same reason: an unanchored optional `layout(...)`
# prefix folded into the main scan made a 13-module build go from 0.5s to 32.5s.
# -----------------------------------------------------------------

ATOMIC_COUNTER_PATTERN = re.compile(
    r"\buniform\s+atomic_uint\s+(\w+)\s*(?:\[[^\]]*\])?\s*;",
    re.MULTILINE,
)

# `offset = M` inside a matched `layout(...)` argument string -- the atomic-counter-specific
# counterpart of `BINDING_PATTERN`.
OFFSET_PATTERN = re.compile(r"\boffset\s*=\s*(\d+)\b")

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

# An identifier immediately followed by '(' -- a call site, a function header, OR a builtin/
# type-constructor invocation (`vec4(...)`, `atomicAdd(...)`). Builtins are never keys in the
# function map this module builds, so they fall out of consideration on their own -- nothing
# here special-cases them.
CALL_SITE_PATTERN = re.compile(r'\b([A-Za-z_]\w*)\s*\(')

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
    def _allocate_opaque_canon(
        ctx: Context, artifact: str, stage_sources: dict[ShaderStage, str], kind: str,
    ) -> dict[str, int]:
        """Assign units for every live sampler/image uniform across `stage_sources`, from one
        pool ('texture' or 'image').

        Mirrors `_allocate_canon`'s pin/conflict/exhaustion policy exactly, minus the per-stage
        count check: GLSL has no per-stage cap on the number of sampler/image *declarations*
        analogous to `GL_MAX_*_SHADER_STORAGE_BLOCKS` -- the driver's own per-stage texture/image
        unit limits gate actual *usage*, not declaration count, and are enforced by the linker
        itself, so there is nothing extra worth checking here.
        """
        pattern = OPAQUE_PATTERN[kind]
        noun = BLOCK_NOUN[kind]
        max_pool, pool_source = BindingRegistry._max_pool(ctx, kind)

        live: set[str] = set()
        explicit: dict[str, int] = {}
        for src in stage_sources.values():
            masked = BindingRegistry._mask(src)
            for match in pattern.finditer(masked):
                name = match.group(2)
                live.add(name)
                prefix = OPAQUE_PREFIX_PATTERN.search(masked, 0, match.start())
                if not prefix or not (m := BINDING_PATTERN.search(prefix.group(1))): continue

                unit = int(m.group(1))
                if unit >= max_pool:
                    raise TlangBindingError(
                        f"Artifact '{artifact}': binding {unit} on {noun} '{name}' exceeds the "
                        f"driver's unit ceiling ({max_pool}, from {pool_source})",
                        SourceLocation(artifact),
                    )

                prev = explicit.setdefault(name, unit)
                if prev != unit:
                    raise TlangBindingError(
                        f"Artifact '{artifact}': {noun} '{name}' has conflicting explicit "
                        f"bindings ({prev} vs {unit}) across its stages",
                        SourceLocation(artifact),
                    )

        # Two different samplers/images pinned to the same unit would silently alias, exactly
        # the hazard this whole feature exists to close -- refuse it outright.
        owner_of: dict[int, str] = {}
        for name, unit in explicit.items():
            if (owner := owner_of.get(unit)) is not None and owner != name:
                raise TlangBindingError(
                    f"Artifact '{artifact}': {noun}s '{owner}' and '{name}' are both explicitly "
                    f"bound to unit {unit}",
                    SourceLocation(artifact),
                )
            owner_of[unit] = name

        # Auto-assign every live, unpinned name to the lowest free unit, deterministically
        # (alphabetically -- there is no cross-module preference ranking for this pool).
        canon: dict[str, int] = dict(explicit)
        reserved = set(explicit.values())
        remaining = sorted(n for n in live if n not in canon)

        slot = 0
        for name in remaining:
            while slot in reserved: slot += 1
            if slot >= max_pool:
                raise TlangBindingError(
                    f"Artifact '{artifact}': out of {noun} units while assigning '{name}' "
                    f"(unit ceiling {max_pool}, from {pool_source})",
                    SourceLocation(artifact),
                )
            canon[name] = slot
            reserved.add(slot)
            slot += 1

        return canon

    @staticmethod
    def allocate_opaque_units(
        ctx: Context, artifact: str, stage_sources: dict[ShaderStage, str],
    ) -> tuple[dict[str, int], dict[str, int]]:
        """Assign texture and image units for every sampler/image uniform declared across
        `stage_sources` -- the sampler/image counterpart of `allocate_artifact`.

        Texture units and image units are two independent GL pools (not merged, exactly like
        the SSBO/UBO split `allocate_artifact` already keeps), so each gets its own call into
        `_allocate_opaque_canon`. Nothing here is patched into the GLSL text -- see
        `assign_opaque_units`, called once the artifact has actually linked. Raises
        `TlangBindingError` on a pin conflict, a pin past the unit ceiling, or an exhausted pool.
        """
        texture_canon = BindingRegistry._allocate_opaque_canon(ctx, artifact, stage_sources, 'texture')
        image_canon = BindingRegistry._allocate_opaque_canon(ctx, artifact, stage_sources, 'image')
        return texture_canon, image_canon

    @staticmethod
    def assign_opaque_units(
        linked: ComputeShader | Program, texture_canon: dict[str, int], image_canon: dict[str, int],
    ) -> None:
        """Push each canon's unit into the linked artifact's reflected uniform, post-link.

        Sampler and image uniforms both reflect as a plain `moderngl.Uniform` (not a distinct
        type the way storage/uniform blocks do) with a writable `.value` that IS the texture/
        image unit -- verified: three separate sampler/image uniforms with no explicit binding
        all reported `value = 0`, i.e. silently colliding. Setting `.value` here is what this
        whole mechanism exists to automate instead.

        GL strips an inactive (declared but never referenced) uniform from reflection entirely.
        Such a name is PRUNED from its canon here, so what survives means "opaque uniforms this
        artifact actually uses" rather than "names the text declared". The distinction is the
        one `remove_dead_functions` draws for blocks: a set that over-approximates makes an
        unbound-at-dispatch check fire on artifacts that never had a bug, which trains people
        to suppress it. The driver's own reflection is authoritative here, so the narrow set
        is free.
        """
        for canon in (texture_canon, image_canon):
            reflected = {n: u for n in list(canon) if isinstance(u := linked.get(n, None), Uniform)}
            for name in [n for n in canon if n not in reflected]: del canon[name]
            for name, unit in canon.items(): reflected[name].value = unit

    @staticmethod
    def _allocate_counter_canon(
        ctx: Context, artifact: str, stage_sources: dict[ShaderStage, str],
    ) -> dict[str, tuple[int, int]]:
        """Assign (binding, offset) for every live `atomic_uint` counter across `stage_sources`.

        Mirrors `_allocate_canon`'s pin/conflict/exhaustion policy, adapted to a 2-D pool: GL is
        designed to pack many counters into one buffer binding at successive 4-byte offsets, so
        unpinned counters of one artifact are packed into a SINGLE binding rather than spending
        one binding per counter (that would exhaust the 8-per-stage counter-buffer budget after
        just 8 counters, for no reason -- GL doesn't require it). Explicit `binding=`/`offset=`
        pins are honoured first and never moved. Raises `TlangBindingError` for a pin conflict
        (two counters at the same (binding, offset)), a pin past the binding-index ceiling, an
        exhausted pool, or a stage referencing more distinct counter-buffer bindings than the
        driver allows.
        """
        noun = BLOCK_NOUN['counter']
        max_pool, pool_source = BindingRegistry._max_pool(ctx, 'counter')

        per_stage_names: dict[ShaderStage, set[str]] = {}
        live: set[str] = set()
        explicit: dict[str, tuple[int, int]] = {}
        for stage, src in stage_sources.items():
            masked = BindingRegistry._mask(src)
            names_here: set[str] = set()
            for match in ATOMIC_COUNTER_PATTERN.finditer(masked):
                name = match.group(1)
                names_here.add(name)
                live.add(name)

                prefix = OPAQUE_PREFIX_PATTERN.search(masked, 0, match.start())
                if not prefix or not (bm := BINDING_PATTERN.search(prefix.group(1))): continue

                binding = int(bm.group(1))
                if binding >= max_pool:
                    raise TlangBindingError(
                        f"Artifact '{artifact}': binding {binding} on {noun} '{name}' exceeds "
                        f"the driver's binding-index ceiling ({max_pool}, from {pool_source})",
                        SourceLocation(artifact),
                    )
                om = OFFSET_PATTERN.search(prefix.group(1))
                offset = int(om.group(1)) if om else 0

                prev = explicit.setdefault(name, (binding, offset))
                if prev != (binding, offset):
                    raise TlangBindingError(
                        f"Artifact '{artifact}': {noun} '{name}' has conflicting explicit "
                        f"bindings ({prev} vs {(binding, offset)}) across its stages",
                        SourceLocation(artifact),
                    )
            per_stage_names[stage] = names_here

        # Two different counters pinned to the same (binding, offset) would silently alias
        # onto one 4-byte slot of GL's atomic counter buffer -- refuse it outright. Sharing a
        # BINDING at different offsets is the idiomatic, intended form and is not checked here.
        owner_of: dict[tuple[int, int], str] = {}
        for name, pos in explicit.items():
            if (owner := owner_of.get(pos)) is not None and owner != name:
                raise TlangBindingError(
                    f"Artifact '{artifact}': {noun}s '{owner}' and '{name}' are both explicitly "
                    f"bound to binding={pos[0]}, offset={pos[1]}",
                    SourceLocation(artifact),
                )
            owner_of[pos] = name

        # Pack every live, unpinned counter into ONE binding (the lowest not already claimed
        # by a pin), at successive 4-byte offsets not already claimed by a pin at that binding.
        canon: dict[str, tuple[int, int]] = dict(explicit)
        reserved_bindings = {b for b, _o in explicit.values()}
        remaining = sorted(n for n in live if n not in canon)

        if remaining:
            slot = 0
            while slot in reserved_bindings: slot += 1
            if slot >= max_pool:
                raise TlangBindingError(
                    f"Artifact '{artifact}': out of {noun} bindings while assigning "
                    f"'{remaining[0]}' (binding-index ceiling {max_pool}, from {pool_source})",
                    SourceLocation(artifact),
                )
            used_offsets = {o for b, o in explicit.values() if b == slot}
            next_offset = 0
            for name in remaining:
                while next_offset in used_offsets: next_offset += 4
                canon[name] = (slot, next_offset)
                used_offsets.add(next_offset)
                next_offset += 4

        # The limit that actually gates linking is the number of distinct counter-buffer
        # BINDINGS live in one stage (analogous to the SSBO/UBO per-stage block-count check),
        # not the number of counter names -- many names can share one binding for free.
        for stage, names in per_stage_names.items():
            if not names: continue
            bindings_used = {canon[n][0] for n in names}
            limit, limit_source = BindingRegistry._stage_limit(ctx, stage, 'counter')
            if len(bindings_used) > limit:
                raise TlangBindingError(
                    f"Artifact '{artifact}': {stage.value} stage references {len(bindings_used)} "
                    f"{noun} buffer binding(s) {sorted(bindings_used)} but the driver allows only "
                    f"{limit} ({limit_source})",
                    SourceLocation(artifact),
                )

        return canon

    @staticmethod
    def allocate_atomic_counters(
        ctx: Context, artifact: str, stage_sources: dict[ShaderStage, str],
    ) -> tuple[dict[ShaderStage, str], dict[str, tuple[int, int]]]:
        """Assign and patch (binding, offset) for every `atomic_uint` counter declared across
        `stage_sources` -- the atomic-counter counterpart of `allocate_artifact`.

        Unlike SSBO/UBO blocks or sampler/image uniforms, an unpinned atomic counter has no
        legal declaration at all on this driver (`atomic counter 'x' declaration requires the
        layout qualifier` -- a real, verified compile error, not a style preference), so every
        live counter this function returns a canon entry for is ALSO patched into the returned
        source text, even ones that already had an explicit pin (patching is a no-op for those
        -- see `_patch_counter_bindings`). Returns `(patched_stage_sources, counter_canon)`.
        """
        canon = BindingRegistry._allocate_counter_canon(ctx, artifact, stage_sources)
        patched = {stage: BindingRegistry._patch_counter_bindings(src, canon) for stage, src in stage_sources.items()}
        return patched, canon

    @staticmethod
    def active_atomic_counter_bindings(linked: ComputeShader | Program) -> set[int]:
        """Every GL binding index actually active on `linked`'s `GL_ATOMIC_COUNTER_BUFFER`
        program-interface resources, queried through raw pyOpenGL post-link.

        Atomic counters are invisible to moderngl's own reflection entirely (verified: a linked
        program with `layout(binding=0) uniform atomic_uint x;`, actually used by `main`,
        reflects no member named 'x' at all), so this raw query is the only way to learn which
        of tlang's textually-discovered counters the driver actually kept active -- exactly the
        role `assign_opaque_units`'s reflection check plays for samplers/images, adapted to a
        driver interface that has no notion of "this uniform's value". The granularity is per
        BINDING (a whole atomic counter buffer), not per counter name -- GL has no active-
        resource concept finer than that, which is also the right granularity here: two counters
        packed into one binding are bound or not bound together regardless of which of them
        `main` happens to touch.
        """
        prog = linked.glo
        count = glGetProgramInterfaceiv(prog, GL_ATOMIC_COUNTER_BUFFER, GL_ACTIVE_RESOURCES)
        props = (ctypes.c_uint * 1)(GL_BUFFER_BINDING)
        bindings: set[int] = set()
        for i in range(count):
            params = (ctypes.c_int * 1)()
            glGetProgramResourceiv(prog, GL_ATOMIC_COUNTER_BUFFER, i, 1, props, 1, None, params)
            bindings.add(int(params[0]))
        return bindings

    @staticmethod
    def _patch_counter_bindings(src: str, canon: dict[str, tuple[int, int]]) -> str:
        """Inject `layout(binding = N, offset = M)` before every `atomic_uint` declarator in
        `src` that doesn't already pin its own binding, using `canon`.

        Unlike `_patch_bindings`, an `atomic_uint` declaration has no `layout(...)` required to
        begin with, so one may need to be manufactured whole cloth rather than merely amended.
        Scans `ATOMIC_COUNTER_PATTERN` once (anchored on the required `uniform` literal, cheap);
        only at each already-matched offset does it check -- via `OPAQUE_PREFIX_PATTERN`, never
        across the whole file -- whether a `layout(...)` already precedes it. This is the same
        two-step shape discovery already uses, for the same performance reason.
        """
        if not canon: return src

        out: list[str] = []
        cursor = 0
        for match in ATOMIC_COUNTER_PATTERN.finditer(src):
            name = match.group(1)
            if name not in canon: continue  # not live per the masked scan -- leave untouched

            prefix = OPAQUE_PREFIX_PATTERN.search(src, 0, match.start())
            if prefix and BINDING_PATTERN.search(prefix.group(1)):
                continue  # already pins its own binding -- leave untouched

            binding, offset = canon[name]
            if prefix:
                out.append(src[cursor:prefix.start()])
                new_args = f"binding = {binding}, offset = {offset}, {prefix.group(1)}".strip().strip(',')
                out.append(f"layout({new_args})")
                cursor = prefix.end()
            else:
                out.append(src[cursor:match.start()])
                out.append(f"layout(binding = {binding}, offset = {offset}) ")
                cursor = match.start()

        out.append(src[cursor:])
        return ''.join(out)

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

    # -----------------------------------------------------------------
    # dead-function elimination -- same textual-approximation class as the
    # block DCE above, sharing `_mask`/`_match_brace`. Must run before
    # `remove_dead_blocks` so the block DCE only sees text `main` can
    # actually reach; see `remove_dead_functions`.
    # -----------------------------------------------------------------

    @staticmethod
    def _function_spans(masked: str) -> list[tuple[str, int, int, int, int]] | None:
        """Every top-level function definition in `masked`, in source order.

        Each entry is `(name, def_start, def_end, body_start, body_end)` where
        `[def_start, def_end)` spans the whole definition (header + body, for
        blanking) and `[body_start, body_end)` spans just the body (for a callee
        scan that must not mistake a function's own header for a call to itself).

        Reuses `FunctionManager.FUNC_PATTERN`/`CONTROL_KEYWORDS` so a multi-line
        parameter list (its `[^\\)]*` spans newlines) and an `else if (` false
        match are handled identically to real function extraction. Returns
        `None` -- "do nothing, bias toward keeping" -- on an unbalanced brace,
        the same policy `remove_dead_blocks` applies per-block.
        """
        spans: list[tuple[str, int, int, int, int]] = []
        pos = 0
        while (m := FUNC_PATTERN.search(masked, pos)):
            def_start, header_end = m.span()
            name = m.group('name')
            ret_type = m.group('ret_type').strip()
            ret_last_word = ret_type.rsplit(None, 1)[-1] if ret_type else ''

            if ret_last_word in CONTROL_KEYWORDS or name in CONTROL_KEYWORDS:
                pos = def_start + 1
                continue

            open_brace = header_end - 1
            if (close_brace := BindingRegistry._match_brace(masked, open_brace)) is None:
                logger.warning("DFE: unbalanced braces in function '%s' -- keeping everything.", name)
                return None

            spans.append((name, def_start, close_brace + 1, open_brace + 1, close_brace))
            pos = close_brace + 1

        return spans

    @staticmethod
    def _analyze_reachability(
        src: str,
    ) -> tuple[list[tuple[str, int, int, int, int]], set[str], set[str]] | None:
        """Walks the call graph of `src` from `main`, on masked text.

        Returns `(spans, reachable_names, unresolved_calls)`:
          - `spans`: every top-level function definition, as `_function_spans` returns.
          - `reachable_names`: function names reachable from `main`, keyed by NAME (an
            overload group is one node -- if any overload is called, the name is reachable
            and every body sharing it is kept; a textual walk cannot resolve overloads, and
            over-inclusion is the safe direction).
          - `unresolved_calls`: names called from `main`, from a name in `reachable_names`,
            or from module-scope code (outside any function -- a global initializer counts
            as a root, same as `main`), that resolve to no definition anywhere in `src`.
            Builtins/type-constructors end up here and are simply never looked up further.

        Returns `None` -- do nothing -- when the text can't be safely analyzed (unbalanced
        braces) or has no `main` to root the walk at.
        """
        masked = BindingRegistry._mask(src)
        if (spans := BindingRegistry._function_spans(masked)) is None: return None

        by_name: dict[str, list[tuple[int, int, int, int]]] = defaultdict(list)
        for name, def_start, def_end, body_start, body_end in spans:
            by_name[name].append((def_start, def_end, body_start, body_end))

        if 'main' not in by_name:
            logger.warning("DFE: no 'main' function found in generated unit -- keeping everything.")
            return None

        graph: dict[str, set[str]] = {}
        unresolved: dict[str, set[str]] = {}
        for name, occurrences in by_name.items():
            callees: set[str] = set()
            stray: set[str] = set()
            for _, _, body_start, body_end in occurrences:
                for call in CALL_SITE_PATTERN.finditer(masked, body_start, body_end):
                    target = call.group(1)
                    (callees if target in by_name else stray).add(target)
            graph[name] = callees
            unresolved[name] = stray

        module_scope_text, cursor = [], 0
        for _, def_start, def_end, _, _ in spans:
            module_scope_text.append(masked[cursor:def_start])
            cursor = def_end
        module_scope_text.append(masked[cursor:])
        module_calls = {m.group(1) for m in CALL_SITE_PATTERN.finditer(''.join(module_scope_text))}

        roots = {'main'} | (module_calls & by_name.keys())
        reachable: set[str] = set()
        stack = list(roots)
        while stack:
            n = stack.pop()
            if n in reachable: continue
            reachable.add(n)
            stack.extend(graph.get(n, ()))

        reachable_unresolved = set(module_calls - by_name.keys())
        for n in reachable:
            reachable_unresolved |= unresolved.get(n, set())

        return spans, reachable, reachable_unresolved

    @staticmethod
    def remove_dead_functions(src: str) -> str:
        """Blanks every top-level function definition in `src` unreachable from `main`.

        Must run before `remove_dead_blocks` (and before `allocate_artifact`) so the block
        DCE sees only text `main` can actually reach -- otherwise an `[export()]`ed helper
        that this particular entry point never calls still keeps whatever SSBO/UBO blocks
        it touches alive. Same bias as `remove_dead_blocks`: keep when uncertain, since a
        wrongly-removed function is a loud compile error, never silent wrongness.

        Blanks with `'\\n' * newline-count`, exactly like `remove_dead_blocks`, so `#line`
        directives further down the unit stay line-accurate.
        """
        if (analysis := BindingRegistry._analyze_reachability(src)) is None: return src
        spans, reachable, _ = analysis

        removable = [(def_start, def_end) for name, def_start, def_end, _, _ in spans if name not in reachable]
        if not removable: return src
        removable.sort()

        out, cursor = [], 0
        for start, end in removable:
            out.append(src[cursor:start])
            out.append('\n' * src.count('\n', start, end))
            cursor = end
        out.append(src[cursor:])
        return ''.join(out)

    @staticmethod
    def find_missing_export_calls(src: str) -> set[str]:
        """Names called from `main`-reachable code (or module scope) in `src` that resolve
        to no function definition anywhere in `src` itself.

        `Shader._build` must call this before `remove_dead_functions` touches the same
        `src`: this diagnostic exists to name the fix for a helper that was never emitted
        at all, and once DFE runs the reachable call sites it needs are the only evidence
        that a call happened -- checking after would mean re-deriving exactly what DFE just
        threw away, for no benefit.

        A name that comes back here is either a builtin/type-constructor (not this
        diagnostic's concern) or a real problem for the caller to classify against whatever
        it knows about the project's module-scope functions (same file, `[include]` closure).
        """
        if (analysis := BindingRegistry._analyze_reachability(src)) is None: return set()
        _, _, reachable_unresolved = analysis
        return reachable_unresolved
