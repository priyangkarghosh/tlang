# -------------------------------------------------------------
# @file          attribute_handlers.py
# @author        Priyangkar Ghosh
# @created       2025-06-17
# @description   The declarative attribute table (REGISTRY) plus the
#                bodies of the handful of attributes that are genuinely
#                imperative. Everything else -- the ~20 bare stage markers
#                that used to be scattered across three hand-synced tables
#                -- is pure data: an AttrSpec row with no code at all.
# @license       MIT
# -------------------------------------------------------------

import logging
logger = logging.getLogger(__name__)

import regex as re
from dataclasses import replace
from typing import Any

from tlang.frontend.attribute_registry import (
    AttrCtx,
    AttrRegistry,
    AttrSpec,
    Param,
    Scope,
    USE_ARG,
    parse_resourceblock,
)
from tlang.errors import SourceLocation, TlangAttributeError, TlangError
from tlang.frontend.function_manager import FunctionDef, FunctionList, InterfaceRef
from tlang.frontend.interface_registry import InterfaceKind, emit_glsl, parse_declarator_at, parse_struct_at
from tlang.shader_stages import ShaderStage
from tlang.shader_utils import mask_comments_and_strings


class AttributeHandlers:
    """Bodies for attributes whose effect can't be expressed as a fixed
    literal or a settings-slot assignment -- they mutate `FunctionList`,
    `glob_attachments`, or a function's `StageConfig` directly."""

    @staticmethod
    def shader(ctx: AttrCtx, args: dict[str, Any]) -> None:
        if not isinstance(funcs := ctx.funcs, FunctionList) or ctx.index is None:
            raise TlangAttributeError("[shader]: internal error -- missing function context", ctx.attr.location)
        if fn := funcs.find_next(ctx.index):
            if fn.stage: raise TlangAttributeError(f"Function '{fn.name}' already has a stage set.", ctx.attr.location)
            fn.stage = ShaderStage.from_token(args['stage'])

    @staticmethod
    def program(ctx: AttrCtx, args: dict[str, Any]) -> None:
        if not isinstance(attachments := ctx.glob_attachments, list):
            raise TlangAttributeError("[program]: internal error -- missing glob_attachments", ctx.attr.location)
        attachments.append(ctx.attr)

    @staticmethod
    def dependency(ctx: AttrCtx, args: dict[str, Any]) -> None:
        if not isinstance(attachments := ctx.glob_attachments, list):
            raise TlangAttributeError("[include]: internal error -- missing glob_attachments", ctx.attr.location)
        attachments.append(ctx.attr)

    @staticmethod
    def extension(ctx: AttrCtx, args: dict[str, Any]) -> None:
        if not isinstance(attachments := ctx.glob_attachments, list):
            raise TlangAttributeError(f"[{ctx.attr.name}]: internal error -- missing glob_attachments", ctx.attr.location)
        attachments.append(ctx.attr)

    @staticmethod
    def link(ctx: AttrCtx, args: dict[str, Any]) -> None:
        """[link(name)] means "this function links against/needs `name`".

        The DECORATED function (`fn`, the one immediately following this
        attribute) gains `name`'s FunctionDef in `fn.links` -- NOT the other
        way around -- so that `Shader._build` (which emits `func.links`
        bodies immediately before `func`'s own body) inlines `name` into
        `fn`'s stage source ahead of `fn`'s entry point.
        """
        if not isinstance(funcs := ctx.funcs, FunctionList) or ctx.index is None:
            raise TlangAttributeError("[link]: internal error -- missing function context", ctx.attr.location)
        if not (fn := funcs.find_next(ctx.index)):
            raise TlangAttributeError("Could not find which function this attribute is attached to", ctx.attr.location)

        target_name = args['name']
        candidates = funcs.keyed_items.get(target_name)
        if not candidates:
            raise TlangAttributeError(f"Could not find function to link with: '{target_name}'", ctx.attr.location)
        if len(candidates) > 1:
            raise TlangAttributeError(
                f"[link('{target_name}')] is ambiguous: {len(candidates)} overloads of '{target_name}' exist",
                ctx.attr.location,
            )
        link_fn = candidates[0]

        if link_fn is fn:
            raise TlangAttributeError(f"Function '{fn.name}' cannot [link] to itself", ctx.attr.location)

        # cycle guard: reject the new edge fn -> link_fn if link_fn can
        # already (transitively) reach fn, which would make the link graph
        # circular
        def reaches(start: FunctionDef, target: FunctionDef, seen: set[FunctionDef]) -> bool:
            if start in seen: return False
            seen.add(start)
            return any(nxt is target or reaches(nxt, target, seen) for nxt in start.links)
        if reaches(link_fn, fn, set()):
            raise TlangAttributeError(
                f"[link('{target_name}')] would create a circular link involving '{fn.name}'", ctx.attr.location,
            )

        # link the two functions -- dedupe so a function is never emitted
        # twice into the same stage (a GLSL redefinition error)
        if not any(l is link_fn for l in fn.links): fn.links.append(link_fn)

    @staticmethod
    def export(ctx: AttrCtx, args: dict[str, Any]) -> None:
        if not isinstance(funcs := ctx.funcs, FunctionList) or ctx.index is None:
            raise TlangAttributeError("[export]: internal error -- missing function context", ctx.attr.location)
        if fn := funcs.find_next(ctx.index): fn.exported = True

    @staticmethod
    def numthreads(ctx: AttrCtx, args: dict[str, Any]) -> None:
        """Deferred (resolved in ShaderProcessor, once this function's
        StageConfig exists) so that a second [numthreads(...)] on the same
        function is a detected conflict instead of a silently-overwritten
        `func.config` entry.

        NOTE: x/y/z are deliberately NOT coerced to int -- `numthreads(BLOCK_SIZE, 1, 1)`
        is a legitimate GLSL preprocessor macro name, resolved later by the
        driver, not a Python integer.
        """
        if ctx.stage_config is None:
            raise TlangAttributeError("[numthreads]: internal error -- missing stage config", ctx.attr.location)
        tokens = {'local_size_x': args['x'], 'local_size_y': args['y'], 'local_size_z': args['z']}
        origin = f"[numthreads({ctx.attr.raw_args})] at {ctx.attr.location}"
        ctx.stage_config.set_layout('in', tokens, origin, loc=ctx.attr.location)

    @staticmethod
    def resourceblock(ctx: AttrCtx, args: dict[str, Any]) -> None:
        """Deferred for the same reason as numthreads: a verbatim
        `layout(...) in|out;` line inside the block must be checked against
        whatever the target stage would otherwise auto-generate for that
        direction, which isn't known until the function's stage is."""
        if ctx.stage_config is None:
            raise TlangAttributeError("[resourceblock]: internal error -- missing stage config", ctx.attr.location)
        claims, remainder = parse_resourceblock(ctx.attr.raw_args)
        origin = f"[resourceblock] at {ctx.attr.location}"
        for direction, tokens in claims.items():
            ctx.stage_config.set_layout(direction, tokens, origin, exclusive=True, loc=ctx.attr.location)
        ctx.stage_config.add_decl(remainder, origin)

    # -- declaration attributes: [varyings]/[uniforms]/[buffer] --

    @staticmethod
    def _expected_after_desc(ctx: AttrCtx) -> str:
        """[buffer] also accepts the single-declarator shorthand; the other
        two declaration attributes only ever take the struct form."""
        if ctx.attr.name == 'buffer':
            return "a 'struct Name { ... };' declaration or a single 'Type name[...];' declarator"
        return "a 'struct Name { ... };' declaration"

    @staticmethod
    def _find_struct_start(ctx: AttrCtx) -> int:
        """First real-content line after this attribute: a still-present
        `ctx.src_map` line, or a function header (even though its body was
        already popped out of `ctx.src_map`)."""
        funcs, src_map = ctx.funcs, ctx.src_map
        assert src_map is not None and ctx.end_index is not None
        expected = AttributeHandlers._expected_after_desc(ctx)

        hi = max(src_map, default=0)
        if isinstance(funcs, FunctionList) and funcs.items:
            hi = max(hi, max(fn.line_end for fn in funcs.items))

        line = ctx.end_index + 1
        while line <= hi:
            if isinstance(funcs, FunctionList) and (fn := funcs.find_within(line)) is not None:
                found = fn.line_body[fn.line_start].data.strip()
                raise TlangAttributeError(
                    f"[{ctx.attr.name}]: expected {expected} on the following line, found '{found}'",
                    ctx.attr.location,
                )
            if line in src_map and mask_comments_and_strings(src_map[line].data).strip():
                return line
            line += 1

        raise TlangAttributeError(
            f"[{ctx.attr.name}]: expected {expected} on the following line, found '<end of file>'",
            ctx.attr.location,
        )

    @staticmethod
    def _declare_struct_same_line(ctx: AttrCtx, kind: InterfaceKind, opts: dict[str, Any]) -> None:
        """`[varyings] struct VertexOut { ... };` -- the struct opens in the tail of the
        attribute's own line. Its body may run on past that line, so the tail is joined with
        the lines after it before parsing, and every location is rebased onto `ctx.end_index`,
        the physical line the struct opens on.
        """
        assert isinstance(ctx.funcs, FunctionList) and ctx.src_map is not None and ctx.end_index is not None
        start = ctx.end_index
        loc = SourceLocation(ctx.shader_name, start)
        opts.pop('name', None)

        following = sorted(n for n in ctx.src_map if n > start)
        joined = ctx.line_tail + ''.join(ctx.src_map[n].data for n in following)

        try:
            result = parse_struct_at(joined, 0, ctx.shader_name, kind, **opts)
        except TlangError as exc:
            raise type(exc)(exc.message, loc) from None
        if result is None:
            raise TlangAttributeError(
                f"[{ctx.attr.name}]: expected {AttributeHandlers._expected_after_desc(ctx)}, "
                f"found '{ctx.line_tail.strip()}'",
                loc,
            )

        decl, end_offset = result
        rebase = start - 1
        decl = replace(decl, line=decl.line + rebase, members=tuple(
            replace(m, line=m.line + rebase) for m in decl.members
        ))
        ctx.funcs.interfaces.add(decl)

        emitted = [] if kind is InterfaceKind.VARYINGS else emit_glsl(decl)
        ctx.result = ('\n'.join(emitted) + '\n') if emitted else '\n'
        ctx.tail_consumed = True
        for n in following[:joined.count('\n', 0, end_offset)]:
            ctx.src_map[n].data = '\n'

    @staticmethod
    def _declare_buffer_same_line(ctx: AttrCtx, opts: dict[str, Any]) -> None:
        """`[buffer] vec2 name[];` -- the declarator sits in the tail of the
        attribute's own line, not on the line after it. `ctx.line_tail` is
        confined to that one physical line, so every location computed
        against it is remapped onto `ctx.end_index` (the real file line)
        before it can reach an author -- on both the success and error paths.
        """
        assert isinstance(ctx.funcs, FunctionList) and ctx.end_index is not None
        block_name = opts.pop('name', None)
        loc = SourceLocation(ctx.shader_name, ctx.end_index)

        try:
            result = parse_declarator_at(
                ctx.line_tail, 0, ctx.shader_name, InterfaceKind.BUFFER, block_name=block_name, **opts,
            )
        except TlangError as exc:
            raise type(exc)(exc.message, loc) from None

        if result is None:
            raise TlangAttributeError(
                f"[buffer]: expected {AttributeHandlers._expected_after_desc(ctx)}, "
                f"found '{ctx.line_tail.strip()}'",
                loc,
            )
        decl, end_offset = result
        decl = replace(decl, line=ctx.end_index, members=tuple(replace(m, line=ctx.end_index) for m in decl.members))

        ctx.funcs.interfaces.add(decl)
        lines = emit_glsl(decl)
        ctx.result = ('\n'.join(lines) if lines else '') + ctx.line_tail[end_offset:]
        ctx.tail_consumed = True

    @staticmethod
    def _declare_interface(ctx: AttrCtx, kind: InterfaceKind, opts: dict[str, Any]) -> None:
        if ctx.src_map is None or ctx.end_index is None or not isinstance(ctx.funcs, FunctionList):
            raise TlangAttributeError(f"[{ctx.attr.name}]: internal error -- missing source map", ctx.attr.location)

        tail_masked = mask_comments_and_strings(ctx.line_tail)
        if tail_masked.strip():
            if re.match(r'\s*struct\b', tail_masked):
                AttributeHandlers._declare_struct_same_line(ctx, kind, opts)
                return
            if kind is not InterfaceKind.BUFFER:
                raise TlangAttributeError(
                    f"[{ctx.attr.name}]: expected {AttributeHandlers._expected_after_desc(ctx)} on the "
                    f"following line, not on the same line as the attribute (found "
                    f"'{ctx.line_tail.strip()}')",
                    ctx.attr.location,
                )
            AttributeHandlers._declare_buffer_same_line(ctx, opts)
            return

        start = AttributeHandlers._find_struct_start(ctx)
        indices = sorted(i for i in ctx.src_map if i >= start)
        joined = ''.join(ctx.src_map[i].data for i in indices)
        is_struct = bool(re.match(r'\s*struct\b', mask_comments_and_strings(ctx.src_map[start].data)))

        if is_struct:
            result = parse_struct_at(joined, 0, ctx.shader_name, kind, **opts)
        elif kind is InterfaceKind.BUFFER:
            # the single-declarator shorthand: `[buffer] vec2 name[];` in
            # place of the struct form -- reuses the same member parser, so
            # emission/the interface table/cross-module comparison need no
            # further change
            block_name = opts.pop('name', None)
            result = parse_declarator_at(joined, 0, ctx.shader_name, kind, block_name=block_name, **opts)
        else:
            result = None

        if result is None:
            raise TlangAttributeError(
                f"[{ctx.attr.name}]: expected {AttributeHandlers._expected_after_desc(ctx)} on the "
                f"following line, found '{ctx.src_map[start].data.strip()}'",
                ctx.attr.location,
            )
        decl, end_offset = result

        # `joined` line 1 == real source line `start` -- rebase both decl
        # and member line numbers back onto the real file
        rebase = start - 1
        decl = replace(decl, line=decl.line + rebase, members=tuple(
            replace(m, line=m.line + rebase) for m in decl.members
        ))
        end_line = start + joined.count('\n', 0, end_offset)

        ctx.funcs.interfaces.add(decl)

        if kind is InterfaceKind.VARYINGS:
            ctx.src_map[start].data = '\n'
        else:
            lines = emit_glsl(decl)
            ctx.src_map[start].data = '\n'.join(lines) + '\n' if lines else '\n'
        for i in range(start + 1, end_line + 1):
            if i in ctx.src_map: ctx.src_map[i].data = '\n'

    @staticmethod
    def varyings(ctx: AttrCtx, args: dict[str, Any]) -> None:
        AttributeHandlers._declare_interface(ctx, InterfaceKind.VARYINGS, {'locations': args['locations']})

    @staticmethod
    def uniforms(ctx: AttrCtx, args: dict[str, Any]) -> None:
        layout = args['layout']
        AttributeHandlers._declare_interface(ctx, InterfaceKind.UNIFORMS, {'layout': layout, 'block': layout == 'std140'})

    @staticmethod
    def buffer(ctx: AttrCtx, args: dict[str, Any]) -> None:
        name = args.get('name')
        if name is not None and not name.isidentifier():
            raise TlangAttributeError(
                f"[buffer(name=...)]: '{name}' is not a valid GLSL block name -- give a valid identifier",
                ctx.attr.location,
            )
        AttributeHandlers._declare_interface(ctx, InterfaceKind.BUFFER, {'layout': args['layout'], 'name': name})

    @staticmethod
    def uses(ctx: AttrCtx, args: dict[str, Any]) -> None:
        """Records the reference; ShaderProcessor.resolve_interfaces emits it."""
        if ctx.func is None:
            raise TlangAttributeError("[uses]: internal error -- missing function context", ctx.attr.location)
        if ctx.attr.location is None:
            raise TlangAttributeError("[uses]: internal error -- missing source location", None)
        ctx.func.iface_refs.append(InterfaceRef(args['name'], args['dir'], ctx.attr.location))


# ---------------------------------------------------------------------------
# the registry -- one row per (name, stage). Bare markers and stage-settings
# meta-attributes carry no handler at all: `sets`/`params` fully describe
# their effect, and ShaderProcessor applies them generically.
# ---------------------------------------------------------------------------

_GEOM_IN_PRIMS = ('points', 'lines', 'triangles', 'lines_adjacency', 'triangles_adjacency')
_GEOM_OUT_PRIMS = ('points', 'line_strip', 'triangle_strip')
_TESE_IN_PRIMS = ('triangles', 'quads', 'isolines')
_TESE_SPACING = ('equal_spacing', 'fractional_even_spacing', 'fractional_odd_spacing')
_TESE_WINDING = ('cw', 'ccw')

SPECS: list[AttrSpec] = [
    # -- imperative, stage-independent, dispatched immediately at attach time --
    AttrSpec(
        name='shader', scope=Scope.GLOBAL, params=(Param('stage', str, positional=0),),
        handler=AttributeHandlers.shader,
        summary="Assigns the shader stage of the next function.",
        example="[shader('compute')]",
    ),
    AttrSpec(
        name='program', scope=Scope.GLOBAL, params=(Param('name', str, positional=0),), variadic=True,
        handler=AttributeHandlers.program,
        summary="Declares a linked program, mapping stage kwargs (vert=, frag=, ...) to entry points.",
        example="[program('default', vert='vs_main', frag='fs_main')]",
    ),
    AttrSpec(
        name='include', scope=Scope.GLOBAL, handler=AttributeHandlers.dependency,
        summary="Pulls another module's exported declarations into this one.",
        example="[include(math)]",
    ),
    AttrSpec(
        name='extend', scope=Scope.GLOBAL, handler=AttributeHandlers.extension,
        summary="Enables a GLSL extension (or named extension group) with ': enable'.",
        example="[extend(int64)]",
    ),
    AttrSpec(
        name='extend!', scope=Scope.GLOBAL, handler=AttributeHandlers.extension,
        summary="Enables a GLSL extension (or named extension group) with ': require'.",
        example="[extend!(int64)]",
    ),
    AttrSpec(
        name='require', scope=Scope.GLOBAL, handler=AttributeHandlers.extension,
        summary="Alias family for [extend!(...)] -- ': require' semantics.",
        example="[require(int64)]",
    ),
    AttrSpec(
        name='link', scope=Scope.GLOBAL, params=(Param('name', str, positional=0),),
        handler=AttributeHandlers.link,
        summary="Inlines another function's body immediately before this one's, in its stage source.",
        example="[link('helper')]",
    ),
    AttrSpec(
        name='export', scope=Scope.GLOBAL, handler=AttributeHandlers.export,
        summary="Marks a non-stage function as part of this module's public (includable) surface.",
        example="[export]",
    ),

    # -- imperative, stage-independent, but deferred until this function's
    #    StageConfig exists (so duplicates/conflicts are detected) --
    AttrSpec(
        name='numthreads', scope=Scope.GLOBAL, deferred=True,
        params=(
            Param('x', str, positional=0, default='1'),
            Param('y', str, positional=1, default='1'),
            Param('z', str, positional=2, default='1'),
        ),
        handler=AttributeHandlers.numthreads,
        summary="Sets the compute work-group size (local_size_x/y/z).",
        example="[numthreads(64, 1, 1)]",
    ),
    AttrSpec(
        name='resourceblock', scope=Scope.GLOBAL, deferred=True,
        handler=AttributeHandlers.resourceblock,
        summary="Injects verbatim GLSL immediately before this function's body.",
        example="[resourceblock(out vec4 fragColor;)]",
    ),

    # -- declaration attributes: desugar the following struct to flat GLSL --
    AttrSpec(
        name='varyings', scope=Scope.GLOBAL,
        params=(Param('locations', bool, default=True),),
        handler=AttributeHandlers.varyings,
        summary="Declares a flat varying interface from the following struct; emits GLSL only per [uses(...)] reference, one direction per stage.",
        example="[varyings]\nstruct VertexOut { vec3 color; vec2 uv; };",
    ),
    AttrSpec(
        name='uniforms', scope=Scope.GLOBAL,
        params=(Param('layout', str, choices=('', 'std140'), default='', positional=0),),
        handler=AttributeHandlers.uniforms,
        summary="Declares loose uniforms (default), or a std140 UBO block, from the following struct.",
        example="[uniforms(std140)]\nstruct Frame { mat4 view; float time; };",
    ),
    AttrSpec(
        name='buffer', scope=Scope.GLOBAL,
        params=(
            Param('layout', str, choices=('std430', 'std140'), default='std430', positional=0),
            Param('name', str, default=None),
        ),
        handler=AttributeHandlers.buffer,
        summary="Declares an SSBO block from the following struct, or from a single declarator "
                "('[buffer] vec2 x[];') whose block name is derived by upper-casing the member's "
                "first character -- override with name='...' when that's wrong.",
        example="[buffer(std430)]\nstruct Particles { vec4 pos[]; };\n[buffer] vec2 ptcPositions[];",
    ),

    # -- reference attribute: ties a stage function to a declared interface --
    AttrSpec(
        name='uses', scope=Scope.GLOBAL, deferred=True,
        params=(
            Param('name', str, positional=0),
            Param('dir', str, choices=('in', 'out'), default='in'),
        ),
        handler=AttributeHandlers.uses,
        summary="References a declared varyings interface for this function's stage, in the given direction.",
        example="[uses(VertexOut, dir='out')]",
    ),

    # -- stage-settings meta attributes: `[frag(...)]`/`[geom(...)]`/... --
    AttrSpec(
        name='frag', scope=Scope.GLOBAL, stages=frozenset({ShaderStage.FRAG}), deferred=True,
        params=(
            Param('early_tests', bool, default=False, direction='in',
                  render=lambda v: ('early_fragment_tests', None) if v else None),
        ),
        summary="Fragment-stage settings.", example="[frag(early_tests=true)]",
    ),
    AttrSpec(
        name='geom', scope=Scope.GLOBAL, stages=frozenset({ShaderStage.GEOM}), deferred=True,
        params=(
            Param('in', str, choices=_GEOM_IN_PRIMS, default='triangles', direction='in', render=lambda v: (v, None)),
            Param('out', str, choices=_GEOM_OUT_PRIMS, default='triangle_strip', direction='out', render=lambda v: (v, None)),
            Param('max_verts', int, default=3, direction='out', render=lambda v: ('max_vertices', str(v))),
            Param('stream', int, default=None, direction='out', render=lambda v: ('stream', str(v)) if v is not None else None),
        ),
        summary="Geometry-stage settings.", example="[geom(in='points', out='triangle_strip', max_verts=6)]",
    ),
    AttrSpec(
        name='tesc', scope=Scope.GLOBAL, stages=frozenset({ShaderStage.TESC}), deferred=True,
        params=(
            Param('vertices', int, default=0, direction='out', render=lambda v: ('vertices', str(v)) if v else None),
        ),
        summary="Tessellation-control-stage settings.", example="[tesc(vertices=3)]",
    ),
    AttrSpec(
        name='tese', scope=Scope.GLOBAL, stages=frozenset({ShaderStage.TESE}), deferred=True,
        params=(
            Param('in', str, choices=_TESE_IN_PRIMS, default='triangles', direction='in', render=lambda v: (v, None)),
            Param('spacing', str, choices=_TESE_SPACING, default='equal_spacing', direction='in', render=lambda v: (v, None)),
            Param('order', str, choices=_TESE_WINDING, default='ccw', direction='in', render=lambda v: (v, None)),
            Param('point_mode', bool, default=False, direction='in', render=lambda v: ('point_mode', None) if v else None),
        ),
        summary="Tessellation-evaluation-stage settings.", example="[tese(in='quads', spacing='fractional_odd_spacing')]",
    ),

    # -- bare/value markers: pure data, no handler. Each just assigns one
    #    settings-slot on its stage's settings dict; ShaderProcessor renders
    #    the final settings dict into layout qualifiers once per function. --
    AttrSpec(name='early_fragment_tests', scope=Scope.GLOBAL, stages=frozenset({ShaderStage.FRAG}), deferred=True,
              sets=('early_tests', True), summary="Shorthand for [frag(early_tests=true)].", example="[early_fragment_tests]"),

    AttrSpec(name='points', scope=Scope.GLOBAL, stages=frozenset({ShaderStage.GEOM}), deferred=True,
              sets=('in', 'points'), summary="Geometry input primitive: points.", example="[points]"),
    AttrSpec(name='lines', scope=Scope.GLOBAL, stages=frozenset({ShaderStage.GEOM}), deferred=True,
              sets=('in', 'lines'), summary="Geometry input primitive: lines.", example="[lines]"),
    AttrSpec(name='triangles', scope=Scope.GLOBAL, stages=frozenset({ShaderStage.GEOM}), deferred=True,
              aliases=('tri', 'tris'), sets=('in', 'triangles'),
              summary="Geometry input primitive: triangles.", example="[triangles]"),
    AttrSpec(name='triangles', scope=Scope.GLOBAL, stages=frozenset({ShaderStage.TESE}), deferred=True,
              sets=('in', 'triangles'), summary="Tess-eval input primitive: triangles.", example="[triangles]"),
    AttrSpec(name='triangles_adjacency', scope=Scope.GLOBAL, stages=frozenset({ShaderStage.GEOM}), deferred=True,
              aliases=('tri_adj',), sets=('in', 'triangles_adjacency'),
              summary="Geometry input primitive: triangles_adjacency.", example="[triangles_adjacency]"),
    AttrSpec(name='lines_adjacency', scope=Scope.GLOBAL, stages=frozenset({ShaderStage.GEOM}), deferred=True,
              aliases=('lines_adj',), sets=('in', 'lines_adjacency'),
              summary="Geometry input primitive: lines_adjacency.", example="[lines_adjacency]"),
    AttrSpec(name='line_strip', scope=Scope.GLOBAL, stages=frozenset({ShaderStage.GEOM}), deferred=True,
              sets=('out', 'line_strip'), summary="Geometry output primitive: line_strip.", example="[line_strip]"),
    AttrSpec(name='triangle_strip', scope=Scope.GLOBAL, stages=frozenset({ShaderStage.GEOM}), deferred=True,
              aliases=('tri_strip',), sets=('out', 'triangle_strip'),
              summary="Geometry output primitive: triangle_strip.", example="[triangle_strip]"),
    AttrSpec(name='max_verts', scope=Scope.GLOBAL, stages=frozenset({ShaderStage.GEOM}), deferred=True,
              params=(Param('value', int, positional=0),), sets=('max_verts', USE_ARG),
              summary="Geometry max output vertices.", example="[max_verts(6)]"),
    AttrSpec(name='stream', scope=Scope.GLOBAL, stages=frozenset({ShaderStage.GEOM}), deferred=True,
              params=(Param('value', int, positional=0),), sets=('stream', USE_ARG),
              summary="Geometry output transform-feedback stream index.", example="[stream(1)]"),

    AttrSpec(name='vertices', scope=Scope.GLOBAL, stages=frozenset({ShaderStage.TESC}), deferred=True,
              aliases=('vert', 'verts', 'patch', 'patch_size'), params=(Param('value', int, positional=0),),
              sets=('vertices', USE_ARG), summary="Tessellation-control output patch size.", example="[vertices(3)]"),

    AttrSpec(name='quads', scope=Scope.GLOBAL, stages=frozenset({ShaderStage.TESE}), deferred=True,
              sets=('in', 'quads'), summary="Tess-eval input primitive: quads.", example="[quads]"),
    AttrSpec(name='isolines', scope=Scope.GLOBAL, stages=frozenset({ShaderStage.TESE}), deferred=True,
              sets=('in', 'isolines'), summary="Tess-eval input primitive: isolines.", example="[isolines]"),
    AttrSpec(name='equal_spacing', scope=Scope.GLOBAL, stages=frozenset({ShaderStage.TESE}), deferred=True,
              aliases=('even',), sets=('spacing', 'equal_spacing'),
              summary="Tess-eval spacing: equal_spacing.", example="[equal_spacing]"),
    AttrSpec(name='fractional_even_spacing', scope=Scope.GLOBAL, stages=frozenset({ShaderStage.TESE}), deferred=True,
              aliases=('frac_even',), sets=('spacing', 'fractional_even_spacing'),
              summary="Tess-eval spacing: fractional_even_spacing.", example="[fractional_even_spacing]"),
    AttrSpec(name='fractional_odd_spacing', scope=Scope.GLOBAL, stages=frozenset({ShaderStage.TESE}), deferred=True,
              aliases=('odd', 'frac_odd'), sets=('spacing', 'fractional_odd_spacing'),
              summary="Tess-eval spacing: fractional_odd_spacing.", example="[fractional_odd_spacing]"),
    AttrSpec(name='cw', scope=Scope.GLOBAL, stages=frozenset({ShaderStage.TESE}), deferred=True,
              sets=('order', 'cw'), summary="Tess-eval winding: clockwise.", example="[cw]"),
    AttrSpec(name='ccw', scope=Scope.GLOBAL, stages=frozenset({ShaderStage.TESE}), deferred=True,
              sets=('order', 'ccw'), summary="Tess-eval winding: counter-clockwise.", example="[ccw]"),
    AttrSpec(name='point_mode', scope=Scope.GLOBAL, stages=frozenset({ShaderStage.TESE}), deferred=True,
              sets=('point_mode', True), summary="Tess-eval point-mode output.", example="[point_mode]"),

    # -- function-body pragmas: fixed literal text, no handler, no target function --
    AttrSpec(name='unroll', scope=Scope.FUNCBODY, literal='#pragma unroll', summary="Requests loop unrolling.", example="[unroll]"),
    AttrSpec(name='flatten', scope=Scope.FUNCBODY, literal='#pragma flatten', summary="Requests branch flattening.", example="[flatten]"),
    AttrSpec(name='branch', scope=Scope.FUNCBODY, literal='#pragma branch', summary="Requests explicit branching.", example="[branch]"),
    AttrSpec(name='loop', scope=Scope.FUNCBODY, literal='#pragma loop', summary="Requests an explicit loop.", example="[loop]"),
    AttrSpec(name='fastopt', scope=Scope.FUNCBODY, literal='#pragma optimize(on)', summary="Enables compiler optimization.", example="[fastopt]"),
    AttrSpec(name='noopt', scope=Scope.FUNCBODY, literal='#pragma optimize(off)', summary="Disables compiler optimization.", example="[noopt]"),
]

REGISTRY = AttrRegistry(SPECS)

__all__ = ['AttributeHandlers', 'SPECS', 'REGISTRY']
