from typing import Any


EXTENSION_GROUPS: dict[str, list[str]] = {
    'subgroup': [
        'GL_KHR_shader_subgroup_basic',
        'GL_KHR_shader_subgroup_vote',
        'GL_KHR_shader_subgroup_ballot',
        'GL_KHR_shader_subgroup_arithmetic',
    ],
    'subgroup_all': [
        'GL_KHR_shader_subgroup_basic',
        'GL_KHR_shader_subgroup_vote',
        'GL_KHR_shader_subgroup_ballot',
        'GL_KHR_shader_subgroup_arithmetic',
        'GL_KHR_shader_subgroup_shuffle',
        'GL_KHR_shader_subgroup_shuffle_relative',
        'GL_KHR_shader_subgroup_clustered',
        'GL_KHR_shader_subgroup_quad',
    ],
}

ALIAS: dict[str, str] = {
    # geometry
    'tri': 'triangles',
    'tris': 'triangles',
    'tri_adj': 'triangles_adjacency',
    'lines_adj': 'lines_adjacency',
    'tri_strip': 'triangle_strip',
    'line_strip': 'line_strip',
    # tess spacing
    'even': 'equal_spacing',
    'odd': 'fractional_odd_spacing',
    'frac_even': 'fractional_even_spacing',
    'frac_odd': 'fractional_odd_spacing',
    # tessellation patch size
    'vert': 'vertices',
    'verts': 'vertices',
    'patch': 'vertices',
    'patch_size': 'vertices',
}

FRAG_DEFAULTS: dict[str, Any] = {
    'early_tests': False,
}

GEOM_DEFAULTS: dict[str, Any] = {
    'in': 'triangles',
    'out': 'triangle_strip',
    'max_verts': 3,
}

TESC_DEFAULTS: dict[str, int] = {
    'vertices': 0,
}

TESE_DEFAULTS: dict[str, str] = {
    'in': 'triangles',
    'spacing': 'equal_spacing',
    'order': 'ccw',
}

FRAG_EMIT_IN: tuple = (
    ('early_tests', lambda v: 'early_fragment_tests' if v else None),
)

GEOM_EMIT_IN: tuple = (
    ('in', lambda v: v),
)

GEOM_EMIT_OUT: tuple = (
    ('out', lambda v: v),
    ('max_verts', lambda v: f'max_vertices = {v}'),
)

TESC_EMIT_OUT: tuple = (
    ('vertices', lambda v: f'vertices = {v}' if v else None),
)

TESE_EMIT_IN: tuple = (
    ('in', lambda v: v),
    ('spacing', lambda v: v),
    ('order', lambda v: v),
)

SIMPLE_ATTR: dict[str, Any] = {
    # frag
    'early_fragment_tests': ('frag', 'early_tests', 'flag'),

    # geom
    'points': ('geom', 'in', 'alias', 'points'),
    'lines': ('geom', 'in', 'alias', 'lines'),
    'triangles': ('geom', 'in', 'alias', 'triangles'),
    'line_strip': ('geom', 'out', 'alias', 'line_strip'),
    'triangle_strip': ('geom', 'out', 'alias', 'triangle_strip'),
    'max_verts': ('geom', 'max_verts', 'value'),

    # tesc
    'vertices': ('tesc', 'vertices', 'value'),

    # tese
    'triangles_tese': ('tese', 'in', 'alias', 'triangles'),
    'quads': ('tese', 'in', 'alias', 'quads'),
    'isolines': ('tese', 'in', 'alias', 'isolines'),
    'equal_spacing': ('tese', 'spacing', 'alias', 'equal_spacing'),
    'fractional_even_spacing': ('tese', 'spacing', 'alias', 'fractional_even_spacing'),
    'fractional_odd_spacing': ('tese', 'spacing', 'alias', 'fractional_odd_spacing'),
    'cw': ('tese', 'order', 'alias', 'cw'),
    'ccw': ('tese', 'order', 'alias', 'ccw'),
}