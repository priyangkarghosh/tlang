# -------------------------------------------------------------
# @file          shader_utils.py
# @author        Priyangkar Ghosh
# @created       2025-06-10
# @description   Extension-group constants plus comment/string-aware text-scanning helpers
#                shared by the attribute and function extraction passes.
# @license       MIT
# -------------------------------------------------------------


def mask_comments_and_strings(src: str) -> str:
    """Return a same-length copy of ``src`` with ``//`` / ``/* */`` comments
    and ``"..."`` / ``'...'`` string literals blanked out to spaces.

    Newlines are always preserved (even inside block comments), so line
    numbers computed against the mask line up exactly with the original
    text. Callers should use the mask only to locate/measure things (regex
    matches, bracket depth, …) and slice the real content out of ``src``.
    """
    out = list(src)
    n = len(src)
    i = 0
    while i < n:
        two = src[i:i + 2]

        if two == '//':
            end = j if (j := src.find('\n', i)) != -1 else n
            for k in range(i, end): out[k] = ' '
            i = end
            continue

        if two == '/*':
            end = j + 2 if (j := src.find('*/', i + 2)) != -1 else n
            for k in range(i, end):
                if out[k] != '\n': out[k] = ' '
            i = end
            continue

        if (quote := src[i]) in ('"', "'"):
            j = i + 1
            while j < n:
                if src[j] == '\\': j += 2; continue
                j += 1
                if src[j - 1] == quote: break
            for k in range(i, min(j, n)):
                if out[k] != '\n': out[k] = ' '
            i = j
            continue

        i += 1
    return ''.join(out)


EXTENSION_GROUPS: dict[str, list[str]] = {
    'vulkan_glsl': [
        'GL_KHR_vulkan_glsl',
    ],
    'int64': [
        'GL_ARB_gpu_shader_int64',
        'GL_EXT_shader_atomic_int64',
        'GL_KHR_shader_atomic_int64',
        'GL_NV_shader_atomic_int64'
    ],
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

