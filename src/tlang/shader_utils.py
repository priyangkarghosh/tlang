# -------------------------------------------------------------
# @file          shader_utils.py
# @author        Priyangkar Ghosh
# @created       2025-06-10
# @description   Extension-group constants plus comment/string-aware text-scanning helpers
#                shared by the attribute and function extraction passes.
# @license       MIT
# -------------------------------------------------------------


def mask_comments_and_strings(src: str, mask_strings: bool = True) -> str:
    """Return a same-length copy of ``src`` with ``//`` / ``/* */`` comments
    (and, unless ``mask_strings`` is False, ``"..."`` / ``'...'`` string
    literals) blanked out to spaces.

    Newlines are always preserved (even inside block comments), so line
    numbers computed against the mask line up exactly with the original
    text. Callers should use the mask only to locate/measure things (regex
    matches, bracket depth, …) and slice the real content out of ``src``.

    ``mask_strings=False`` masks comments only, leaving quote characters and
    their contents untouched -- for callers whose own parsing (e.g. a regex
    with quoted-value alternatives) needs to see real quotes, but still
    wants a comment's contents to never look structural.
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
            # Always skip over the whole string (so e.g. a `//` inside a quoted URL isn't
            # mistaken for a comment start below); only blank it when mask_strings is set.
            if mask_strings:
                for k in range(i, min(j, n)):
                    if out[k] != '\n': out[k] = ' '
            i = j
            continue

        i += 1
    return ''.join(out)


# Tlang-defined convenience aliases for `#extension` sets, expanded by
# [extend(...)]/[require(...)] (see ShaderProcessor._process_global_attrs). Each
# group name below is a single token a shader can request; the listed GL
# extensions are what actually gets emitted as `#extension ... : enable/require`
# lines. Keeping the exact membership documented here means "I asked for group X
# and feature Y doesn't resolve" is a debuggable statement instead of a driver
# mystery -- see the int64 group's comment below for the case that prompted this.
EXTENSION_GROUPS: dict[str, list[str]] = {
    'vulkan_glsl': [
        'GL_KHR_vulkan_glsl',
    ],
    # NVIDIA gates the 64-bit atomic builtin overloads (e.g. atomicCompSwap on a
    # uint64_t/u64vec2 in an SSBO) behind GL_NV_gpu_shader5 -- its int64 *type*
    # extension -- not behind GL_NV_shader_atomic_int64, despite the name.
    # Verified empirically (RTX 3090, NVIDIA 616.64, GL 4.6): with GL_NV_gpu_shader5
    # absent, atomicCompSwap(uint64_t, u64vec2) fails to compile with
    # "error C1115: unable to find compatible overloaded function", which reads
    # like a hardware limitation but isn't. Do not remove this as a "duplicate"
    # of GL_ARB_gpu_shader_int64.
    'int64': [
        'GL_ARB_gpu_shader_int64',
        'GL_EXT_shader_atomic_int64',
        'GL_KHR_shader_atomic_int64',
        'GL_NV_shader_atomic_int64',
        'GL_NV_gpu_shader5',
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

