import re
import logging

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────────────────────
# 1) BUFFER_PATTERN now matches:
#      [coherent ] buffer<Type> Name[Size] [: @binding(N)];
#    and captures qualifier="coherent " (if present), type, name, size, binding.
# ──────────────────────────────────────────────────────────────────────────────
BUFFER_PATTERN = re.compile(
    r'''
    (?P<qualifier>coherent\s*)?          # optional "coherent " 
    buffer<(?P<type>\w+)>\s+             #      "buffer<Type>" 
    (?P<name>\w+)\[(?P<size>[\w\d_]+)\]  #      "Name[Size]"
    (?:\s*:\s*@binding\((?P<binding>\d+)\))?  # optional ": @binding(N)"
    \s*;                                  # trailing semicolon
    ''',
    re.VERBOSE
)

# ──────────────────────────────────────────────────────────────────────────────
# 2) SHARED_OR_UNIFORM_PATTERN matches:
#      shared<Type> Name[ ]?;
#      or uniform<Type> Name;
#    capturing qualifier, type, name, and optional array "[]".
# ──────────────────────────────────────────────────────────────────────────────
SHARED_OR_UNIFORM_PATTERN = re.compile(
    r'''
    (?P<su>(?:shared|uniform))<(?P<type>\w+)>\s+  # "shared<Type>" or "uniform<Type>"
    (?P<name>\w+)
    (?P<array>\[\])?                              # optional "[]"
    \s*;                                          # semicolon
    ''',
    re.VERBOSE
)

# ──────────────────────────────────────────────────────────────────────────────
# 3) ATOMIC_UINT_PATTERN matches:
#      atomic_uint Name : @binding(N) @offset(M);
#    capturing name, binding, and offset.
# ──────────────────────────────────────────────────────────────────────────────
ATOMIC_UINT_PATTERN = re.compile(
    r'''
    atomic_uint\s+                      # "atomic_uint "
    (?P<name>\w+)\s*                    #    "Name"
    :\s*@binding\((?P<binding>\d+)\)\s+#    ": @binding(N)"
    @offset\((?P<offset>\d+)\)\s*;      #   "@offset(M);"
    ''',
    re.VERBOSE
)


class MemoryManager:
    @classmethod
    def refactor(cls, src: str) -> str:
        logger.debug("Refactoring memory declarations")
        # First rewrite shared/uniform, then atomic, then buffers.
        out = cls._rewrite_shared_or_uniform(src)
        out = cls._rewrite_atomic_uint(out)
        out = cls._rewrite_buffers(out)
        return out

    @staticmethod
    def _rewrite_buffers(src: str) -> str:
        def repl(m: re.Match) -> str:
            qualifier = m.group('qualifier') or ''     # either "coherent " or ""
            dtype     = m.group('type')                # e.g. "uint"
            name      = m.group('name')                # e.g. "test123"
            size      = m.group('size')                # e.g. "NUM_ELEMS"
            binding   = m.group('binding')             # e.g. "0" or None

            # build layout(...) with std430 and optional binding
            layout_parts = ['std430']
            if binding:
                layout_parts.append(f"binding = {binding}")
            layout = f"layout({', '.join(layout_parts)})"

            # e.g. 
            #   buffer test123Buffer { uint test123[NUM_ELEMS]; };
            # with or without "coherent " immediately before "buffer".
            logger.debug("Rewriting buffer %s", name)
            return (
                f"{layout} "
                f"{qualifier}buffer {name}Buffer {{ {dtype} {name}[{size}]; }};"
            )

        return BUFFER_PATTERN.sub(repl, src)

    @staticmethod
    def _rewrite_shared_or_uniform(src: str) -> str:
        def repl(m: re.Match) -> str:
            su    = m.group('su')      # "shared" or "uniform"
            dtype = m.group('type')    # e.g. "uint"
            name  = m.group('name')    # e.g. "test3"
            array = m.group('array') or ''  # either "[]" or "", for shared arrays

            if su == 'shared':
                logger.debug("Rewriting shared %s", name)
                # e.g. "shared uint test3;"  or "shared uint test1[];"
                return f"shared {dtype} {name}{array};"
            else:
                logger.debug("Rewriting uniform %s", name)
                # e.g. "uniform uint test2;"
                return f"uniform {dtype} {name};"

        return SHARED_OR_UNIFORM_PATTERN.sub(repl, src)

    @staticmethod
    def _rewrite_atomic_uint(src: str) -> str:
        def repl(m: re.Match) -> str:
            name    = m.group('name')     # e.g. "test2131"
            binding = m.group('binding')  # e.g. "0"
            offset  = m.group('offset')   # e.g. "5"

            # produce: layout(binding = 0, offset = 5) uniform atomic_uint test2131;
            logger.debug("Rewriting atomic counter %s", name)
            return (
                f"layout(binding = {binding}, offset = {offset}) "
                f"uniform atomic_uint {name};"
            )

        return ATOMIC_UINT_PATTERN.sub(repl, src)
