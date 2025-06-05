import re
import logging

logger = logging.getLogger(__name__)

# matches:
#   OPTIONAL[coherent] buffer<Type> Name[Size] : @binding(N) ;
BUFFER_PATTERN = re.compile(
    r'(?P<qualifier>coherent\s*)?buffer<(?P<type>\w+)>\s+'  
    r'(?P<name>\w+)\[(?P<size>[\w\d_]+)\]\s*'
    r'(?::\s*@binding\((?P<binding>\d+)\))?;'
)

# matches either:
# 1) shared<Type> Name[...] : @binding(N) ;
# 2) uniform<Type> Name[...] : @binding(N) ;
# or
# 3) atomic_uint Name : @binding(N) @offset(M) ;
UNIFORM_PATTERN = re.compile(
    r'''
    # shared or uniform declarations
    (?P<qualifier>shared|uniform)\s*<(?P<type>\w+)>\s+(?P<name>\w+)(\[\])?
        (?:\s*:\s*@binding\((?P<binding>\d+)\))?
        \s*;
    |
    # atomic_uint declarations with optional @binding and/or @offset
    (?P<atomic>atomic_uint)\s+(?P<aname>\w+)\s*:?\s*
    (?:(?=.*@binding\((?P<abinding>\d+)\)))?
    (?:(?=.*@offset\((?P<aoffset>\d+)\)))?
    (?:(?:@binding\(\d+\)|@offset\(\d+\))\s*)+
    ;
    ''',
    re.VERBOSE
)

class MemoryManager:
    @classmethod
    def refactor(cls, src: str) -> str:
        logger.debug("Refactoring memory declarations")
        return cls._rewrite_buffers(
            cls._rewrite_uniforms_and_shared(src)
        )

    @staticmethod
    def _rewrite_buffers(src: str) -> str:
        def repl(m: re.Match) -> str:
            qualifier = m.group('qualifier') or ''
            dtype = m.group('type')
            name = m.group('name')
            size = m.group('size')
            binding = m.group('binding')

            # build the layout
            layout_clauses = ['std430']
            if binding: layout_clauses.append(f"binding = {binding}")
            if qualifier.strip(): layout_clauses.append(qualifier.strip())
            layout = f"layout({', '.join(layout_clauses)})"

            logger.debug("Rewriting buffer %s", name)
            return f"{layout} buffer {name}Buffer {{ {dtype} {name}[{size}]; }};"

        return BUFFER_PATTERN.sub(repl, src)

    @staticmethod
    def _rewrite_uniforms_and_shared(src: str) -> str:
        def repl(m: re.Match) -> str:
            # atomic_uint
            if m.group('atomic'):
                name = m.group('aname')
                binding = m.group('abinding')
                offset = m.group('aoffset')

                layout_parts = []
                if binding:
                    layout_parts.append(f"binding = {binding}")
                if offset:
                    layout_parts.append(f"offset = {offset}")
                layout = f"layout({', '.join(layout_parts)})" if layout_parts else ''
                logger.debug("Rewriting atomic counter %s", name)
                return f"{layout} uniform atomic_uint {name};"

            # shared<type> or uniform<type>
            qualifier = m.group('qualifier')
            dtype = m.group('type')
            name = m.group('name')
            is_array = m.group(4)   # matches '[]'
            binding = m.group('binding')

            if qualifier == 'uniform':
                logger.debug("Rewriting uniform %s", name)
                return f"uniform {dtype} {name};"
            elif qualifier == 'shared':
                logger.debug("Rewriting shared %s", name)
                if is_array: return f"shared {dtype} {name}[];"
                else: return f"shared {dtype} {name};"

            # default if no changes needed
            return m.group(0)

        return UNIFORM_PATTERN.sub(repl, src)