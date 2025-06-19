import logging
logger = logging.getLogger(__name__)

import re
from typing import Callable
from tlang.attribute import Attribute
from tlang.function_manager import FunctionList
from tlang.shader_stages import ShaderStage


DECL_RE = re.compile(r'''
    ^\s*
    (?P<qual>uniform|shared|in|out)            # qualifier we trust
    \s+
    (?P<type>[A-Za-z_]\w*)\s+                  # GLSL type
    (?P<name>[A-Za-z_]\w*)                     # identifier
    (?:\s*:\s*LOC(?P<loc>\d+))?               # optional : LOC#
    \s*
    (?://(?P<comment>.*))?                    # optional // comment
    \s*$
''', re.VERBOSE)


class AttributeHandlers:
    @staticmethod
    def unroll(attr: Attribute, **kwargs) -> str: return '#pragma unroll'
    @staticmethod
    def flatten(attr: Attribute, **kwargs) -> str: return '#pragma flatten'
    @staticmethod
    def branch(attr: Attribute, **kwargs) -> str: return '#pragma branch'
    @staticmethod
    def loop(attr: Attribute, **kwargs) -> str: return '#pragma loop'
    @staticmethod
    def fastopt(attr: Attribute, **kwargs) -> str: return '#pragma optimize(on)'
    @staticmethod
    def noopt(attr: Attribute, **kwargs) -> str: return '#pragma optimize(off)'
    
    @staticmethod
    def program(attr: Attribute, **kwargs) -> str:
        # type checks
        if not isinstance(glob_attachments := kwargs.get('glob_attachments'), list):
            raise TypeError("Expected 'glob_attachments' to be a list")
        
        # add attribute to attachments
        glob_attachments.append(attr)
        return f"//<<PROGRAM DEC '{attr.name}'>>//\n"

    @staticmethod
    def shader(attr: Attribute, **kwargs) -> str:
        # type checks
        if not isinstance(funcs := kwargs.get('funcs'), FunctionList):
            raise TypeError("Expected 'funcs' to be a FunctionList")
        if not isinstance(index := kwargs.get('index'), int):
            raise TypeError("Expected 'index' to be an int")

        # set function stage
        if fn := funcs.find_next(index): 
            if not fn.stage: fn.stage = ShaderStage.from_token(attr.args[0])
            else: raise ValueError('Function already has a stage set.')
        return f"//<<SHADER STAGE DECL'>>//\n"
    
    @staticmethod
    def numthreads(attr: Attribute, **kwargs) -> str:
        # type checks
        if not isinstance(funcs := kwargs.get('funcs'), FunctionList):
            raise TypeError("Expected 'funcs' to be a FunctionList")
        if not isinstance(index := kwargs.get('index'), int):
            raise TypeError("Expected 'index' to be an int")

        # find function to attach to
        if fn := funcs.find_next(index):
            if attr.args:
                numthreads = [1, 1, 1]
                for i in range(min(len(attr.args), 3)):
                    numthreads[i] = int(attr.args[i])
            else:
                numthreads = (
                    int(attr.kwargs.get('x', 1)),
                    int(attr.kwargs.get('y', 1)),
                    int(attr.kwargs.get('z', 1)),
                )
            
            # add layout str to config
            fn.config.append(
                f'layout(local_size_x={numthreads[0]}, local_size_y={numthreads[1]}, local_size_z={numthreads[2]}) in;'
            )
        return f"//<<ATTR '{attr.name}'>>//\n"
    
    @staticmethod
    def dependency(attr: Attribute, **kwargs) -> str:
        # type checks
        if not isinstance(glob_attachments := kwargs.get('glob_attachments'), list):
            raise TypeError("Expected 'glob_attachments' to be a list")
        
        # add attribute to attachments
        glob_attachments.append(attr)
        return '\n'.join([f"{{% include '{sn}' %}}" for sn in attr.args])
    
    @staticmethod
    def extension(attr: Attribute, **kwargs) -> str:
        # type checks
        if not isinstance(glob_attachments := kwargs.get('glob_attachments'), list):
            raise TypeError("Expected 'glob_attachments' to be a list")
        
        # add attribute to attachments
        glob_attachments.append(attr)
        return f"//<<EXTENSION '{attr.name}', ARGS: {attr.args}>>//\n"
    
    @staticmethod
    def resourceblock(attr: Attribute, **kwargs) -> str:
        # type checks
        if not isinstance(funcs := kwargs.get('funcs'), FunctionList):
            raise TypeError("Expected 'funcs' to be a FunctionList")
        if not isinstance(index := kwargs.get('index'), int):
            raise TypeError("Expected 'index' to be an int")

        # find a function to attach to
        if fn := funcs.find_next(index):
            # add to config for each resource attribute
            for item in attr.args:
                m = DECL_RE.match(item)
                if not m: raise ValueError(f"Un-recognised resource declaratopm: {item!r}")
                
                q, loc = m['qual'], m['loc'], 
                layout = f"layout(location = {loc}) " if loc and q in ('in', 'out') else ''
                line   = f"{layout}{q} {m['type']} {m['name']};"
                if (cmt := m['comment']): line += f" //{cmt.strip()}"
                fn.config.append(line)
        return f"//<<RESOURCE BLOCK DECL>>//\n"
    
    @staticmethod
    def passthrough(attr: Attribute, **kwargs) -> str:
        # type checks
        if not isinstance(funcs := kwargs.get('funcs'), FunctionList):
            raise TypeError("Expected 'funcs' to be a FunctionList")
        if not isinstance(index := kwargs.get('index'), int):
            raise TypeError("Expected 'index' to be an int")
        
        if fn := funcs.find_next(index): fn.attrs.append(attr)
        return f"//<<{attr.name}>>//\n"

GLOB_CTX_ATTR_MAP: dict[str, Callable[..., str]] = {
    'shader':         AttributeHandlers.shader,
    'numthreads':     AttributeHandlers.numthreads,
    'program':        AttributeHandlers.program,
    'include':        AttributeHandlers.dependency,
    'extend':         AttributeHandlers.extension,
    'extend!':        AttributeHandlers.extension,
    'require':        AttributeHandlers.extension,
    'resourceblock':  AttributeHandlers.resourceblock,

    'frag':           AttributeHandlers.passthrough,
    'geom':           AttributeHandlers.passthrough,
    'tesc':           AttributeHandlers.passthrough,
    'tese':           AttributeHandlers.passthrough,

    'vertices':                   AttributeHandlers.passthrough,
    'early_fragment_tests':       AttributeHandlers.passthrough,
    'points':                     AttributeHandlers.passthrough,
    'lines':                      AttributeHandlers.passthrough,
    'triangles':                  AttributeHandlers.passthrough,
    'triangles_adjacency':        AttributeHandlers.passthrough,
    'line_strip':                 AttributeHandlers.passthrough,
    'triangle_strip':             AttributeHandlers.passthrough,
    'max_verts':                  AttributeHandlers.passthrough,
    'stream':                     AttributeHandlers.passthrough,
    'quads':                      AttributeHandlers.passthrough,
    'isolines':                   AttributeHandlers.passthrough,
    'equal_spacing':              AttributeHandlers.passthrough,
    'fractional_even_spacing':    AttributeHandlers.passthrough,
    'fractional_odd_spacing':     AttributeHandlers.passthrough,
    'cw':                         AttributeHandlers.passthrough,
    'ccw':                        AttributeHandlers.passthrough,
    'point_mode':                 AttributeHandlers.passthrough,
}

FUNC_CTX_ATTR_MAP: dict[str, Callable[..., str]] = {
    'unroll': AttributeHandlers.unroll,
    'flatten': AttributeHandlers.flatten,
    'branch' : AttributeHandlers.branch,
    'loop'   : AttributeHandlers.loop,
    'fastopt': AttributeHandlers.fastopt,
    'noopt'  : AttributeHandlers.noopt,
}