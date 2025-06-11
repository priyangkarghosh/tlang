import moderngl as mgl

from glsl_preprocessor.processor import Processor
from glsl_preprocessor.shader_manager import ShaderManager

ctx = mgl.create_standalone_context()
# processor = GLSLPreprocessor(ctx, 'shaders', {
#     'NUM_ELEMS': 7,
# })

processor2 = Processor('shaders', {
    'NUM_ELEMS': 7,
})
ShaderManager(ctx, '460 core', processor2._processed_sources, processor2._entries, processor2._programs)
