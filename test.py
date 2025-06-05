import moderngl as mgl

from glsl_preprocessor.processor import Processor

ctx = mgl.create_standalone_context()
# processor = GLSLPreprocessor(ctx, 'shaders', {
#     'NUM_ELEMS': 7,
# })

processor2 = Processor('shaders', {
    'NUM_ELEMS': 7,
})