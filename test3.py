from shader_manager import ShaderManager
import moderngl as mgl

ctx = mgl.create_standalone_context()
sm = ShaderManager(ctx, '460 core', 'shaders')