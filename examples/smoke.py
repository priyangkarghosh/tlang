"""Minimal end-to-end check: builds examples/shaders and reports what came out.

Run from the repo root:  python examples/smoke.py
"""
import logging
import sys

logging.basicConfig(level=logging.WARNING, stream=sys.stdout)

import moderngl as mgl

from tlang import ShaderManager

ctx = mgl.create_context(require=460, standalone=True)
sm = ShaderManager(ctx=ctx, version='460 core', dir='shaders', constants={'BLOCK_SIZE': 256})

shader = sm.get_shader('demo')
assert shader is not None, "demo shader was not built"
print('kernels: ', sorted(shader.kernels))
print('programs:', sorted(shader.programs))

assert 'cs_go' in shader.kernels, "compute kernel missing -> [include]/[export] regression"
assert 'default' in shader.programs, "program missing -> linking regression"
print('OK')
