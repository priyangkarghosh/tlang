"""End-to-end feature check over examples/shaders.

Asserts the behaviour each feature is supposed to have, not merely that the tree
builds -- a shader that compiles while binding the wrong buffer is the failure
mode this exists to catch.

Run from the repo root:  python examples/smoke.py
"""
import logging
import struct
import sys

logging.basicConfig(level=logging.WARNING, stream=sys.stdout)

import moderngl as mgl

from tlang import BufferPool, ShaderManager

ctx = mgl.create_context(require=460, standalone=True)
sm = ShaderManager(ctx=ctx, version='460 core', dir='shaders',
                   constants={'BLOCK_SIZE': 256, 'BLOCK': 64})

checks = []


def check(name, condition, detail=''):
    checks.append((name, bool(condition), detail))


# --- module resolution, [include]/[export] ---------------------------------

demo = sm.get_shader('demo')
compute = sm.get_shader('compute')
check('every module built', demo is not None and compute is not None)
check('failed modules report as None', sm.get_shader('nope') is None)
check('no build failures', sm.failures == {}, str(sm.failures))

# --- raster: [program], [varyings], [uses], [uniforms] ---------------------

check('program linked', 'default' in demo.programs)
check('varyings interface registered', 'VertexOut' in demo.interfaces)
check('uniforms interface registered', 'Frame' in demo.interfaces)

# --- compute: declaration forms all bind by the name written ---------------

tally = compute.get_kernel('tally')
accumulate = compute.get_kernel('accumulate')
summarise = compute.get_kernel('summarise')

check('shorthand binds by member name', 'counts' in tally.bindings)
check('struct form binds by struct name', 'Totals' in summarise.bindings)
check('raw GLSL block binds by its name', 'Scratch' in accumulate.bindings)

# --- dead-code elimination scopes bindings to what main() reaches ----------

check('DCE: tally declares only what it touches',
      set(tally.bindings) == {'counts'}, sorted(tally.bindings))
check('DCE: summarise keeps only Totals',
      set(summarise.bindings) == {'Totals'}, sorted(summarise.bindings))
check('[link] pulls a helper in: accumulate reaches Scratch',
      set(accumulate.bindings) == {'counts', 'Scratch'}, sorted(accumulate.bindings))

# --- local_size read off the linked program, not the attribute text --------

check('[extern] int drives numthreads', tally.local_size == (64, 1, 1), str(tally.local_size))
check('[extern] default applied without being supplied',
      'GAIN' in compute.externs, sorted(getattr(compute, 'externs', [])))
check('local_size differs per kernel', summarise.local_size == (1, 1, 1))

# --- the pool is the name -> buffer map; bind() takes no arguments ---------

N = 128
pool = BufferPool(ctx)
pool.persistent_buffer('counts', size=N * 4)
pool.persistent_buffer('Scratch', size=N * 4)
pool.persistent_buffer('Totals', size=8)
sm.buffer_source = pool

tally.set_uniforms(elementCount=N)
tally.bind()
tally.dispatch_for(N)

accumulate.set_uniforms(elementCount=N)
accumulate.bind()
accumulate.dispatch_for(N)

summarise.bind()
summarise.dispatch_for(1)
ctx.finish()

counts = struct.unpack(f'{N}I', pool['counts'].read(N * 4))
scratch = struct.unpack(f'{N}I', pool['Scratch'].read(N * 4))
hits, misses = struct.unpack('2I', pool['Totals'].read(8))

# tally adds 1 twice per element; accumulate doubles it via math.mul
check('argument-free bind + dispatch_for produced the right values',
      all(c == 2 for c in counts) and all(s == 4 for s in scratch),
      f'counts[0]={counts[0]} scratch[0]={scratch[0]}')
check('dispatch_for covered every element, none beyond',
      len(set(counts)) == 1 and len(set(scratch)) == 1)
check('[include]d [export]ed helpers ran (add/mul from math.tlang)',
      counts[0] == 2 and scratch[0] == 4)
check('struct-form block written through its members', (hits, misses) == (1, 2),
      f'{hits},{misses}')

# --- a required block left unbound is an error, not silent garbage ---------

from tlang.errors import TlangBindingError

lone = ShaderManager(ctx=ctx, version='460 core', dir='shaders',
                     constants={'BLOCK_SIZE': 256, 'BLOCK': 64}).get_shader('compute').get_kernel('tally')
try:
    lone.set_uniforms(elementCount=1)
    lone.dispatch_for(1)
    check('unbound required block raises', False, 'no error raised')
except TlangBindingError as exc:
    check('unbound required block raises', 'counts' in str(exc), str(exc)[:60])

# --- source text is not retained by default --------------------------------

check('generated GLSL dropped by default',
      sum(len(v) for v in compute.sources.values()) == 0,
      f'{sum(len(v) for v in compute.sources.values())} bytes retained')

kept = ShaderManager(ctx=ctx, version='460 core', dir='shaders',
                     constants={'BLOCK_SIZE': 256, 'BLOCK': 64}, keep_sources=True).get_shader('compute')
check('keep_sources=True retains it', len(kept.get_source('tally')) > 0)

# --- report ----------------------------------------------------------------

width = max(len(n) for n, _, _ in checks)
failed = 0
for name, ok, detail in checks:
    if not ok:
        failed += 1
    print(f"{'ok  ' if ok else 'FAIL'}  {name:{width}}  {detail if not ok else ''}".rstrip())

print(f"\n{len(checks) - failed}/{len(checks)} checks passed")
sys.exit(1 if failed else 0)
