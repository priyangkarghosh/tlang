# Design: declared buffer IO, and the authoring-friction fixes

Settles the five design questions in pbd's
`docs/superpowers/specs/2026-09-15-tlang-foundations.md`, and records the
intended-vs-bug calls for T14 and T15 before any semantics change.

---

## Part A

### A.1 — T1 / T2: isolate and attribute, don't go lazy

**Decision: eager compilation stays.** Lazy per-`get_shader()` compilation would
break the contract the skill documents and every pbd fixture relies on — "a
successful `ShaderManager(...)` with `strict=True` means the whole tree compiled".
Moving compilation to first use moves every error to a random later call site,
which is T2's complaint generalised rather than fixed.

Instead, the failure is isolated and attributed:

* Every module is built inside its own try/except. One module's failure never
  aborts the others, so the tree always finishes building.
* Failures are collected per module and raised **once** at the end, listing every
  failed module and its error. Aborting on file 3 of 13 hides the other ten.
* `Shader.ok` / `Shader.failures` expose per-module status.
* `ShaderManager.get_shader(name)` returns `None` for a module that did not fully
  build, so the obvious check — `assert sm.get_shader(n) is not None` — is also
  the correct one. `get_shader(name, allow_failed=True)` retrieves it anyway, and
  `ShaderManager.failures` holds the per-module errors for diagnosis.

### A.2 — T14: **a parser bug, not a comment bug**

Root cause found and measured (`ATTR_PATTERN`, `attribute_manager.py`):

```
(?P<name>\w+!?) (?: \( (?P<args> (?: [^()] | (?R) )* ) \) )?
```

`(?:[^()]|(?R))*` is ambiguous: `(?R)` re-enters the whole attribute pattern,
whose `\w+!?` can consume 1..k characters of any word that `[^()]` could also
consume one character of. Every word of length k therefore has ~2^(k-1) parses.
While the overall match succeeds this costs nothing, because the first parse
found is accepted. As soon as the argument text contains a parenthesis that
`(?R)` cannot match — **any `(` or `)` not immediately preceded by an identifier**
— the match fails and the engine enumerates that entire exponential space.

Measured on this machine (`regex.fullmatch`, `timeout=10`):

| ordinary characters before a lone `(` | time |
|---|---|
| 6 | 0.29 ms |
| 10 | 13 ms |
| 14 | 631 ms |
| 16 | 5.3 s |
| 18 | > 10 s |

~5x per added character. **This is why a minimal repro did not reproduce it**:
`resourceblock(foo bar (x))` takes 0.37 ms and looks fine. The real comment —
`// FU-3: the substep length (Sec4.3 Eq. 10), used for the Eq. 10 clamp` — has
~40 characters ahead of its `(`, i.e. 2^40 units of work. Not a hang: an
exponential.

The trigger is **not** "a comment", and not the comment's content in any
mysterious sense. It is *a parenthesis inside an attribute argument list that is
not a function-call paren*, plus enough preceding text. Prose is simply where
parentheses naturally appear. A `//` comment is also parsed as argument text at
all — comments are not stripped inside attribute argument lists — which is the
second, independent defect: a comma inside such a comment mis-splits the
argument list too.

**Verdict: bug, on both counts.** Three fixes:

1. **Make the pattern linear.** Replace the ambiguous recursion with a proper
   balanced-paren run matched possessively:
   `(?P<args>(?:[^()]++|\((?&args)\))*+)`. Verified: the exact T14 body goes from
   unbounded to 0.03 ms, and it now *matches* — a balanced parenthesis in a
   comment becomes legal rather than fatal. An unbalanced one fails in 0.01 ms
   with a clear "malformed attribute" error instead of hanging.
2. **Mask comments for structural decisions** in attribute splitting and argument
   parsing, so a comment's parens and commas cannot steer the parse. The raw text
   is still what `[resourceblock(...)]` emits, so comments survive into the GLSL.
3. **Bound every preprocessor regex that scans user text.** Pass `timeout=` and
   convert the resulting `TimeoutError` into a `TlangSyntaxError` naming the
   attribute and its location. A parser that can livelock will otherwise livelock
   in CI where nobody can attach a debugger.

### A.3 — T15: `[export()]` is **intended**; the silence is not

**Decision: the requirement is deliberate — do not change the semantics.**

tlang's whole model is that each entry point gets the smallest translation unit
that will link: `[link('helper')]` exists precisely to pull one helper into one
entry point, and `[export()]` to publish a helper module-wide. Emitting every
module-scope function into every entry point would delete that distinction, grow
every artifact, and keep alive the SSBO blocks those helpers touch — which is the
T13 over-approximation, made mandatory.

What is not intended is a helper that is defined in the file, called by a kernel,
and emitted nowhere, failing as a raw GLSL `undefined variable` at a generated
line number. So the fix is a **diagnostic**: when an entry point's emitted source
calls a name that exists as a module-scope function in the same file but was not
emitted into that translation unit, raise a tlang error naming the function, the
caller, and the attribute it needs.

This reuses the call-site scanner that Part B's dead-function elimination needs
anyway. It also catches T10's symptom (a backwards `[link]` leaves the helper
unemitted) for free.

---

## Part B — declared buffer IO

### B.0 — the prerequisite: dead-function elimination (retires T13)

`kernel.bindings` is currently "blocks reachable from anything `[include]`d",
not "blocks this kernel touches", because every `[export()]`ed helper's body is
emitted into every entry point of the including file whether called or not, and
block DCE then keeps the blocks those uncalled bodies reference.

Fix it where the facts are: add **dead-function elimination** to the generated
GLSL, immediately before the existing dead-block pass in `Shader._build`. Parse
top-level function definitions out of the emitted unit, walk the call graph from
`main`, and blank the bodies of everything unreachable. The existing
`remove_dead_blocks` then sees only reachable text and strips the rest.

This is the same textual approximation class as the block DCE it feeds, and biased
the same way (keep when uncertain). Its failure mode is safe: wrongly removing a
needed function is a loud compile error, never silent wrongness.

Consequence: `kernel.bindings` becomes exactly "blocks reachable from `main()`" —
T13's ask, and the precision every check below depends on.

### B.1-B.4 — the `[buffers(...)]` attribute: **DEFERRED, and here is why**

The five design questions were answered in favour of a new `[buffers(Name, ..., dir=)]`
attribute. A red-team of that answer, plus two measurements on this machine, overturned
it. Recorded in full because the spec asked for the attribute by name and this is a
deliberate departure from it.

**The attribute retires nothing that B.0 and B.5 do not already retire:**

| issue | retired by | needs the attribute? |
|---|---|---|
| T13 — `kernel.bindings` over-broad | B.0 (dead-function elimination) | no |
| T11, T12 case 1 — silent cross-wiring | B.5.1 (per-kernel sticky rebind) | no |
| T12 case 2 — a forgotten bind | B.5.2 (raise on unbound required block) | no |
| T7 — binding a stripped block raises | B.5.3 (`bind(**superset)` ignores extras) | no |
| T16 — stale bind after a GLSL edit | B.5.3 (same) | no |

**And each thing it adds carries a measured cost:**

1. **Pinning costs a hard block slot.** Measured: 17 SSBO blocks declared with 1
   referenced fails to link — `error C5058: no buffers available for bindable storage
   buffer` — against `GL_MAX_COMPUTE_SHADER_STORAGE_BLOCKS = 16`. A pin is not a free
   annotation, it is budget. It is also unnecessary once `bind()` is lenient about names
   the artifact does not declare.

2. **`dir=` does not enforce what B.2 claimed.** Measured: `atomicAdd` through a
   `readonly` SSBO block **compiles cleanly**; only plain assignment is rejected. pbd's
   buffer writers are overwhelmingly atomics (`reserveSlot`, `addConstraint`, `atomicMin`,
   `pairOverflowCount`), so `dir='in'` would miss precisely the code that matters. B.2's
   "enforcement is then the driver's" was wrong as written.

   It is also not independent of B.0 as B.2 implied: a write through a `readonly` block
   inside a helper that DFE kept but `main` never calls is still a compile error, so B.2's
   correctness depends on B.0's keep-when-uncertain bias never being wrong.

3. **Completeness checking is one-directional.** Erroring on "live but undeclared" leaves
   the reverse case — a declaration naming a block the GLSL no longer touches — pinning a
   dead buffer that U1 then demands a bind for, with nothing telling the author it is dead.
   That is **T16 re-introduced in a new place**. Adding the reverse check makes declared
   identical to inferred, i.e. a checked restatement of what tlang already computes.

**Decision: ship B.0 and B.5; defer B.1-B.4 entirely.** No language change, so no other
tlang consumer inherits anything. Revisit only if a real write-to-an-input bug is ever
observed — and if so, implement it as per-artifact qualifier injection through
`_patch_bindings` (which already runs per artifact) rather than as a language attribute.

Answers to the spec's five questions, as they now stand:

1. **Syntax** — no new syntax. Had one been added, it would have had to be a separate
   attribute rather than an extension of `[uses]`: `[uses]` resolves against the interface
   table, and pbd's 17 buffer sites are raw GLSL `layout(std430) buffer` blocks that are
   not in that table at all, so extending `[uses]` would have forced an atomic migration.
2. **Is direction real or documentation?** — neither; there is no direction. Had it been
   added it would have been *partial* enforcement, and calling it real would have misled.
3. **`[include]`d helpers that touch buffers** — inferred, but now inferred *precisely*:
   B.0 makes the inferred set exactly "blocks reachable from `main()`", which is what T13
   asked for. No declaration needed.
4. **Migration path** — nothing to migrate. Existing shaders are untouched; only
   `kernel.bindings` narrows, and `Kernel.bind()` absorbs that.
5. **Does Python bind automatically?** — yes, and this was always the load-bearing half.
   See B.5.

### B.0a — DFE validated against pbd's real artifacts, before implementing

A prototype of the pass was run against pbd's 13 compiled shaders and its prediction
compared to `LEGITIMATELY_UNUSED_BY_MAIN`, the exclusion table pbd derived by hand:

**16 of 18 kernels matched exactly.** Every block pbd had reasoned was
declared-but-unreachable is exactly the set the pass drops. Artifacts shrink
substantially — `update` 10 blocks -> 6, `init` 6 -> 2, `setPairCount` 6 -> 2.

The two mismatches are both real, and both say the **hand-maintained table is wrong**:
for `solvePtcPtc` and `solvePtcPtcColoured` the pass also drops `ConstraintOffsets`,
which pbd's table lists as required and pbd's Python binds. Checked against the
source: `addConstraint` touches only `constraintCounts` and `constraintSlots`; only
`reserveSlot` touches `constraintOffsets`, and neither kernel calls `reserveSlot`.
pbd's own comment ("addConstraint -> counts/Offsets/Slots") is mistaken. The binding
is harmless today, but it is a hand-derived fact that drifted from the code — which is
the argument for this whole change in miniature.

**Migration consequence, and it is a breaking one:** once those two blocks are dropped,
pbd's existing `bind_ssbos(ConstraintOffsets=...)` on those two kernels raises
`TlangBindingError` — the T7 shape. This is why B.0 must land *together with*
`Kernel.bind(**available)` (B.5 item 3), which ignores names the artifact does not
declare, and why pbd's bind sites must move to `bind()` in the same step rather than
after it.

### B.0b — pinning is driver-dependent; do not rely on it

"Declared means present" was checked on this machine (RTX 3090, NVIDIA 616.64, GL 4.3):
a declared-but-unreferenced SSBO block **is** kept, reflected, assigned the binding
tlang pinned, and bindable — so `verify_link` does not trip. But GL does not *require*
a driver to report an unreferenced block as an active resource, so this must not be
load-bearing.

Therefore pinning is best-effort, and the guarantee is stated on the Python side
instead: **binding a declared name is always legal — either the block is really there,
or the bind is a no-op.** That retires T7 and T16 by construction on every driver,
rather than on the ones that happen to keep the block.

### B.6 — Predictions

Stated before implementing, so they can be checked honestly afterwards:

* pbd's `tests/test_bindings.py` (~300 lines, two tests, a 19-entry `KERNEL_MODULE`
  table and a 15-entry `LEGITIMATELY_UNUSED_BY_MAIN` table) becomes redundant and is
  deleted. tlang's own dispatch check replaces it for every consumer.
* Every entry in `LEGITIMATELY_UNUSED_BY_MAIN` disappears from `kernel.bindings`
  after B.0, so the exclusion table would be empty even if the test were kept — that
  is the check that B.0 actually worked.
* `physics/colouring.py::INCLUDE_ONLY_BLOCKS` and `bind_declared` are deleted.
* pbd's suite holds at 90 passed, 5 xfailed, 0 failed.
* tlang's suite grows from 214; no existing test changes meaning.
