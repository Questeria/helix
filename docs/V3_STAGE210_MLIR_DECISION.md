# V3.0 Stage 210 — MLIR Dependency & Dialect-Strategy Decision

> **STATUS: RATIFIED — Stage 210 decision record (2026-05-20).**
> Produced as Phase-E scoping research, then reviewed and finalized as
> the approved Stage 210 decision. An independent architecture review
> verified the op-mapping against the real `tir.py` / `tile_ir.py` and
> confirmed all three recommendations sound; its corrections are
> applied throughout and consolidated in Section 6. This is the
> approved decision that gates Phase E (Stages 211-216). No compiler
> code was written or modified to produce it.

---

## 0. Scope & charter

`V3_PLAN.md` defines Stage 210 as:

> "MLIR dependency + dialect-strategy decision. Evaluate MLIR
> Python-binding availability; decide upstream dialects vs. a custom
> Helix dialect vs. hybrid; write the decision record."

This draft answers the three questions Stage 210 must close:

1. **How does a Python-hosted compiler obtain and call MLIR?**
   (Python-binding availability, Windows constraint, pip path.)
2. **How should Helix IR be represented in MLIR?**
   (upstream-only vs. custom dialect vs. hybrid.)
3. **How does the V3_PLAN mock-path requirement constrain the design?**

The two IR layers being replaced are:

- `helixc/ir/tir.py` (~600 lines) — the value-semantic **Tensor IR**:
  whole-tensor ops, named axes, layout-as-type, SSA-with-block-params
  (no phi). 98 `OpKind` members.
- `helixc/ir/tile_ir.py` (~520 lines) — the mid-level **Tile IR**:
  explicit memory spaces (HBM/SMEM/REG/TMEM), async memory (TMA,
  barriers), explicit tile sizes, warp/CTA scheduling. 34
  `TileOpKind` members + the Stage 117-120 adjoint table.

---

## 1. MLIR Python-binding availability

### 1.1 The core fact

MLIR's Python bindings are a **first-class, supported** part of upstream
LLVM/MLIR, but **upstream LLVM does not publish an official `mlir`
package to PyPI.** The official documentation
(`mlir.llvm.org/docs/Bindings/Python/`) describes only the
build-from-source path: configure the LLVM/MLIR CMake build with
`-DMLIR_ENABLE_BINDINGS_PYTHON=ON`, install the deps in
`mlir/python/requirements.txt`, and add
`tools/mlir/python_packages/mlir_core/` to `PYTHONPATH` (or
`ninja install`). Upstream bindings now use **nanobind** (migrated
from pybind11). A non-EOL Python 3 and a virtualenv are required.

So a Python-hosted compiler that wants `import mlir` has **four**
realistic acquisition paths, of which only some are pip/conda-style.

### 1.2 The four acquisition options

| # | Path | Install command | Windows? | Maintenance | Notes |
|---|------|-----------------|----------|-------------|-------|
| A | **conda-forge** `mlir-python-bindings` | `conda install -c conda-forge mlir-python-bindings` | **Yes — `win-64` is a listed platform** | Active — feedstock updated **2026-01-08**; companion `mlir` package at v22.x updated **2026-02**; ~373K downloads | Cleanest pip-style story; requires a conda/mamba environment, not bare `pip`. |
| B | **`llvm/eudsl`** wheels | `pip install mlir-python-bindings -f https://llvm.github.io/eudsl` | Wheels target `{x86_64, aarch64} x {Linux, macOS, Windows}` minus Linux-aarch64 / Windows-ARM64 | **Alpha** quality, but it is now the *official LLVM-org* successor | This is the **successor to `makslevental/mlir-wheels`**, which was archived **2025-11-29** with "all work has migrated to llvm/eudsl". Bare `pip` works but needs the `-f` find-links index (not on PyPI proper). |
| C | **Build from source** | CMake `-DMLIR_ENABLE_BINDINGS_PYTHON=ON` + ninja | Yes, but historically fragile — multiple LLVM patches (D125122/D125134/D125284) were needed to make Windows + Debug builds work; nanobind/CMake/Python interplay is the usual failure point | N/A (we build it) | Multi-hour build, ~tens of GB of disk, a full MSVC + CMake + ninja toolchain. Gives exact version control and the ability to compile a custom C++ dialect. |
| D | **Reuse an MLIR a dependency already ships** | (transitive) | Depends on the host package | Varies | `torch-mlir`, Triton, IREE, etc. each vendor their own MLIR Python build. Helix has no such dependency today; not a viable primary path, listed for completeness. |

A non-binding alternative — `spcl/pymlir` — is a **pure-Python MLIR
parser/AST library**. It is *not* a binding to libMLIR: it cannot run
MLIR passes, verify ops against real dialect definitions, or lower to
LLVM. It is unsuitable as the Phase E engine but is noted because its
"define a dialect in pure Python" ergonomics are a useful reference
for the mock path (Section 3).

### 1.3 Windows reality check

The project runs on Windows (per the Kovostov environment).
Windows is the historically weakest MLIR-Python surface:

- conda-forge **explicitly lists `win-64`** — option A is the lowest-risk
  Windows path and is **actively maintained into 2026**.
- eudsl claims Windows-x86_64 wheels (ARM64 excluded) — option B is
  plausible on Windows but is self-described **alpha**.
- From-source on Windows is doable (the bot exists) but is the most
  failure-prone of the three; it should be a fallback, not the plan.

### 1.4 Recommendation for the dependency question

**Primary (resolved at ratification — see below): the eudsl
find-links wheels** (option B) — `pip install mlir-python-bindings -f
https://llvm.github.io/eudsl`. The Helix dev environment is bare
`pip` + venv on Windows (Python 3.13; no conda / mamba present), so
the bare-`pip`-installable path is the one Phase E targets. Its
self-described **alpha** status is a tracked risk (Section 3.4),
mitigated by Phase E being mock-path-first: a binding install that
fails or regresses only DEFERS the real path, it never blocks
development.

**Documented alternative:** the **conda-forge `mlir-python-bindings`**
package (option A) — the cleaner, `win-64`-confirmed, 2026-maintained
path — is the recommended choice should the project ever adopt a
conda/mamba environment.

**Explicitly rejected as the primary path:** build-from-source
(option C) — kept only as the last-resort path if both A and B fail on
a target machine, or if Phase E ever needs a *custom compiled C++
dialect* (see Section 2.4, which argues it should not).

**Hard pin discipline.** MLIR has no stable C++/Python API across LLVM
releases — IR, dialect, and pass APIs churn every major version. Phase
E must pin **one** LLVM major version (e.g. LLVM 22.x, the current
conda-forge line as of 2026-02) for the entire phase and treat a
version bump as its own tracked stage. The pin must be recorded in the
project's environment spec, not just in prose.

> **RESOLVED at Stage-210 ratification:** the Helix environment is
> bare `pip` + venv (Windows, Python 3.13; no conda / mamba present),
> so **option B (eudsl) is the primary dependency path** and its alpha
> status is a tracked risk. Stage 211 provisions the binding install;
> because Phase E is mock-path-first, a failed or absent install
> simply leaves the real MLIR path DEFERRED and does not block
> development.

---

## 2. Dialect strategy

### 2.1 The three options

(a) **Upstream-only** — represent all Helix IR using existing upstream
dialects: `func`, `arith`, `math`, `scf`/`cf`, `tensor`, `linalg`,
`vector`, `memref`, `gpu`, `nvgpu`/`nvvm`, `llvm`.

(b) **Custom Helix dialect** — define a `helix` dialect with ops that
mirror `tir.OpKind` / `TileOpKind` one-to-one, plus Helix-specific
types.

(c) **Hybrid** — a *small* custom `helix` dialect that holds **only**
the ops with no faithful upstream equivalent, and which **lowers** to
upstream dialects for everything else. Helix-specific ops are modelled
custom; standard tensor/scalar/GPU algebra reuses upstream.

### 2.2 Mapping the existing op sets onto upstream dialects

#### Tensor IR (`tir.py`, 98 `OpKind`s)

| `tir.OpKind` group | Best upstream home | Fit |
|--------------------|--------------------|-----|
| Scalar `CONST_*`, `ADD..MOD`, `BIT_*`, `SHL/SHR`, `NEG`, `ABS`, `CMP_*` | `arith` | Excellent — near 1:1. |
| `EXP/LOG/SQRT/RECIP/POW/TANH` | `math` | Excellent. |
| `RELU/GELU/SILU/SIGMOID` | `linalg` named ops / `math` + `arith` composite | Good — `linalg` has some; activations may decompose. |
| Elementwise tensor `ADD..MINIMUM`, `SELECT`, `WHERE` | `linalg.generic` / `linalg.map`, `arith.select` | Excellent. |
| `MATMUL`, `CONV1D/2D` | `linalg.matmul`, `linalg.conv_*` | Excellent — this is `linalg`'s core purpose. |
| `REDUCE_*` | `linalg.reduce` | Excellent. |
| `RESHAPE/TRANSPOSE/BROADCAST/SLICE/CONCAT` | `tensor.*` (`expand/collapse_shape`, `extract_slice`, `concat`) + `linalg.transpose` | Good. |
| `TENSOR_ZEROS/ONES/FULL/RAND` | `tensor.empty` + `linalg.fill` (rand needs a runtime call) | Good. |
| `CAST/BITCAST` | `arith` ext/trunc/bitcast, `tensor.cast` | Good. |
| `CALL/RETURN/BR/COND_BR` | `func`, `cf` | Excellent — block-params + no-phi is exactly MLIR's model. |
| `ALLOC_VAR/LOAD_VAR/STORE_VAR`, `ALLOC_ARRAY/LOAD_ELEM/STORE_ELEM` | `memref` | Good. |
| `THREAD_IDX`, `TILE_INDEX_LOAD/STORE` | `gpu` dialect | Good. |
| `PRINT`, `TRAP`, `TRACE_ENTRY/EXIT`, `FFI_CALL`, `STR_PTR/STR_BYTE` | lower to `func.call` of runtime symbols; or **`helix` dialect** | Marginal — modellable but Helix-semantic. |
| `QUANTIZE/DEQUANTIZE` | `quant` dialect (or `helix`) | Marginal — `quant` is narrow; Helix's fp8/mxfp4/nvfp4/ternary set is front-end-only today. |
| **`GRAD/JVP/VMAP`** (`transform.*`) | **no faithful upstream op** | **Poor — Helix-specific.** |
| **`QUOTE/SPLICE/MODIFY/REFLECT_HASH`** (`agi.*`) | **no upstream equivalent** | **Poor — Helix-specific, AGI metaprogramming.** |
| **`ARENA_PUSH/GET/SET/LEN/PUSH_PAIR/PUSH_TRIPLE`** | `memref` *approximates* a bump allocator but loses the atomic-pair/triple invariant | **Poor — Helix-specific bump allocator with atomicity guarantees.** |
| `RESULT_PACK/TAG/PAYLOAD` | `arith` shifts/masks *can* express the bit-twiddling | Marginal — works but loses the discriminated-union intent the `tir.py` comments deliberately encode. |

#### Tile IR (`tile_ir.py`, 34 `TileOpKind`s)

| `TileOpKind` group | Best upstream home | Fit |
|--------------------|--------------------|-----|
| `TILE_ZEROS/CONST`, `TILE_ADD/SUB/MUL`, `TILE_REDUCE`, `TILE_TRANSPOSE/RESHAPE` | `vector` dialect (+ `arith`) | Good — `vector` is MLIR's tile/SIMD layer. |
| `TILE_MATMUL` (accumulating) | `vector.contract` / `linalg` at the tile level | Good. |
| `TILE_LOAD/STORE_GLOBAL/SHARED` | `memref` + `gpu` (address spaces) | Good. |
| `TMA_LOAD/TMA_STORE`, `BARRIER_WAIT` | `nvgpu` (`nvgpu.tma.async.load`, mbarrier ops) | Good **for NVIDIA**; AMD/Metal/WebGPU async has weaker upstream coverage. |
| `THREAD_IDX`, `TILE_INDEX_LOAD/STORE_HBM` | `gpu` | Good. |
| `SCALAR_*`, `CALL`, `RETURN` | `arith`, `func` | Excellent. |
| **The adjoint table** (`TILE_OP_ADJOINTS`, Stage 117-120) | not an op set — it is a **transform** | Stays Helix-side as a pass; MLIR has no built-in tensor-core-aware reverse-mode AD. |

**Conclusion of the mapping:** roughly **80-85%** of both op sets has
an excellent or good upstream home — `linalg` + `vector` + `arith` +
`gpu` + `func` + `memref` cover the entire numerical/structural core.
The residual **~15-20%** is genuinely Helix-specific: the
compositional transforms (`GRAD/JVP/VMAP`), the AGI metaprogramming
ops (`QUOTE/SPLICE/MODIFY/REFLECT_HASH`), the arena allocator with
its atomic multi-push invariants, and — more weakly — the
Result/quantize encodings and the effectful runtime ops.

### 2.3 Evaluating the three options

**(a) Upstream-only — rejected.**
- Pro: zero custom-dialect maintenance; maximal reuse of upstream
  passes; `linalg`/`vector` are exactly the right abstractions for the
  80% numerical core.
- Con: the Helix-specific 15-20% has **no faithful representation**.
  Forcing `GRAD`/`QUOTE`/`ARENA_PUSH_PAIR` into `arith`+`memref`+`call`
  *erases the very invariants the current `tir.py` is carefully built
  around* — e.g. the Stage-36 comment on `ARENA_PUSH_PAIR` exists
  precisely so DCE/CSE/scheduler reordering cannot split the pair;
  modelled as two `memref.store`s an MLIR pass is free to split them.
  Upstream-only would either lose those guarantees or smuggle them
  back as fragile attribute conventions on generic ops — strictly
  worse than today.

**(b) Custom Helix dialect (full 1:1) — rejected.**
- Pro: a clean, total mapping; the migration is mechanical
  (`OpKind` -> `helix.op`).
- Con: throws away the **single biggest reason `V3_PLAN.md` gives for
  the migration** — "reusable upstream dialects (`linalg`, `vector`,
  `gpu`, `llvm`)" and their pass infrastructure. A full custom dialect
  would still need every pattern (matmul tiling, fusion, vectorization,
  lowering-to-LLVM) hand-written — i.e. it re-creates the home-grown
  tile-IR's central weakness inside MLIR. It also maximizes
  custom-C++/ODS surface, which collides with the
  no-build-from-source recommendation of Section 1.

**(c) Hybrid — recommended.** See Section 2.4.

### 2.4 Recommendation: HYBRID — a thin `helix` dialect over upstream

**Represent the numerical/structural core (~80-85%) directly in
upstream dialects** — `func`, `arith`, `math`, `cf`/`scf`, `tensor`,
`linalg`, `vector`, `memref`, `gpu`, `nvgpu`. **Define a small
`helix` dialect for only the ops with no faithful upstream home:**

- `helix.grad`, `helix.jvp`, `helix.vmap` — the compositional
  transforms. Kept as first-class ops so a Helix pass can pattern-match
  and materialize them (the current `transform.*` semantics) before
  lowering the result into `linalg`/`vector`.
- `helix.quote`, `helix.splice`, `helix.modify`, `helix.reflect_hash`
  — the AGI metaprogramming ops. These have no analogue anywhere in
  MLIR and are core to the project's purpose; they **must** be modelled
  explicitly.
- `helix.arena_push` / `arena_get` / `arena_set` / `arena_len` /
  `arena_push_pair` / `arena_push_triple` — the bump allocator. Custom
  ops let the dialect's verifier and op-trait system *enforce* the
  atomic-pair/triple invariant (e.g. mark them with appropriate
  memory-effect traits / make them un-splittable) instead of relying
  on comments.
- Likely also `helix.result_pack/tag/payload` and the Helix quantize
  ops, to preserve discriminated-union and quant intent — **flag for
  review**: these *can* be upstream `arith`/`quant`; the call is a
  fidelity-vs-reuse trade the orchestrator should make explicitly.
- Effectful runtime ops (`print`, `trap`, `trace`, `ffi_call`) can
  start as `func.call` of runtime symbols; promote to `helix` ops only
  if a pass needs to reason about them.

**The `helix` dialect lowers to upstream.** Every `helix` op gets a
lowering pattern: `helix.grad` expands (after differentiation) into
`linalg`/`vector` ops; arena ops lower to `memref`; etc. After the
`helix`-lowering pass the IR is pure upstream dialects and the standard
upstream `-to-llvm` / `gpu`-lowering pipelines carry it to the Phase-D
LLVM path and the four GPU backends. This is the "progressive
lowering" `V3_PLAN.md` Stage 214 calls for.

**Rationale:**
1. It captures **all** of the `V3_PLAN.md` motivation — upstream
   `linalg`/`vector`/`gpu` + their passes do the heavy lifting for the
   80% that is standard tensor algebra.
2. It loses **none** of the Helix-specific invariants — the 15-20%
   that has no upstream home gets explicit, verifiable ops.
3. It **minimizes** custom surface — the dialect is small (a dozen-ish
   ops), which keeps it within reach of the IRDL / Python-defined
   dialect path and avoids forcing a from-source C++ build.
4. It is **incremental and reversible** — the `helix`-to-upstream
   lowering can be built op-group by op-group; until an op's lowering
   exists it can fall back to the home-grown tile-IR path. This is
   exactly the additive/parity-gated discipline `V3_PLAN.md` mandates
   (Stages 212-215).

**Dialect-definition mechanism (sub-decision for Stage 211).** A
custom dialect can be defined three ways: full C++/ODS (needs a
from-source build), **IRDL** (MLIR's in-IR dialect-definition dialect —
loadable from Python, no C++ build), or pure-Python op wrappers over a
generically-registered dialect. Given the Section-1 recommendation to
**avoid building MLIR from source**, Stage 211 should prefer **IRDL or
the Python dialect-registration path** (`append_dialect_search_prefix`)
so the `helix` dialect needs no compiled C++ extension. This keeps the
whole Phase E installable from a conda/eudsl binary package. **If** a
`helix` op turns out to need a custom C++ trait that IRDL cannot
express, that is the *only* scenario that reopens the
build-from-source question — and it should be escalated, not assumed.

> **Open items for the orchestrator (Stage 210/211 boundary):**
> - Decide `result_pack`/`quantize`: custom `helix` ops vs. upstream
>   `arith`/`quant`.
> - Confirm IRDL can express the arena-atomicity traits; if not,
>   decide whether that one need justifies a from-source build or
>   whether the invariant can be held by a verification pass instead.
> - Decide whether the Tile-IR async ops target `nvgpu` only
>   (NVIDIA-faithful) or a `helix` async abstraction that fans out to
>   all four GPU backends — `nvgpu` has no AMD/Metal/WebGPU analogue,
>   so a Helix async op may be unavoidable for backend parity.

---

## 3. Migration risk & the mock-path requirement

### 3.1 What V3_PLAN mandates

`V3_PLAN.md` is explicit (migration-strategy point 4):

> "External tools are optional at rest. MLIR and LLVM command-line
> tools (`mlir-opt`, `opt`, `llc`) may be absent on a given machine.
> Every stage ships a mock-validation path that needs no toolchain and
> gates real dispatch behind tool-detection — the exact pattern
> `helixc/backend/gpu_ci.py` already uses for real-HW GPU dispatch."

Stage 201 already realized this for the LLVM toolchain in
`helixc/backend/llvm_toolchain.py`: `detect_llvm_tools()` via
`shutil.which`; `dispatch_validate_ll()` always runs a toolchain-free
`mock_validate_ll`, and only assembles for real when `llvm-as` is
present; a tool-less machine yields a tri-state **`DEFERRED`** result,
never `FAILED`. `gpu_ci.py` does the same for `ptxas`/`naga`/`llvm-mc`/
`xcrun metal` with a frozen `ValidationResult` whose `__post_init__`
makes "a failure with no diagnostic" unrepresentable.

**Phase E must reproduce this discipline for MLIR.** This is the single
biggest design constraint and it has a subtle twist the LLVM case
does not have.

### 3.2 The twist: MLIR is a *library import*, not just a CLI tool

Stage 201's tools (`llvm-as`, `llc`) are **subprocesses** — "absent"
just means `shutil.which` returns `None`. MLIR's Python bindings are an
**`import mlir`** — "absent" means an `ImportError` (or, worse, an
*incompatible-version* import that fails deep inside an API call). The
mock path therefore has to guard a *different failure mode*:

1. **Import-guarded engine.** Phase E must wrap `import mlir` (and any
   `mlir.dialects.*`) in a capability probe — e.g. a
   `detect_mlir_python()` analogous to `detect_llvm_tools()` — that
   returns a structured "available / absent / version-mismatch"
   result. **No module in `helixc/` may import `mlir` at module
   top-level**, or the entire compiler fails to import on a machine
   without the bindings. The MLIR import must be lazy, inside the
   capability probe and the real-dispatch path only.
2. **Tri-state result type.** Reuse the proven shape:
   a frozen dataclass with `mock_passed` / `mock_findings` /
   `real_attempted` / `real_passed` / `real_findings` and a
   `PASSED/FAILED/DEFERRED` status enum, `__post_init__`-validated so
   "absent bindings" => `DEFERRED`, never `FAILED`. A tool-less /
   binding-less CI runner must stay green.
3. **A toolchain-free mock validator for emitted MLIR.** Just as
   `mock_validate_ll` shape-checks `.ll` text and `gpu_ci`'s
   `_validate_ptx` shape-checks PTX, Phase E needs a `mock_validate_mlir`
   that shape-checks emitted MLIR **textual assembly** — `module {`,
   `func.func @`, balanced braces, expected dialect-op prefixes — with
   **no dependency on libMLIR**. This is where `spcl/pymlir` (the
   pure-Python MLIR parser) is genuinely useful: it can parse MLIR text
   into an AST for structural checks without the binary bindings. The
   real path, when the bindings are present, additionally round-trips
   the IR through `mlir.ir.Module.parse()` + the real verifier +
   `mlir-opt`, which is the genuine well-formedness gate.
4. **Two real surfaces to detect, not one.** Phase E may use *both*
   the in-process Python bindings *and* the `mlir-opt` CLI. Detection
   should cover each independently: bindings present? `mlir-opt` on
   PATH? Each absent surface degrades to the mock path on its own.

### 3.3 Why this also de-risks the migration

The mock path is not just a CI-greenness device — it is the mechanism
that makes Phase E **reversible**, which `V3_PLAN.md` requires of every
stage before the single Stage 221 cutover:

- The MLIR path is **additive** — `tir.py` / `tile_ir.py` stay
  untouched and functional. Stage 212's translation and Stage 213's
  lowering are new modules.
- On any machine (or any op) where the MLIR path is unavailable or
  incomplete, the compiler **falls back to the home-grown tile-IR
  path**. The mock/real tri-state is exactly the signal that selects
  the fallback.
- The Stage 215 parity gate then compares the MLIR path against the
  home-grown path across the whole `helixc/tests/` corpus — and only a
  green parity gate authorizes the Stage 221 retirement of the
  incumbents.

### 3.4 Residual risks to record

| Risk | Severity | Mitigation |
|------|----------|------------|
| MLIR Python API churns between LLVM majors | High | Pin one LLVM major for all of Phase E (Section 1.4); version bump = its own stage. |
| Windows binding availability regresses | Medium | conda-forge `win-64` is maintained today; eudsl is the fallback; both monitored at Stage 211. |
| eudsl is alpha (if it becomes primary) | Medium | Prefer conda-forge primary; if forced to eudsl, treat its stability as a tracked risk with a from-source escape hatch. |
| A `helix` op needs C++ traits IRDL cannot express | Medium | Section 2.4: escalate; do not silently adopt a from-source build. |
| `nvgpu` async ops have no AMD/Metal/WebGPU analogue | Medium | A `helix` async abstraction may be required for 4-backend parity (Stage 213 open item). |
| Top-level `import mlir` anywhere in `helixc/` | High | Forbidden by policy; MLIR import must be lazy + capability-probed (Section 3.2). |
| MLIR's own AD / tensor-core story does not match Helix's | Low/Medium | The Stage 117-120 adjoint table stays a Helix-side pass over `helix`/`vector` ops; not delegated to MLIR. |

---

## 4. Summary of recommendations (for the orchestrator to ratify)

1. **Dependency.** The Helix environment is bare `pip` + venv (no
   conda), so depend on the **eudsl find-links wheels**
   (`mlir-python-bindings` via `-f https://llvm.github.io/eudsl`),
   pinned to one LLVM major version; the conda-forge
   `mlir-python-bindings` package is the documented alternative if a
   conda environment is later adopted. **Do not** build MLIR from
   source as the primary path. Phase E is mock-path-first, so the
   binding install does not block development.
2. **Dialect strategy.** Adopt the **hybrid**: upstream dialects
   (`linalg`/`vector`/`arith`/`gpu`/`func`/`memref`/`nvgpu`) for the
   ~80-85% numerical/structural core; a **small `helix` dialect** for
   the Helix-specific residual (`grad`/`jvp`/`vmap`, the `agi.*`
   metaprogramming ops, the atomic arena allocator, and — pending
   review — the Result/quant encodings). The `helix` dialect **lowers
   to upstream**. Define it via **IRDL / Python registration**, not
   C++/ODS, to keep the no-from-source posture.
3. **Mock path.** Reproduce the Stage-201 / `gpu_ci` discipline for
   MLIR: a `detect_mlir_python()` + `mlir-opt` capability probe, a
   **lazy** (never top-level) MLIR import, a toolchain-free
   `mock_validate_mlir` shape-checker (pure-Python, `pymlir`-style),
   and a frozen tri-state `PASSED/FAILED/DEFERRED` result type so a
   binding-less / tool-less machine yields `DEFERRED` and CI stays
   green. This same tri-state is what selects the home-grown-tile-IR
   fallback that keeps Phase E reversible until the Stage 221 cutover.

---

## 5. Open items carried to Stage 211 (consolidated)

- [x] ~~Confirm whether Helix's dev/CI environment can adopt
  conda/mamba.~~ **RESOLVED (Stage 210):** bare `pip` + venv, no conda
  — the eudsl find-links wheels are the primary dependency path
  (Section 1.4).
- [ ] Pin the exact LLVM major version for all of Phase E; record it
  in the environment spec.
- [ ] Decide `result_pack`/`quantize`: custom `helix` ops vs. upstream
  `arith`/`quant`.
- [ ] Confirm IRDL can express the arena atomic-pair/triple invariant;
  if not, decide verification-pass vs. from-source C++.
- [ ] Decide Tile-IR async representation: `nvgpu`-only vs. a `helix`
  async abstraction fanning out to all four GPU backends.
- [ ] Specify the `mock_validate_mlir` shape grammar and whether to
  vendor `spcl/pymlir` for the pure-Python parse.
- [ ] (Stage 213) Decide how the Stage 117-120 adjoint table
  (`TILE_OP_ADJOINTS`, currently keyed by `TileOpKind`) migrates: do
  its keys become MLIR op-name strings, or does the Helix AD pass
  pattern-match over the lowered `helix` / `vector` ops? — flagged by
  the Stage-210 architecture review.

---

## 6. Stage-210 ratification note — architecture-review corrections

This decision was finalized after an independent architecture review
that verified the op-mapping against the real `tir.py` / `tile_ir.py`.
The review confirmed all three recommendations sound; its corrections,
applied to this record:

1. **Op counts** — corrected to the exact figures: `tir.py` has 98
   `OpKind` members (the draft said ~110); `tile_ir.py` has 34
   `TileOpKind` members (the draft said ~30). The ~80-85%
   upstream-mappable conclusion is unaffected.
2. **`TENSOR_LOAD` / `TENSOR_STORE`** belong with the effectful
   runtime ops (`PRINT` / `TRAP` / `FFI_CALL`), NOT the in-graph
   `tensor.*` ops — they are external host / file I/O boundaries and
   lower to runtime `func.call`s. Marginal fit, same class as
   `FFI_CALL`.
3. **`scf` vs `cf`** — the Tensor IR's `BR` / `COND_BR` map to `cf`,
   but a loop reconstructed as `scf.for` / `scf.if` is what the
   upstream `linalg` lowering passes expect. Stage 212's tile-IR ->
   MLIR translation should prefer raising structured loops to `scf`
   rather than leaving them as raw `cf` branches.
4. **`detect_mlir_python()`** must probe the dialect sub-modules
   (`mlir.dialects.linalg`, etc.) independently of the top-level
   `import mlir`: the eudsl / conda-forge packages ship dialects as
   separately-importable modules, so a partial install (core present,
   dialect bindings absent) must degrade to DEFERRED rather than pass
   the probe and then fail deep inside a dialect call.
5. The **adjoint-table key-type migration** is added as a Stage-213
   open item (Section 5).

---

## Appendix A — Sources consulted (2026-05-20)

- MLIR Python Bindings — official docs:
  https://mlir.llvm.org/docs/Bindings/Python/
- MLIR `irdl` dialect (Python) — for IRDL-based dialect definition:
  https://mlir.llvm.org/python-bindings/autoapi/mlir/dialects/irdl/index.html
- LLVM forum, "PSA: better dialect types and attributes support in the
  Python bindings" (Jan 2026):
  https://discourse.llvm.org/t/psa-better-dialect-types-and-attributes-support-in-the-python-bindings/89370
- `llvm/eudsl` — successor to mlir-wheels, official LLVM-org:
  https://github.com/llvm/eudsl
- `makslevental/mlir-wheels` — archived 2025-11-29, migrated to eudsl:
  https://github.com/makslevental/mlir-wheels
- conda-forge `mlir-python-bindings` (win-64; updated 2026-01-08):
  https://anaconda.org/conda-forge/mlir-python-bindings
- conda-forge `mlir` (v22.x; updated 2026-02):
  https://anaconda.org/conda-forge/mlir
- `spcl/pymlir` — pure-Python MLIR parser (mock-path reference):
  https://github.com/spcl/pymlir
- LLVM issue #74245 — registering upstream dialects from Python:
  https://github.com/llvm/llvm-project/issues/74245
- MLIR "Creating a Dialect" tutorial:
  https://mlir.llvm.org/docs/Tutorials/CreatingADialect/
- LLVM Windows-bindings build patches (D125122 / D125134 / D125284) —
  Windows from-source fragility evidence:
  https://reviews.llvm.org/D125122

*End of DRAFT — awaiting orchestrator review for Phase E.*
