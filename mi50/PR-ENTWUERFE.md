# PR- und Issue-Entwürfe für Niko1221/Strata

Diese Texte sind auf Englisch, weil Upstream englisch ist. Sie gehen nur mit deinem OK raus, und zwar von deinem Account
(Fork unter freapsy). Regeln aus CONTRIBUTING.md: ein Problem je PR, Titel mit `[Engine]`/`[Docs]`/`[Benchmark]`,
Validation getrennt nach Passed/Failed/Not run, keine privaten Pfade. Was nicht auf jedem Backend gebaut ist, geht als
**Draft** raus.

## Reihenfolge

| # | PR | Commit(s) auf `mi50-gfx906` | braucht | Vorher erledigen |
| --- | --- | --- | --- | --- |
| 1 | Router dispatch stall | `[Engine] BF16 GEMV: STRATA_MMVF_ROWS=1 …` | – | – |
| 2 | HIP pipeline host fixes (Codex) | `[Engine] HIP pipeline …` | – | Codex als Finder nennen |
| 3 | GR_SPLIT race (Bugfix) | Teil von `[Engine] gfx906 hyper-connection read …` | – | Race-Fix + `return q8` als eigenen Commit herauslösen |
| 4 | Sampler/verify loads in flight | `[Engine] Sampler and verify …` | – | CUDA-Build bitten |
| 5 | HIP shuffles via DPP | `[Engine] HIP: a constant-mask …` | – | – |
| 6 | Scope issue: gfx906 kernel series | – | – | Issue A unten, auf Antwort warten |
| 7 | gfx906 MMVQ Q4_K/Q6_K | `[Engine] gfx906 MMVQ …` | 5, 6 | – |
| 8 | gfx906 IQ GEMV + experts | `[Engine] gfx906 IQ GEMV …` | 5, 6 | `amdgpu_waves_per_eu` auf gfx906 begrenzen |
| 9 | gfx906 hc read | Rest von `[Engine] gfx906 hyper-connection read …` | 3, 5, 6 | – |
| 10 | Benches | `[Engine] Benches …` | 9 | – |
| 11 | Docs + Benchmark | `[Docs] 2x MI50 32 GB …` | 1–10 | Docs und `bench/results` je nach Wunsch der Maintainer trennen |

Pro PR:

```sh
git fetch upstream && git switch -c pr/router-stall upstream/main
git cherry-pick <commit>          # Konflikte lösen, falls Upstream weiter ist
# gfx906 bauen, die drei Anfragen laufen lassen, Hashes vergleichen
git push -u fork pr/router-stall
gh pr create --draft --repo Niko1221/Strata --title "<Titel>" --body-file <Body unten>
```

---

## PR 1: `[Engine] gfx906: the one-token BF16 router GEMV avoids a 512-block dispatch stall (STRATA_MMVF_ROWS=1)`

### Problem and result
On gfx906, a launch of 509..512 blocks of 256 threads that does more than about 0.5 µs of work never finishes in under
about 46 µs. The blocks are done after about 1.5 µs; the rest is a gap before the next kernel. The BF16 router projection
(512 rows, one 256-thread block per row) hits this on every layer for one-token GEMVs. With `STRATA_MMVF_ROWS=1`, the
one-token case now also takes the rows kernel, with 2 rows per block: the router takes about 11 µs instead of about 46 µs.
End to end on 2x MI50 32 GB with Qwen3.8 Flash Next IQ3_M, long10k goes from 70.9 to 71.6 tok/s. The texts are
bit-identical.

### Related discussion
none (opt-in switch that already exists; the default path is unchanged)

### Approach
`bf16_gemv_fp32_mmvf` sends one token with n_out ≥ 64 to `bf16_gemv_fp32_mmvf_multi` when `STRATA_MMVF_ROWS=1`, and
the rows kernel produces the same bits as the per-row kernel. `RPB` goes from 4 to 2. This also changes the block shape
of the existing multi-token rows path on CUDA (same sums).

### Validation
**Passed:** gfx906 build (ROCm 7.14 community), base 870aedd9 + this commit. Three fixed requests (code, prose, 9,927-token
prompt, 192 greedy tokens each) give identical text hashes with and without the switch. Router launch timing comes from
`AMD_LOG_LEVEL=3` and a standalone `s_memrealtime` kernel.
**Failed:** none observed
**Not run:** CUDA build and CUDA timing of `STRATA_MMVF_ROWS=1` with RPB=2 (no NVIDIA card here).

### Limits
Measured on one machine and one model. The stall is a hardware or firmware property of gfx906; this commit only
avoids it.

---

## PR 2: `[Engine] HIP pipeline: reserve the MTP prefill, do not unpin under a window in flight`

### Problem and result
With `--pipeline-windows 2` on 2x MI50, runs could hang (#267-like: "an earlier window never finished on the GPU").
Three host-side causes, found in a Codex session on the same machine:
1. `prepare_prefill` sizes the drafter's prefill inputs for 3 rows, but the pipelined read hands it 5. The first larger
   window then frees and allocates memory while the device waits on the host's doorbell service.
2. `hipHostUnregister` in `pl_apply` waits for windows that still need the host to serve their flags.
3. The gfx906 build (`STRATA_HIP_GFX906` without `STRATA_USE_HIP`) took the CUDA drafter-stream priority rule.

After the fix, the daily 128k config runs without hangs. Code, prose and long10k texts are bit-identical to before.

### Approach
`mtp.reserve_prefill(o.spec, e)`, an existing function, instead of `prepare_prefill(e)`. `pl_apply` returns early on
HIP while pinned memory is live and a stage still has a verify in flight, and retries on the next round. In
`dflash2.cpp` the `#if` also covers `STRATA_HIP_GFX906`.

### Validation
**Passed:** gfx906 build. The 131,072-context config with `--pipeline-windows 2`: three requests give the reference
texts.
**Failed:** none observed
**Not run:** CUDA (the `reserve_prefill` change is not behind an `#if`), SYCL.

### Limits
The hang was timing-dependent. There is no deterministic repro test.

---

## PR 3: `[Engine] gfx906 STRATA_GR_SPLIT: one slice-sum slot per stream (the drafter raced the verifier)`

*(Vorher aus dem hc-Commit herauslösen: `g_gr_part` je Stream-Slot, `gr_stream_slot`, `m.slot`, sowie `return q8` im GR_SPLIT-Zweig.)*

### Problem and result
With `STRATA_GR_SPLIT=1` on gfx906, the slice sums were one module-scope array shared by every stream. The MTP drafter's
hyper-connection read on its own stream raced the verifier's, and the window's text changed from run to run. The
branch also returned `false` although the up kernel had already written the q8_1 image, so each read quantized twice.
Now each stream has its own slot (an error if the slots run out), and the branch returns `q8`. The text is
deterministic.

### Validation
**Passed:** gfx906 build; repeated runs with `STRATA_GR_SPLIT=1` give the same text; fused_gr_bench is T-invariant (T 1..8).
**Not run:** other backends (the code is under `STRATA_HIP_GFX906`).

---

## PR 4: `[Engine] Sampler and verify: eight logit loads in flight per thread`

Body: the commit message of `[Engine] Sampler and verify …`, sorted into the template sections. Ask the maintainers for
a CUDA build (it changes the default path on every backend).

## PR 5: `[Engine] HIP: a constant-mask __shfl_xor_sync is a DPP or ds_swizzle step`

Body: the commit message. Note that this changes code generation on every HIP target, wave32 too. ds_swizzle in
bitmask mode and DPP quad_perm are the same lane exchange there, but it was not tested on RDNA.

## PR 7–9: gfx906 kernel series

Only after the answer to Issue A. Body: the commit messages. Measurements come from `docs/AMD_MI50.md`. Each switch is
opt-in except the code-generation changes listed under "Default path".

---

## Issue A (scope): `[Engine] gfx906: opt-in kernel switches for 2x MI50 decode (+33 % long10k, bit-identical)`

On 2x MI50 32 GB (Qwen3.8 Flash Next IQ3_M), a series of opt-in gfx906 kernel changes raises decode from 61.8 to
82.3 tok/s at a 9,927-token prompt (code request 90.1 → 115.0) with bit-identical texts. Some of the gain also needs a
draft vocabulary. The changes cover:

- Q4_K/Q6_K GEMV with activations in LDS and four lanes per thread;
- IQ GEMV with staged rows and two rows per warp;
- grouped experts (H2, global loads, occupancy);
- a new hyper-connection down kernel.

They are measured in `docs/AMD_MI50.md` on our fork. Would you take them as one draft PR per kernel family (MMVQ,
IQ/experts, hyper-connection), all behind `STRATA_*` switches that are off by default? We can only build gfx906.

## Issue B (bug): `[Engine] long-prompt text depends on pipeline timing with --pipeline-windows 2`

### What happens
With `--pipeline-windows 2 --spec 3`, the same 9,927-token prompt (temperature 0, seed 42) sometimes produces another
text: about 1 run in 9, with a different window structure (for example 101 instead of 99 windows, 22 "on the path").
The kernels are identical between those runs (same binary, same config). Inserting dummy graph nodes
(`STRATA_DUMMY_NODES=1` or `3`) also changes the text, each to another fixed text. Short prompts
(code, prose) were stable in every run.

### Expected
The same text in every run, as with `--pipeline-windows 0`.

### Environment
2x Instinct MI50 32 GB, gfx906 build of 0.2.8 (870aedd9), ROCm 7.14, `--layer-split 24 --spec 3 --kv int8
--pipeline-windows 2 --trim-stage-weights --resident-experts`, MTP drafter on the last card.

### Guess
Some part of the speculative window path (gate decision, draft acceptance, or which window's results are applied) is not
invariant to how long the stages take. The hyper-connection split path had such a race (PR 3). This one occurs with
the default path too.
