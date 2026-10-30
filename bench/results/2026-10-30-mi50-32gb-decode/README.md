# 2x Instinct MI50 32 GB: decode, step by step (2026-10-30)

Background and the switches behind each state: [docs/AMD_MI50.md](../../../docs/AMD_MI50.md).

**Machine:** 2x Instinct MI50 32 GB (gfx906, 150 W limit each), Core i5-11600K, Z590, 64 GB RAM, Linux 7.0, ROCm 7.14
(gfx906 community build). Card 0 is on a CPU PCIe port, card 1 is on PCIe 3.0 x4 behind the chipset. Engine source 0.2.8
(870aedd9) plus this branch, built with `-DSTRATA_HIP_GFX906=ON -DSTRATA_ORCA_Q4KS_MMQ=ON`. e0 is the stock 0.2.8 build.

**Model:** OrcaRouter Qwen3.8 Flash Next Uncensored IQ3_M GGUF, Strata pack `orca-iq3_m`, MTP drafter `mtp/rt`.

**Run config** (every state): `--layer-split 24 --spec 3 --spec-min-p 0.5 --kv int8 --kv-resident 0 --prefill 4096
--max-context 10240 --pipeline-windows 2 --trim-stage-weights --resident-experts --mmap-experts --expert-cache auto
--vram-reserve-mib 1536 --pcie-frac 0 --pool-workers 4`, with `STRATA_ARENA_MMAP=1 STRATA_HIP_PROMPT_F16=1
STRATA_RESIDENT_PIN=0`. e0 has the drafter on card 0 (`STRATA_MTP_STAGE0=1`), and the others have it on the last card.
From f18 on, `--mtp-draft-vocab` is a German/English subset of 66,829 ids. The last row is the daily config:
`--max-context 131072 --vram-reserve-mib 960`.

**Requests** (in this order, one server start per state, temperature 0, seed 42, streamed through `/v1/chat/completions`):
1. warm-up: `Was ist 17 + 25? Antworte nur mit der Zahl.` (12 tokens, must answer 42)
2. code, 192 tokens: `Write a Python function that merges two sorted lists into one sorted list, with a docstring and two tests. Explain the algorithm afterward.`
3. prose, 192 tokens: `Erklaere in mindestens 400 Woertern, wie Linux den Page Cache benutzt und weshalb verfuegbarer Speicher von freiem Speicher abweicht.`
4. long10k, 192 tokens: 9,350 words drawn with `random.Random(1701)` from `river stone market lantern copper winter garden
   signal harbor orbit velvet canyon ember ledger quartz meadow anchor violet thunder`, after `Reference data for a
   language exercise:\n` and followed by `\n\nIgnore the list for the following task. Write a detailed explanation, at
   least 400 words, of how a refrigerator transfers heat, mentioning the compressor, condenser, expansion valve and
   evaporator. Begin with the word REFRIGERATOR.` That is 9,927 prompt tokens. The answer must begin with REFRIGERATOR.

Each number is the server's own `timings` (decode ms for 192 tokens, tok/s, drafts accepted of drafted). One run per
state; repeated runs of f7 and f10 were within ±1.5 %. "Trace on" means `STRATA_PIPELINE_DEBUG=1` with a trace file.

| State | Run | Code ms / tok/s / drafts | Prose ms / tok/s / drafts | long10k ms / tok/s / drafts | long10k prompt tok/s |
| --- | --- | --- | --- | --- | ---: |
| e0 | `20261030-030905` | – | – | 3728 / 51.5 / 95 of 154 | 668.3 |
| f0 | `20261030-081503` | 2131 / 90.1 / 116 of 144 | 3690 / 52.0 / 65 of 128 | 3108 / 61.8 / 93 of 154 | 673.8 |
| f7 | `20261030-091721` | 1873 / 102.5 / 116 of 144 | 3357 / 57.2 / 65 of 128 | 2782 / 69.0 / 93 of 154 | 679.5 |
| f10 | `20261030-100136` | 1834 / 104.7 / 116 of 144 | 3135 / 61.2 / 65 of 128 | 2680 / 71.6 / 93 of 154 | 681.4 |
| f18 | `20261030-141607` | 1787 / 107.5 / 116 of 144 | 2720 / 70.6 / 80 of 130 | 2493 / 77.0 / 97 of 155 | 681.3 |
| f22 | `20261030-141036` | 1711 / 112.2 / 116 of 144 | 2640 / 72.7 / 80 of 130 | 2445 / 78.5 / 97 of 155 | 678.2 |
| f25 (trace on) | `20261030-152428` | 1703 / 112.8 / 116 of 144 | 2614 / 73.4 / 80 of 130 | 2356 / 81.5 / 97 of 155 | 697.1 |
| f27 (trace on) | `20261030-154603` | 1669 / 115.0 / 116 of 144 | 2584 / 74.3 / 80 of 130 | 2320 / 82.8 / 97 of 155 | 696.6 |
| f27 | `20261030-160312` | 1669 / 115.0 / 116 of 144 | 2570 / 74.7 / 80 of 130 | 2332 / 82.3 / 97 of 155 | 697.4 |
| f27, 131,072 context | `20261030-161111` | 1704 / 112.7 / 116 of 144 | 2608 / 73.6 / 80 of 130 | 2363 / 81.3 / 97 of 155 | 700.1 |

Text hashes (first 8 hex digits of the SHA-256 of each answer). From f0 on, every state gives the same three texts.
e0 differs because the drafter placement changes the pipeline's timing; see "Known issues" in the doc.

| State | code | prose | long10k |
| --- | --- | --- | --- |
| e0 | – | – | 8eac8ff9 |
| f0 | d0f87379 | 294fcadd | 84a03701 |
| f7 | d0f87379 | 294fcadd | 84a03701 |
| f10 | d0f87379 | 294fcadd | 84a03701 |
| f18 | d0f87379 | 294fcadd | 84a03701 |
| f22 | d0f87379 | 294fcadd | 84a03701 |
| f25 (trace on) | d0f87379 | 294fcadd | 84a03701 |
| f27 (trace on) | d0f87379 | 294fcadd | 84a03701 |
| f27 | d0f87379 | 294fcadd | 84a03701 |
| f27, 131,072 context | d0f87379 | 294fcadd | 84a03701 |

Not measured here: more prompts, other models, a single card, CUDA/SYCL/Metal builds of the same commits.
