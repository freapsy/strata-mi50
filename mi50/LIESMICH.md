# Strata auf 2× MI50 32 GB: alle Verbesserungen

Dieses Repo ist Strata 0.2.8 (Upstream-Commit `870aedd9`, github.com/Niko1221/Strata) plus alle Verbesserungen für
gfx906 aus der Optimierung am 30.10.2026. Der Branch heißt `mi50-gfx906`, `main` steht unverändert auf Upstream.

## Ergebnis

Gemessen auf derselben Maschine, mit demselben Modell (Qwen3.8 Flash Next IQ3_M) und denselben drei Anfragen, je
192 Tokens:

| Stand | Code | Prosa (deutsch) | long10k (9.927-Token-Prompt) |
| --- | ---: | ---: | ---: |
| Original 0.2.8 (e0) | – | – | 51,5 t/s |
| Referenz f0 (vorhandene Schalter, Drafter auf der letzten Karte) | 90,1 | 52,0 | 61,8 |
| **Jetzt f27** | **115,0** | **74,7** | **82,3** |
| f27 mit 131.072 Kontext (dein Alltags-Setup) | 112,7 | 73,6 | 81,3 |

Ab f0 sind alle Antworttexte bitgleich: Die Kernel rechnen exakt dasselbe, nur schneller. Die Prosa profitiert
zusätzlich vom deutsch-englischen Draft-Vokabular (Codex).

Die Erklärung auf Englisch, für Upstream: [docs/AMD_MI50.md](../docs/AMD_MI50.md). Alle Läufe mit Prompts und Hashes:
[bench/results/2026-10-30-mi50-32gb-decode](../bench/results/2026-10-30-mi50-32gb-decode/README.md).

## Was drin ist (Commits auf `mi50-gfx906`)

| Commit | Was | Schalter |
| --- | --- | --- |
| `[Engine] HIP: a constant-mask __shfl_xor_sync …` | Warp-Summen per DPP/ds_swizzle statt LDS-Rundreise | immer an (HIP) |
| `[Engine] Sampler and verify: eight logit loads …` | Sampler wartet nicht mehr 243× nacheinander auf den Speicher | immer an |
| `[Engine] BF16 GEMV: STRATA_MMVF_ROWS=1 …` | Router umgeht den gfx906-Dispatch-Stall (46 → 11 µs je Schicht) | `STRATA_MMVF_ROWS=1` |
| `[Engine] gfx906 MMVQ (Q4_K, Q6_K) …` | Aktivierungen im LDS, „vier Lanes in einer“ | `STRATA_MMVQ_XLDS/WPB/Q6X4/Q4X4` |
| `[Engine] gfx906 IQ GEMV and grouped experts …` | Experten-Kernel: globale Loads, volle Waves, H2, Q5_0 per LDS | `STRATA_IQ_MMVQ_*`, `STRATA_EXP_*` |
| `[Engine] gfx906 hyper-connection read …` | Neuer hc-down-Kernel, up-Kernel ohne zweite Runde, GR_SPLIT-Race behoben | `STRATA_GR_DOWN_PF=1`, `STRATA_GR_DOWN_TG=a` |
| `[Engine] Benches …` | Bench prüft T-Invarianz und Hash vor der Quantisierung | – |
| `[Engine] HIP pipeline …` | Host-Fixes gegen Pipeline-Hänger (**von Codex**) | – |
| `[Docs] 2x MI50 32 GB …` | Doku und Messungen | – |
| `[Other] mi50/ …` | Dieser Ordner (nur für deinen Fork, geht nicht in PRs) | – |

## Selbst bauen und starten (generisch, ohne deine Pfade)

```sh
cmake -S . -B build-906 -G Ninja -DSTRATA_HIP_GFX906=ON -DSTRATA_ORCA_Q4KS_MMQ=ON -DCMAKE_HIP_ARCHITECTURES=gfx906 \
  -DSTRATA_GGML_DIR=$PWD/third_party/llama.cpp \
  -DCMAKE_C_COMPILER=<ROCM>/llvm/bin/clang -DCMAKE_CXX_COMPILER=<ROCM>/llvm/bin/clang++
ninja -C build-906 strata
```

Danach [strata-mi50.example.json](strata-mi50.example.json) kopieren, die `<…>`-Platzhalter ersetzen und wie gewohnt
mit `serve/server.py` starten. Auf deinem Rechner läuft das schon so: `~/Downloads/strata/START-MI50.sh --start`.

## Veröffentlichen (erst nach deinem OK)

**Wichtig:** Ein GitHub-Fork eines öffentlichen Repos ist immer öffentlich. Damit es erst privat bleibt, legst du
unter freapsy ein **neues privates Repo** an (keinen Fork) und pushst dorthin:

```sh
# erledigt am 30.10.2026:
git fetch --unshallow upstream            # das lokale Repo war flach (nur 870aedd9)
gh repo create freapsy/strata-mi50 --private
git remote add origin https://github.com/freapsy/strata-mi50.git
git push -u origin mi50-gfx906 main
```

Später, für die PRs an Niko1221/Strata: Upstream auf GitHub forken (der Fork ist öffentlich), je PR einen Branch aus
`upstream/main` anlegen, die passenden Commits per `git cherry-pick` übernehmen und als **Draft-PR** öffnen. Reihenfolge und
Texte: [PR-ENTWUERFE.md](PR-ENTWUERFE.md). Vorher `upstream/main` holen und neu aufsetzen, denn Upstream ist inzwischen
weiter.

**Autor der Commits:** `freapsy <212416078+freapsy@users.noreply.github.com>` (GitHub-Noreply, die private Mail bleibt
verborgen).

Die Commits tragen am Ende `Co-Authored-By: Claude` und einen Session-Link. Beides kann vor dem Push entfernt werden.

## Was nicht getestet ist

- Nur gfx906 gebaut und gemessen. CUDA, SYCL, Metal und RDNA-HIP wurden nicht gebaut. Einige Änderungen betreffen
  gemeinsamen Code (Sampler, Shuffles, `__launch_bounds__`), deshalb gehen die PRs als Draft raus.
- Die einzelnen Commits wurden nicht jeder für sich gebaut. Gebaut ist der Endstand, und der entspricht dem
  Produktions-Binary.
- `amdgpu_waves_per_eu(3)` am Experten-gate/up-Kernel gilt für alle HIP-Ziele (auch gfx1151). Vor dem Upstream-PR
  sollte das auf gfx906 beschränkt werden.
- Gemessen wurde nur ein Modell auf einer Maschine.
