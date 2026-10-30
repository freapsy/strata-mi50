# Strata MI50: Qwen3.8 Flash Next auf zwei Instinct MI50 32 GB

Schnellerer Decode für **Qwen3.8 Flash Next IQ3_M** auf **2× AMD Instinct MI50 32 GB (gfx906)**. Die Basis ist die
Inferenz-Engine [Strata](https://github.com/Niko1221/Strata). Die Änderungen sind gfx906-Kernel, die genau dieselben
Ergebnisse liefern wie vorher, nur schneller: Die Antworttexte sind Bit für Bit gleich.

*English: gfx906 kernel work for Strata on two MI50 32 GB cards. long10k decode 51.5 → 82.3 tok/s, code 90 → 115 tok/s,
bit-identical output. Details in [docs/AMD_MI50.md](docs/AMD_MI50.md).*

## Ergebnis

Gemessen wurden drei feste Anfragen mit je 192 Tokens auf derselben Maschine:

| | Code | Prosa (deutsch) | long10k (9.927-Token-Prompt) |
| --- | ---: | ---: | ---: |
| Strata 0.2.8, unverändert | – | – | 51,5 tok/s |
| Referenz (vorhandene Schalter, Drafter auf der zweiten Karte) | 90,1 | 52,0 | 61,8 |
| **Dieses Repo** | **115,0** | **74,7** | **82,3** |
| Dieses Repo mit 131.072 Kontext (Alltag) | 112,7 | 73,6 | 81,3 |

Alle Läufe mit Prompts, Zeiten und Text-Hashes stehen unter
[bench/results/2026-10-30-mi50-32gb-decode](bench/results/2026-10-30-mi50-32gb-decode/README.md).

## Was verbessert wurde

1. **Router ohne Dispatch-Stall.** gfx906 hängt bei Launches von 512 Blöcken × 256 Threads rund 46 µs fest. Der Router
   lief in jeder Schicht hinein. Jetzt braucht er ~11 µs.
2. **GEMVs mit weniger Adress-Instruktionen.** Die Aktivierungen liegen im LDS, und ein Thread übernimmt die Arbeit von
   vier („vier Lanes in einer“). Q6_K-Kopf und Q4_K-Projektionen sind 20–27 % schneller.
3. **Experten-Kernel mit vollen Waves.** Globale statt Flat-Loads, `v_perm` statt Viertel-Takt-Multiplikation, H2-Aufteilung
   für IQ3_S, Q5_0-down über LDS.
4. **Hyper-Connection-Lesen.** Ein neuer down-Kernel verteilt die Arbeit auf 55 statt 41 CUs und lädt alles vorab. Der
   up-Kernel kommt ohne zweite Runde aus. Warp-Summen laufen per DPP statt LDS. Außerdem ist ein Race im
   `GR_SPLIT`-Pfad behoben.
5. **Sampler.** Acht Loads gleichzeitig statt 243 Speicherwartezeiten nacheinander.
6. **Pipeline-Hänger behoben** (Host-Fixes aus der Codex-Session).
7. **Deutsch-englisches Draft-Vokabular** für den MTP-Drafter. Mehr akzeptierte Drafts bei deutscher Prosa.

Jeder Punkt ist mit Messwert und Prüfung beschrieben in [docs/AMD_MI50.md](docs/AMD_MI50.md).

## Hardware

| | |
| --- | --- |
| GPUs | 2× Instinct MI50 32 GB, je 150 W; Karte 0 an der CPU, Karte 1 über PCIe 3.0 ×4 am Chipsatz |
| CPU / RAM | Core i5-11600K, 64 GB |
| Software | Linux 7.0, ROCm 7.14 (gfx906-Community-Build) |
| Modell | OrcaRouter Qwen3.8 Flash Next Uncensored IQ3_M, Strata-Pack `orca-iq3_m` |

## Bauen und starten

```sh
git clone https://github.com/freapsy/strata-mi50.git && cd strata-mi50
cmake -S . -B build-906 -G Ninja -DSTRATA_HIP_GFX906=ON -DSTRATA_ORCA_Q4KS_MMQ=ON -DCMAKE_HIP_ARCHITECTURES=gfx906 \
  -DSTRATA_GGML_DIR=$PWD/third_party/llama.cpp \
  -DCMAKE_C_COMPILER=<ROCM>/llvm/bin/clang -DCMAKE_CXX_COMPILER=<ROCM>/llvm/bin/clang++
ninja -C build-906 strata
```

Danach [mi50/strata-mi50.example.json](mi50/strata-mi50.example.json) kopieren, die `<…>`-Platzhalter (Modell, Pack, ROCm)
eintragen und den Server mit dieser Config starten:

```sh
python3 serve/server.py --config strata-mi50.json
```

Die Config schaltet alle Verbesserungen ein. Jeder neue Schalter (`STRATA_…` unter `env`) ist ohne die Config aus.

## Stand

- Gebaut und gemessen wurde nur auf gfx906, mit einem Modell und auf einer Maschine.
- CUDA, SYCL, Metal und RDNA wurden nicht gebaut.
- Die Änderungen sollen später als einzelne Pull Requests an Strata gehen. Die Entwürfe stehen in
  [mi50/PR-ENTWUERFE.md](mi50/PR-ENTWUERFE.md).
- Bekanntes Problem aus Strata selbst: Bei langen Prompts mit `--pipeline-windows 2` hängt der Text manchmal vom Timing
  ab (etwa 1 von 9 Läufen).

## Credits

- [Strata](https://github.com/Niko1221/Strata) von Niko1221 und den Strata-Mitwirkenden, MIT-Lizenz (siehe [LICENSE](LICENSE)).
- MI50-Optimierung: freapsy, mit Claude (Kernel, Messungen) und Codex (Host-Fixes, Draft-Vokabular).
