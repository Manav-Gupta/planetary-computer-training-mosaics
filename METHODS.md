# Methods log

Running record of methodological choices made in this repo, in the order
they were decided. Each entry: what was decided, why, and what alternatives
were considered.

## Cloud masking (branch `fix/addCloudMasking`)

Context: the existing pipeline masks clouds using the Sentinel-2 L2A
Scene Classification Layer (SCL), keeping pixels whose SCL class is in
`valid_scl_classes` (config default `[4, 5, 6, 11]` = vegetation / bare
soil / water / snow). This is fast (no extra compute) but SCL is known to
miss thin cloud and mis-classify cloud shadow in some scenes.

Goal: add an optional GPU-accelerated deep-learning cloud/shadow mask using
[OmniCloudMask](https://github.com/DPIRD-DMA/OmniCloudMask) (DPIRD-DMA),
run on the Red/Green/NIR bands, as a higher-accuracy alternative or
supplement to SCL.

- 2026-08-20: Repo scaffolding for this work — new branch
  `fix/addCloudMasking`, `.claude/` git-ignored, `CLAUDE.md` project rules
  added, this file created. No masking code yet; architectural choices
  (SCL replace vs. supplement, pipeline integration point, GPU package
  install, thresholds) to be decided with the user before implementation.
