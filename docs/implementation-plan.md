# Bandit inference implementation plan

## Goal

Provide `bandit-infer`, an inference-only Python package for two independent
upstream Bandit code lines: Bandit v1 (`840d5eb9ede59d64569c423244547e58cb00f647`)
and Bandit v2 (`d5563d9031e95fdaa3e5a73d5020b9a0df61adb6`). The package must
not convert checkpoints or share model layers between these backends.

## Grounded inventory

Zenodo record 10160698 contains 23 files: 21 Bandit-family artifacts and two
explicitly excluded baselines, `dnr-demucs.ckpt` and `dnr-umxhq.ckpt`. Zenodo
record 12701995 contains seven v2 artifacts. Therefore the intended catalog is
28 entries. Both upstream repositories have Apache-2.0 LICENSE files. The v1
Zenodo record declares CC-BY-NC-4.0 weights and the v2 record declares
CC-BY-SA-4.0 weights; those are distinct from the Apache-2.0 code license.

## Boundaries

- Ship only model code, audio normalization, inference chunking, cache/download,
  device validation, and registry-driven session lifecycle.
- Keep copied/adapted v1 graph under `bandit_infer._v1` and v2 graph under
  `bandit_infer._v2`; do not extract a common model base or checkpoint converter.
- Share only non-numerical resolver/device/manifest utilities.
- v1 model keys identify their exact graph family/config and native rate;
  v2 keys use the single `bandit-mus64` graph at 48 kHz.

## Execution sequence

1. Establish the package/data/lifecycle contract and a catalog that refuses any
   artifact whose SHA-256 has not been independently computed. This first
   commit makes the integrity gap visible rather than inventing hashes.
2. Port the v1 inference graph verbatim enough to load each distinct family,
   then record a real representative Music64 golden fixture from untouched v1.
3. Port the v2 graph independently, record a multi-language v2 fixture, and
   compare every returned stem with the untouched upstream run.
4. Fill every catalog SHA-256 from locally downloaded official artifacts,
   execute real-load compatibility for every v1 graph family and all seven v2
   checkpoints, and only then enable official artifact auto-download.

## Acceptance evidence

The permanent commands are listed in `CLAUDE.md`. Float golden fixtures record
torch/device metadata and skip bit-digest checks when the recording environment
does not match. Until steps 2–4 are complete, the package intentionally reports
official model graphs and SHA-256 values as unverified; it must not claim
end-to-end support merely from a filename or a Zenodo MD5 digest.
