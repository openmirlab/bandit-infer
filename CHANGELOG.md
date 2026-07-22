# Changelog

## 0.1.0 - 2026-07-22

- Added the initial inference-only package boundary, registry, cache resolver,
  explicit device validation, and `BanditSession` lifecycle contract.
- Added independent v1 Music64 and v2 Multi runtime graphs, each validated
  stem-by-stem against an untouched-upstream fixture from an official local
  checkpoint.
- Recorded SHA-256 integrity values for the official `v1-mus64-l1snr` and
  `v2-multi` artifacts. Their cache download path is enabled; all other catalog
  entries remain fail-closed until equivalent evidence exists.
