# Backups — full project snapshots

This directory holds **complete, restorable backups of the entire project** (every
file and all git history), as single-file git bundles. The large `.bundle` files
are git-ignored (they would bloat the repo); each is documented + checksummed here
so it is verifiable.

A git bundle is a one-file clone source: `git clone <file>.bundle <dir>` restores
the whole repository — all branches, all tags, all history — with no network and
no remote.

---

## v0 — pre-K4 full snapshot (INCLUDING the Python reference compiler)

- **File:** `kovostov-native-v0-pre-k4-8d593da.bundle`
- **Size:** 9,951,905 bytes
- **SHA-256:** `00e36eb92cf38e05d3c28f90ca758ca329f7dd35760fa454aaca4c8c50f18eba`
- **HEAD captured:** `8d593da194c19a92dd25f6aabaa47ce6f2bc3fc0`
- **Git tag:** `v0-pre-k4-full-with-python` (also pushed to origin)
- **Date:** 2026-05-31

**Why this exists.** This is the complete project state **before any K4 Python
deletion** — it deliberately includes the **Python reference compiler**
(`helixc/` — 164 `.py` files: frontend, backend, ir, and the 93-file test suite).
After the bootstrap was proven Python-free (the diverse-double-compile passed, 5/5
consecutive clean audits — see `../docs/K_DDC_RESULT.md`), this snapshot preserves
the Python reference so it can never be lost. Keep it for:
- **Backup** — full disaster recovery of the entire repo from one file.
- **Audit** — anyone can restore this exact state and re-run the whole trust chain.
- **Re-DDC** — the Python reference is the diverse second route of the
  diverse-double-compile; retaining it lets the DDC be re-run against future
  versions of `kovc.hx` even after Python is removed from the live tree.

**What it contains** (verified by a test-restore on 2026-05-31): the from-raw-binary
ladder (`stage0/`, incl. the hand-authored `hex0` root), the `helixc-bootstrap`
seed (`seed.c`), the frozen Helix compiler sources (`helixc/bootstrap/*.hx`), the
Python reference compiler + tests (`helixc/`), the DDC harnesses + proof
(`stage0/helixc-bootstrap/ddc_check.py`, `ddc_battery.py`, `docs/K_DDC_RESULT.md`),
and all git history + tags.

### Restore / verify

```bash
# verify integrity against the checksum above
sha256sum backups/kovostov-native-v0-pre-k4-8d593da.bundle

# git's own completeness check
git bundle verify backups/kovostov-native-v0-pre-k4-8d593da.bundle

# full restore into a fresh directory (no network needed)
git clone backups/kovostov-native-v0-pre-k4-8d593da.bundle /path/to/restore
git -C /path/to/restore checkout v0-pre-k4-full-with-python
```

> Note: the durable backup is the **`v0-pre-k4-full-with-python` git tag** (in
> history, pushed to origin). The `.bundle` here is the offline, single-file copy.
