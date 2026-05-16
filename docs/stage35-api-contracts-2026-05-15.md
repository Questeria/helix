# Stage 35 API Contract Notes - 2026-05-15

This document is a repo-local audit surface for Stage 35 API-contract wording.
The generated/public website contract source remains `helix_website/api_contracts.ts`.

Current live contract boundaries:

- The production compiler is still Python-hosted `helixc`.
- The live bootstrap root is the 299-byte `hex0` artifact.
- Later bootstrap links and full self-hosting are roadmap targets until they
  are implemented and reproducibly verified.
- Stage 35 live AI/ML APIs are the tested stdlib tensor, neural-network,
  autodiff, reverse-AD, tile, CLI, and PTX surfaces covered by the progress
  ledger and pytest collection.
- Published counts should be refreshed with
  `python -m pytest helixc/tests --collect-only -q -p no:cacheprovider` before
  public release.

Audit rule:

- Public wording must distinguish implemented behavior from target capability.
