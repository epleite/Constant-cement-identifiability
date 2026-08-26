# Contributing

This repository is a frozen research-compendium release. Please open an issue
before proposing changes that alter numerical results, model assumptions, data
filters or declared nuisance scales.

For code or documentation changes:

1. create a branch from the tagged release;
2. keep the compact Volve data provenance and source conditions intact;
3. add or update a deterministic verification check;
4. run `python scripts/run_all.py --verify-only`;
5. explain whether the change modifies a manuscript result or only packaging.

Do not commit credentials, proprietary well data, or data derived from sources
whose terms do not permit redistribution.
