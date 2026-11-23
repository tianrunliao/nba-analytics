# Test fixtures

The tests that parse real NBA payloads need two fixture folders that are **not** in the
public repository (NBA terms of use restrict redistribution). Without them those tests are
skipped (`needs_raw_subset`, `needs_injury_pdfs` in `tests/conftest.py`).

`raw_subset/` — stats.nba.com payloads trimmed to a handful of games. After
`python -m nba_sos.pipeline fetch`, run

    .venv/bin/python tests/fixtures/make_raw_subset.py

`injury_reports/` — six official injury-report PDFs, downloadable from
`https://ak-static.cms.nba.com/referee/injury/Injury-Report_<name>`:

    2019-04-13_05PM.pdf  2019-11-15_05PM.pdf  2019-12-08_05PM.pdf
    2021-12-01_05PM.pdf  2023-11-10_05PM.pdf  2026-11-17_05_30PM.pdf
