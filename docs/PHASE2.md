# Phase 2: dbt models on the Postgres log table

## What this is

A small `dbt-core` project (`dbt/`) that turns the raw `classification_log`
table written by the Phase 1 FastAPI service into three analysis-ready
tables: a daily flag-rate rollup, a per-category breakdown, and a rolling
false-positive-rate trend computed against known ground truth rather than
raw flag counts.

## Why dbt, and why these three marts specifically

A raw event log answers "what happened," not "is the system behaving
correctly." The project brief specifically asked for a false-positive-rate
trend "using labeled eval data to compute this, not just raw flag counts" --
that distinction is the whole point of this phase. A flag-rate rollup on its
own can't tell you if the probe is getting worse, because it has no notion
of *right* or *wrong*, only *flagged* or *not*. `mart_fpr_trend` fixes that
by joining logged requests against a ground-truth reference table
(`labeled_eval_examples`, a dbt seed built from the same JBB/Alpaca splits
used in Phase 1) and computing FPR only over the requests it can verify.

## Project layout

```
dbt/
  dbt_project.yml
  profiles.yml                        # reads connection details from env vars
  seeds/labeled_eval_examples.csv      # ground-truth reference (built by scripts/build_dbt_seed.py)
  models/
    staging/
      sources.yml                     # declares classification_log as a dbt source
      stg_classification_log.sql      # trims text, normalizes category, derives request_date
      schema.yml                      # not_null/unique tests
    marts/
      mart_daily_flag_rate.sql
      mart_category_breakdown.sql
      mart_fpr_trend.sql
      schema.yml
  tests/
    assert_flag_rate_between_0_and_1.sql   # singular test, no dbt_utils dependency
    assert_fpr_between_0_and_1.sql
```

`profiles.yml` reads `POSTGRES_HOST/PORT/USER/PASSWORD/DB` from the
environment (same `.env` the FastAPI service uses, just split into fields)
rather than hardcoding credentials, so it works against the same
docker-compose Postgres instance from Phase 1 without duplicating config.

## Models

- **`stg_classification_log`** (staging, view) -- trims prompt text,
  converts blank categories to `null`, derives `request_date` from
  `created_at`. Every downstream model reads from here, not the raw table
  directly.
- **`mart_daily_flag_rate`** (mart, table) -- `request_date`, request count,
  flagged count, flag rate. All traffic, labeled or not.
- **`mart_category_breakdown`** -- per-category request count, flag count,
  flag rate, mean score. `category` is only populated on requests that
  supplied one (real production traffic mostly won't); unlabeled requests
  roll up into `'unknown'`.
- **`mart_fpr_trend`** -- inner-joins the log against `labeled_eval_examples`
  on exact prompt text, keeps only rows where the ground-truth label is
  benign, and reports both the daily FPR and a rolling 7-day FPR (window
  function over `request_date`). Unlabeled traffic is excluded, not assumed
  benign -- an important distinction from `mart_daily_flag_rate`.

## Seeding the log table with real (not fabricated) data

The Postgres log table from Phase 1 only had two rows in it (from manual
curl testing). To give these marts something non-trivial to compute over,
`scripts/seed_traffic.py` replays the full labeled dataset (train + test +
ood_benign, 400 examples) through the **actual trained probe**, and inserts
the real resulting scores/flags into `classification_log` with `created_at`
backdated across the last 14 days (uniformly random, seeded for
reproducibility). To be explicit about what's real and what isn't: every
score and flag is a genuine probe output on a genuine prompt -- nothing
about the classification result is synthetic. The only synthetic part is
the timestamp, backdated purely so the daily/rolling marts have more than
one day of data to show. This is a stand-in for real production traffic,
which doesn't exist yet for a portfolio project; it is not offered as
evidence of the probe's accuracy (that evidence is Phase 1's held-out eval,
which never touches this replayed data's timestamps).

## Actual output (this run)

**`mart_category_breakdown`** (11 categories: 10 JBB categories at 20
examples each -- 10 harmful + 10 benign per category from train+test
combined -- plus the 200-example OOD Alpaca benign set):

| category | n_requests | flag_rate | avg_score |
|---|---|---|---|
| ood_benign_alpaca | 200 | 0.030 | 0.041 |
| Privacy | 20 | 0.550 | 0.564 |
| Fraud/Deception | 20 | 0.550 | 0.559 |
| Physical harm | 20 | 0.550 | 0.541 |
| Harassment/Discrimination | 20 | 0.550 | 0.539 |
| Economic harm | 20 | 0.550 | 0.527 |
| Disinformation | 20 | 0.500 | 0.495 |
| Malware/Hacking | 20 | 0.500 | 0.505 |
| Expert advice | 20 | 0.500 | 0.493 |
| Sexual/Adult content | 20 | 0.450 | 0.478 |
| Government decision-making | 20 | 0.450 | 0.472 |

The ~0.45-0.55 flag rates on the 10 JBB categories make sense: each mixes
10 harmful + 10 benign examples, so a well-calibrated probe should flag
roughly half. The 3% flag rate on the 200-example OOD Alpaca set (100%
benign) matches Phase 1's held-out OOD false-positive-rate number exactly.

**`mart_fpr_trend`** (excerpt, full table has 15 days from the replay
window): daily FPR bounces between 0% and 12.5% depending on which handful
of labeled-benign examples happened to land on a given simulated day (small
daily samples, real sampling noise); the rolling 7-day FPR is much steadier,
sitting in the 3-6% band across the whole window. No drift is visible or
expected here -- it's the same static probe scoring the same fixed dataset,
just replayed with different timestamps. A real drift signal would need to
come from Phase 3's nightly re-evaluation against fresh data, not from this
replay.

## Tests

22 dbt tests, all passing: built-in `not_null`/`unique` checks on the
source table, the staging model, and every mart's primary key column, plus
two singular tests (`assert_flag_rate_between_0_and_1`,
`assert_fpr_between_0_and_1`) that fail if any computed rate falls outside
`[0, 1]` -- written by hand rather than pulling in the `dbt_utils` package
for two simple range checks.

## Reproducing this phase

```bash
conda activate activation-probe
docker compose up -d                        # Postgres from Phase 1
python scripts/build_dbt_seed.py            # builds dbt/seeds/labeled_eval_examples.csv
python scripts/seed_traffic.py --days 14 --clear   # replays labeled data into classification_log

cd dbt
set -a && source ../.env && set +a          # exposes POSTGRES_* to dbt's profiles.yml
DBT_PROFILES_DIR=. dbt seed
DBT_PROFILES_DIR=. dbt run
DBT_PROFILES_DIR=. dbt test
```

Query the results directly, e.g.:

```bash
docker exec activation-probe-postgres psql -U probe -d activation_probe \
  -c "select * from analytics_marts.mart_fpr_trend order by request_date;"
```

## A bug this phase found in Phase 1

While wiring dbt's `profiles.yml` to read `.env`, `scripts/seed_traffic.py`
failed to connect to Postgres on the first run: `db.py` read `DATABASE_URL`
from the environment at **module import time**, but `load_dotenv()` was
only called later in the calling script -- so it silently fell back to the
default `localhost:5432` (the wrong port; the docker-compose Postgres is on
5433) instead of raising a clear "not configured" error. Fixed by moving
`load_dotenv()` into `db.py` itself, so the correct URL is loaded regardless
of what order a caller imports things in. This also means the FastAPI
service no longer depends on `.env` being sourced into the shell manually
before `uvicorn` starts, which Phase 1's README technically required and
this phase's testing exposed.
