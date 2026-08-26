-- dbt singular test: fails if any row returned. flag_rate is a ratio and
-- must stay within [0, 1]; anything outside that means the rollup math broke.
select *
from {{ ref('mart_daily_flag_rate') }}
where flag_rate < 0 or flag_rate > 1
