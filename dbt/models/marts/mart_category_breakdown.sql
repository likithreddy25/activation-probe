-- Per-category volume, flag rate, and mean score. `category` is only
-- populated by the service when a caller supplies a known label (e.g. eval
-- replay traffic); real anonymous production requests fall into 'unknown'.

select
    coalesce(category, 'unknown') as category,
    count(*) as n_requests,
    sum(case when flagged then 1 else 0 end) as n_flagged,
    round(sum(case when flagged then 1 else 0 end)::numeric / nullif(count(*), 0), 4) as flag_rate,
    round(avg(score)::numeric, 4) as avg_score
from {{ ref('stg_classification_log') }}
group by 1
order by n_requests desc
