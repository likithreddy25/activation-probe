-- Daily volume + flag rate across all traffic, labeled or not.

with daily as (
    select
        request_date,
        count(*) as n_requests,
        sum(case when flagged then 1 else 0 end) as n_flagged
    from {{ ref('stg_classification_log') }}
    group by 1
)

select
    request_date,
    n_requests,
    n_flagged,
    round(n_flagged::numeric / nullif(n_requests, 0), 4) as flag_rate
from daily
order by request_date
