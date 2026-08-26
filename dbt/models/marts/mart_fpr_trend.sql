-- Rolling false-positive-rate trend, computed only over requests we can
-- match to a ground-truth label (via the labeled_eval_examples seed) --
-- this is NOT the same as the raw flag rate in mart_daily_flag_rate, which
-- has no notion of "false" without ground truth. Unmatched (unlabeled)
-- production traffic is intentionally excluded rather than assumed benign.

with matched as (
    select
        l.request_date,
        l.flagged
    from {{ ref('stg_classification_log') }} l
    inner join {{ ref('labeled_eval_examples') }} e
        on trim(l.prompt) = trim(e.prompt)
    where e.label = 0  -- only ground-truth-benign examples inform a false-positive rate
),

daily as (
    select
        request_date,
        count(*) as n_benign_labeled,
        sum(case when flagged then 1 else 0 end) as n_false_positives
    from matched
    group by 1
)

select
    request_date,
    n_benign_labeled,
    n_false_positives,
    round(n_false_positives::numeric / nullif(n_benign_labeled, 0), 4) as daily_fpr,
    round(
        sum(n_false_positives) over (
            order by request_date rows between 6 preceding and current row
        )::numeric
        / nullif(
            sum(n_benign_labeled) over (
                order by request_date rows between 6 preceding and current row
            ),
            0
        ),
        4
    ) as rolling_7d_fpr
from daily
order by request_date
