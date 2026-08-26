-- Same bounds check for both the daily and rolling FPR figures.
select *
from {{ ref('mart_fpr_trend') }}
where daily_fpr < 0 or daily_fpr > 1
   or rolling_7d_fpr < 0 or rolling_7d_fpr > 1
