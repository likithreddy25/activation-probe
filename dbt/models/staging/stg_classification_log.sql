-- Cleans the raw event table written by the FastAPI service: trims text,
-- normalizes blank categories to null, and derives a date bucket used by
-- every downstream mart.

with source as (
    select * from {{ source('activation_probe_app', 'classification_log') }}
)

select
    id,
    trim(prompt) as prompt,
    score,
    flagged,
    nullif(trim(category), '') as category,
    model_layer_used,
    model_name,
    created_at,
    (created_at at time zone 'UTC')::date as request_date
from source
where prompt is not null
  and trim(prompt) != ''
