"""Drift checks run after the nightly eval rerun and the dbt run. Compares
the latest numbers against both an absolute floor and a rolling baseline of
prior runs, and raises an Alert (a logged Postgres row + a stub webhook
print -- no real webhook endpoint exists for this portfolio project, but the
call site is exactly where a real one would go) for anything that crosses a
threshold.

Two independent signals are checked, matching the project brief:
1. eval accuracy/AUC drift, from eval_history (Phase 3's own nightly reruns)
2. flag-rate drift, from mart_daily_flag_rate (Phase 2's dbt mart)
"""

from __future__ import annotations

from dataclasses import dataclass

from sqlalchemy import text

from activation_probe.service import db

# Absolute floors: below these, the probe is not doing its job regardless of history.
ACCURACY_FLOOR = 0.65
AUC_FLOOR = 0.75

# Relative drift: how far the latest run may fall below the mean of the
# preceding N runs before it's flagged, even if still above the absolute floor.
ROLLING_WINDOW = 7
RELATIVE_ACCURACY_DROP = 0.10

# Flag rate: how far a day's flag rate may deviate (absolute, in flag-rate units)
# from the trailing mean of prior days before it's flagged.
FLAG_RATE_ABS_DEVIATION = 0.15


@dataclass
class AlertRaised:
    alert_type: str
    message: str
    metric_value: float
    threshold: float


def send_alert(session, alert_type: str, message: str, metric_value: float, threshold: float) -> None:
    session.add(
        db.Alert(alert_type=alert_type, message=message, metric_value=metric_value, threshold=threshold)
    )
    session.commit()
    # Stub for a real notification channel (Slack webhook, PagerDuty, email...).
    # Kept as a print rather than an actual HTTP call: no real endpoint exists
    # for a portfolio project, and ground rules for this project rule out
    # fabricating one just to look more "production."
    print(f"[ALERT STUB] would notify: [{alert_type}] {message} (value={metric_value}, threshold={threshold})")


def check_eval_drift(session) -> list[AlertRaised]:
    raised = []
    rows = (
        session.query(db.EvalHistory)
        .filter(db.EvalHistory.split_name == "test")
        .order_by(db.EvalHistory.created_at.desc())
        .limit(ROLLING_WINDOW + 1)
        .all()
    )
    if not rows:
        return raised

    latest = rows[0]
    prior = rows[1:]

    if latest.accuracy < ACCURACY_FLOOR:
        msg = f"eval accuracy {latest.accuracy:.4f} fell below absolute floor {ACCURACY_FLOOR}"
        send_alert(session, "eval_accuracy_drift", msg, latest.accuracy, ACCURACY_FLOOR)
        raised.append(AlertRaised("eval_accuracy_drift", msg, latest.accuracy, ACCURACY_FLOOR))

    if latest.auc is not None and latest.auc < AUC_FLOOR:
        msg = f"eval AUC {latest.auc:.4f} fell below absolute floor {AUC_FLOOR}"
        send_alert(session, "eval_auc_drift", msg, latest.auc, AUC_FLOOR)
        raised.append(AlertRaised("eval_auc_drift", msg, latest.auc, AUC_FLOOR))

    if prior:
        rolling_mean_accuracy = sum(r.accuracy for r in prior) / len(prior)
        drop = rolling_mean_accuracy - latest.accuracy
        if drop > RELATIVE_ACCURACY_DROP:
            msg = (
                f"eval accuracy {latest.accuracy:.4f} dropped {drop:.4f} below the "
                f"{len(prior)}-run rolling mean {rolling_mean_accuracy:.4f}"
            )
            send_alert(session, "eval_accuracy_drift", msg, latest.accuracy, rolling_mean_accuracy - RELATIVE_ACCURACY_DROP)
            raised.append(AlertRaised("eval_accuracy_drift", msg, latest.accuracy, rolling_mean_accuracy))

    return raised


def check_flag_rate_drift(session) -> list[AlertRaised]:
    raised = []
    rows = session.execute(
        text(
            """
            select request_date, flag_rate
            from analytics_marts.mart_daily_flag_rate
            order by request_date desc
            limit :n
            """
        ),
        {"n": ROLLING_WINDOW + 1},
    ).fetchall()

    if len(rows) < 2:
        return raised  # not enough history to define a baseline yet

    latest_date, latest_rate = rows[0]
    prior_rates = [r[1] for r in rows[1:]]
    rolling_mean = sum(prior_rates) / len(prior_rates)
    deviation = abs(latest_rate - rolling_mean)

    if deviation > FLAG_RATE_ABS_DEVIATION:
        msg = (
            f"flag rate on {latest_date} was {latest_rate:.4f}, deviating {deviation:.4f} "
            f"from the {len(prior_rates)}-day rolling mean {rolling_mean:.4f}"
        )
        send_alert(session, "flag_rate_drift", msg, latest_rate, rolling_mean)
        raised.append(AlertRaised("flag_rate_drift", msg, latest_rate, rolling_mean))

    return raised


def check_and_alert() -> list[AlertRaised]:
    db.init_db()
    session_factory = db.get_session_factory()
    with session_factory() as session:
        raised = check_eval_drift(session) + check_flag_rate_drift(session)

    if raised:
        print(f"{len(raised)} alert(s) raised")
    else:
        print("no drift detected")
    return raised


if __name__ == "__main__":
    check_and_alert()
