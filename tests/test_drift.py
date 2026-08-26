"""Tests for the pure threshold logic in eval/drift.py, using fake
session/row objects instead of a real Postgres connection -- consistent
with the rest of the suite staying fully offline."""

from dataclasses import dataclass
from types import SimpleNamespace

from activation_probe.eval import drift


@dataclass
class FakeEvalRow:
    accuracy: float
    auc: float | None


class FakeQuery:
    def __init__(self, rows):
        self._rows = rows

    def filter(self, *a, **k):
        return self

    def order_by(self, *a, **k):
        return self

    def limit(self, n):
        return FakeQuery(self._rows[:n])

    def all(self):
        return self._rows


class FakeSession:
    def __init__(self, eval_rows=None, flag_rate_rows=None):
        self._eval_rows = eval_rows or []
        self._flag_rate_rows = flag_rate_rows or []
        self.added = []

    def query(self, *a, **k):
        return FakeQuery(self._eval_rows)

    def execute(self, *a, **k):
        return SimpleNamespace(fetchall=lambda: self._flag_rate_rows)

    def add(self, obj):
        self.added.append(obj)

    def commit(self):
        pass


def test_check_eval_drift_no_alert_when_healthy():
    session = FakeSession(eval_rows=[FakeEvalRow(accuracy=0.80, auc=0.90)])
    raised = drift.check_eval_drift(session)
    assert raised == []
    assert session.added == []


def test_check_eval_drift_fires_on_absolute_floor_breach():
    session = FakeSession(eval_rows=[FakeEvalRow(accuracy=0.50, auc=0.55)])
    raised = drift.check_eval_drift(session)
    types = {a.alert_type for a in raised}
    assert "eval_accuracy_drift" in types
    assert "eval_auc_drift" in types
    assert len(session.added) == 2  # one Alert row per raised alert


def test_check_eval_drift_fires_on_relative_rolling_drop():
    # latest run is above the absolute floor but far below the rolling mean
    rows = [FakeEvalRow(accuracy=0.68, auc=0.90)] + [FakeEvalRow(accuracy=0.90, auc=0.95) for _ in range(5)]
    session = FakeSession(eval_rows=rows)
    raised = drift.check_eval_drift(session)
    assert any(a.alert_type == "eval_accuracy_drift" for a in raised)


def test_check_eval_drift_no_history_is_a_noop():
    session = FakeSession(eval_rows=[])
    assert drift.check_eval_drift(session) == []


def test_check_flag_rate_drift_no_alert_when_stable():
    rows = [("2026-08-26", 0.25)] + [("2026-08-2" + str(i), 0.24) for i in range(9)]
    session = FakeSession(flag_rate_rows=rows)
    assert drift.check_flag_rate_drift(session) == []


def test_check_flag_rate_drift_fires_on_deviation():
    rows = [("2026-08-26", 0.90)] + [("2026-08-2" + str(i), 0.20) for i in range(9)]
    session = FakeSession(flag_rate_rows=rows)
    raised = drift.check_flag_rate_drift(session)
    assert len(raised) == 1
    assert raised[0].alert_type == "flag_rate_drift"


def test_check_flag_rate_drift_needs_at_least_two_days():
    session = FakeSession(flag_rate_rows=[("2026-08-26", 0.90)])
    assert drift.check_flag_rate_drift(session) == []
