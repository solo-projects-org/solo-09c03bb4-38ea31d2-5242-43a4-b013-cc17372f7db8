"""Background worker loop tests (real thread, temporary SQLite file)."""

import csv
import io
import time

from app import db, service
from app.main import Worker
from app.parsing import parse_network, parse_points


def wait_for(predicate, timeout=10.0):
    deadline = time.time() + timeout
    while time.time() < deadline:
        value = predicate()
        if value:
            return value
        time.sleep(0.05)
    raise AssertionError("condition not met in time")


def pass_qc(path):
    conn = db.connect(path)
    try:
        service.open_quality_check(conn)
        check = service.claim_quality_check(conn)
        assert check is not None
        assert service.execute_quality_check(conn, check) == "passed"
    finally:
        conn.close()


def upload(conn, kind, text):
    raw = text.encode()
    if kind == "network":
        return service.accept_network_revision(
            conn, parse_network(text), "n.csv", "", raw
        )
    return service.accept_points_revision(
        conn, parse_points(text), "p.csv", "", raw
    )


def csv_text(header, rows):
    out = io.StringIO()
    writer = csv.writer(out)
    writer.writerow(header)
    writer.writerows(rows)
    return out.getvalue()


NET = csv_text(
    ["路段号", "起点x", "起点y", "终点x", "终点y"],
    [["A", 0, 0, 100, 0], ["B", 100, 0, 200, 0]],
)
PTS = csv_text(
    ["时刻", "x", "y"],
    [["2026-11-18T00:00:00Z", 20, 0], ["2026-11-18T00:00:10Z", 150, 0]],
)


def test_worker_publishes_result(tmp_path):
    path = tmp_path / "w.sqlite3"
    conn = db.connect(path)
    db.init_db(conn)
    upload(conn, "network", NET)
    upload(conn, "points", PTS)
    conn.close()
    pass_qc(path)

    worker = Worker(str(path))
    worker.start()
    try:
        conn = db.connect(path)
        result = wait_for(lambda: service.current_result(conn))
    finally:
        worker.stop()
        conn.close()
    assert result["status"] == "valid"


def test_worker_runs_quality_check_then_matching(tmp_path):
    path = tmp_path / "w2.sqlite3"
    conn = db.connect(path)
    db.init_db(conn)
    upload(conn, "network", NET)
    upload(conn, "points", PTS)
    service.open_quality_check(conn)
    conn.close()

    worker = Worker(str(path))
    worker.start()
    try:
        conn = db.connect(path)

        def qc_passed():
            row = service.latest_quality_check(conn)
            return row if row and row["state"] == "passed" else None

        check = wait_for(qc_passed)
        assert check["finding_count"] == 0
        result = wait_for(lambda: service.current_result(conn))
    finally:
        worker.stop()
        conn.close()
    assert result["status"] == "valid"


def test_worker_retries_quality_check_after_transient_failure(tmp_path, monkeypatch):
    path = tmp_path / "w3.sqlite3"
    conn = db.connect(path)
    db.init_db(conn)
    upload(conn, "network", NET)
    service.open_quality_check(conn)
    conn.close()

    calls = {"n": 0}
    real_run = service.run_quality_check

    def flaky(segments):
        calls["n"] += 1
        if calls["n"] == 1:
            raise RuntimeError("transient qc error")
        return real_run(segments)

    monkeypatch.setattr(service, "run_quality_check", flaky)
    real_fail = service.fail_quality_check

    def immediate_fail(conn_, check_id, token, message, attempts):
        real_fail(conn_, check_id, token, message, 6)
        conn_.execute(
            "UPDATE quality_checks SET next_attempt_at = NULL WHERE id = ?",
            (check_id,),
        )

    monkeypatch.setattr(service, "fail_quality_check", immediate_fail)

    worker = Worker(str(path))
    worker.start()
    try:
        conn = db.connect(path)

        def qc_passed():
            row = service.latest_quality_check(conn)
            return row if row and row["state"] == "passed" else None

        check = wait_for(qc_passed, timeout=15)
    finally:
        worker.stop()
        conn.close()
    assert calls["n"] >= 2
    assert check["finding_count"] == 0


def test_worker_retries_after_transient_failure(tmp_path, monkeypatch):
    path = tmp_path / "w.sqlite3"
    conn = db.connect(path)
    db.init_db(conn)
    upload(conn, "network", NET)

    calls = {"n": 0}
    real_match = service.match_points

    def flaky(*args, **kwargs):
        calls["n"] += 1
        if calls["n"] == 1:
            raise RuntimeError("transient")
        return real_match(*args, **kwargs)

    monkeypatch.setattr(service, "match_points", flaky)

    # Retry immediately rather than after the real exponential backoff.
    real_fail = service.fail_job

    def immediate_fail(conn_, job_id, token, message, attempts):
        real_fail(conn_, job_id, token, message, 6)
        conn_.execute(
            "UPDATE jobs SET next_attempt_at = NULL WHERE id = ?", (job_id,)
        )

    monkeypatch.setattr(service, "fail_job", immediate_fail)

    upload(conn, "points", PTS)
    conn.close()
    pass_qc(path)

    worker = Worker(str(path))
    worker.start()
    try:
        conn = db.connect(path)
        result = wait_for(lambda: service.current_result(conn), timeout=15)
    finally:
        worker.stop()
        conn.close()
    assert calls["n"] >= 2
    assert result["status"] == "valid"
