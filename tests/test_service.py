"""Tests for revision replacement, invalidation and job state management."""

import csv
import io
import json

import pytest

from app import db, service
from app.matching import Segment, TrackPoint
from app.parsing import parse_network, parse_points

from conftest import pass_quality_check, run_job


def network_csv(rows):
    out = io.StringIO()
    writer = csv.writer(out)
    writer.writerow(["路段号", "起点x", "起点y", "终点x", "终点y"])
    writer.writerows(rows)
    return out.getvalue()


def points_csv(rows):
    out = io.StringIO()
    writer = csv.writer(out)
    writer.writerow(["时刻", "x", "y"])
    writer.writerows(rows)
    return out.getvalue()


STRAIGHT_NETWORK = [
    ["S1", 0, 0, 100, 0],
    ["S2", 100, 0, 200, 0],
]


def upload(conn, kind, text, name="data.csv", note=""):
    raw = text.encode("utf-8")
    if kind == "network":
        parsed = parse_network(text)
        return service.accept_network_revision(conn, parsed, name, note, raw)
    parsed = parse_points(text)
    return service.accept_points_revision(conn, parsed, name, note, raw)


def test_accept_network_creates_incrementing_revisions(conn):
    first = upload(conn, "network", network_csv(STRAIGHT_NETWORK))
    second = upload(conn, "network", network_csv(STRAIGHT_NETWORK + [["S3", 200, 0, 300, 0]]))
    rows = conn.execute(
        "SELECT revision_no, segment_count FROM network_revisions ORDER BY id"
    ).fetchall()
    assert [r["revision_no"] for r in rows] == [1, 2]
    assert [r["segment_count"] for r in rows] == [2, 3]
    assert db.get_active_revision_id(conn, "network") == second != first


def test_uploading_points_enqueues_single_pending_job(conn):
    upload(conn, "network", network_csv(STRAIGHT_NETWORK))
    upload(conn, "points", points_csv([["2026-11-18T00:00:00Z", 10, 0]]))
    upload(conn, "points", points_csv([["2026-11-18T00:00:10Z", 50, 0]]))
    jobs = conn.execute("SELECT * FROM jobs ORDER BY id").fetchall()
    # The superseded points revision's job is retired; exactly one pending job
    # remains for the newest pair of active sources.
    assert [j["state"] for j in jobs] == ["stale", "pending"]


def test_publish_then_network_replacement_invalidates_result(conn):
    upload(conn, "network", network_csv(STRAIGHT_NETWORK))
    upload(conn, "points", points_csv([["2026-11-18T00:00:00Z", 10, 1],
                                      ["2026-11-18T00:00:10Z", 150, 1]]))
    pass_quality_check(conn)
    job, result_id = run_job(conn)
    result = service.current_result(conn)
    assert result is not None and result["status"] == "valid"
    assert result["invalidated_at"] is None
    segs = json.loads(result["assignments_json"])
    assert [s["segment_id"] for s in segs] == ["S1", "S2"]

    upload(conn, "network", network_csv(STRAIGHT_NETWORK))
    old = conn.execute("SELECT * FROM results WHERE id = ?", (result_id,)).fetchone()
    assert old["invalidated_at"] is not None
    assert old["invalidated_reason"] == "NETWORK_REPLACED"
    assert service.current_result(conn) is None
    pending = conn.execute(
        "SELECT * FROM jobs WHERE state = 'pending'"
    ).fetchall()
    assert len(pending) == 1


def test_points_replacement_invalidates_result(conn):
    upload(conn, "network", network_csv(STRAIGHT_NETWORK))
    upload(conn, "points", points_csv([["2026-11-18T00:00:00Z", 10, 0]]))
    pass_quality_check(conn)
    run_job(conn)
    assert service.current_result(conn) is not None
    upload(conn, "points", points_csv([["2026-11-18T00:00:00Z", 20, 0]]))
    stale = conn.execute(
        "SELECT * FROM results WHERE invalidated_reason = 'POINTS_REPLACED'"
    ).fetchall()
    assert len(stale) == 1
    assert service.current_result(conn) is None


def test_radius_change_invalidates_and_requeues(conn):
    upload(conn, "network", network_csv(STRAIGHT_NETWORK))
    upload(conn, "points", points_csv([["2026-11-18T00:00:00Z", 10, 9]]))
    pass_quality_check(conn)
    run_job(conn)
    assert service.current_result(conn)["status"] == "valid"

    service.set_radius(conn, 5.0)
    assert service.current_result(conn) is None
    stale = conn.execute(
        "SELECT * FROM results WHERE invalidated_reason = 'RADIUS_CHANGED'"
    ).fetchall()
    assert len(stale) == 1
    # 9 m from the road, 5 m radius -> review result after recompute.
    _, result_id = run_job(conn)
    fresh = conn.execute("SELECT * FROM results WHERE id = ?", (result_id,)).fetchone()
    assert fresh["status"] == "review"
    assert fresh["reason"] == "NO_CANDIDATE"
    assert fresh["radius"] == pytest.approx(5.0)


def test_stale_job_cannot_publish_after_source_change(conn):
    upload(conn, "network", network_csv(STRAIGHT_NETWORK))
    upload(conn, "points", points_csv([["2026-11-18T00:00:00Z", 10, 0]]))
    pass_quality_check(conn)
    job = service.claim_job(conn)
    assert job is not None

    # A new network revision arrives while the job is running.
    upload(conn, "network", network_csv(STRAIGHT_NETWORK))

    result_id = service.execute_job(conn, job)
    assert result_id is None
    row = conn.execute("SELECT * FROM jobs WHERE id = ?", (job["id"],)).fetchone()
    assert row["state"] == "stale"
    # The upload already retired the running job; the publish guard refused it.
    assert row["error"] == "NETWORK_REPLACED"
    assert service.current_result(conn) is None
    # The replacement revision has no passing QC and therefore no claimable job.
    assert service.claim_job(conn) is None
    # Once the new revision passes QC, its own pending job becomes available.
    pass_quality_check(conn)
    pending = conn.execute(
        "SELECT * FROM jobs WHERE state = 'pending'"
    ).fetchall()
    assert len(pending) == 1


def test_radius_change_during_compute_blocks_publish(conn):
    upload(conn, "network", network_csv(STRAIGHT_NETWORK))
    upload(conn, "points", points_csv([["2026-11-18T00:00:00Z", 10, 0]]))
    pass_quality_check(conn)
    job = service.claim_job(conn)
    service.set_radius(conn, 20.0)
    assert service.execute_job(conn, job) is None
    row = conn.execute("SELECT * FROM jobs WHERE id = ?", (job["id"],)).fetchone()
    assert row["state"] == "stale"
    assert service.current_result(conn) is None


def test_job_not_claimable_until_quality_check_passes(conn):
    upload(conn, "network", network_csv(STRAIGHT_NETWORK))
    upload(conn, "points", points_csv([["2026-11-18T00:00:00Z", 10, 0]]))
    # Auto-enqueued job exists but the QC gate keeps it from being claimed.
    assert conn.execute("SELECT COUNT(*) AS n FROM jobs").fetchone()["n"] == 1
    assert service.claim_job(conn) is None
    pass_quality_check(conn)
    job = service.claim_job(conn)
    assert job is not None
    assert service.execute_job(conn, job) is not None


def test_rejected_quality_check_blocks_publication(conn):
    # Two segments cross at an interior point.
    crossing = [["A", 0, 0, 100, 0], ["B", 50, -50, 50, 50]]
    upload(conn, "network", network_csv(crossing))
    upload(conn, "points", points_csv([["2026-11-18T00:00:00Z", 50, 0]]))
    check_id = service.open_quality_check(conn)
    check = service.claim_quality_check(conn)
    state = service.execute_quality_check(conn, check)
    assert state == "rejected"
    row = conn.execute(
        "SELECT * FROM quality_checks WHERE id = ?", (check_id,)
    ).fetchone()
    assert row["finding_count"] == 1
    # The pending matching job was retired; nothing can be published.
    assert service.claim_job(conn) is None
    assert service.current_result(conn) is None
    job_states = {
        j["state"] for j in conn.execute("SELECT state FROM jobs").fetchall()
    }
    assert job_states == {"stale"}


def test_restart_resets_running_job_to_pending(conn):
    upload(conn, "network", network_csv(STRAIGHT_NETWORK))
    upload(conn, "points", points_csv([["2026-11-18T00:00:00Z", 10, 0]]))
    pass_quality_check(conn)
    job = service.claim_job(conn)
    assert job["state"] == "running"
    assert service.reset_running_jobs(conn) == 1
    row = conn.execute("SELECT * FROM jobs WHERE id = ?", (job["id"],)).fetchone()
    assert row["state"] == "pending"
    assert row["claim_token"] is None
    # And it can be picked up again.
    again = service.claim_job(conn)
    assert again["id"] == job["id"]
    assert service.execute_job(conn, again) is not None


def test_failure_returns_job_to_pending_with_backoff(conn):
    upload(conn, "network", network_csv(STRAIGHT_NETWORK))
    upload(conn, "points", points_csv([["2026-11-18T00:00:00Z", 10, 0]]))
    pass_quality_check(conn)
    job = service.claim_job(conn)
    service.fail_job(conn, job["id"], job["claim_token"], "boom", 0)
    row = conn.execute("SELECT * FROM jobs WHERE id = ?", (job["id"],)).fetchone()
    assert row["state"] == "pending"
    assert row["error"] == "boom"
    assert row["attempts"] == 1
    assert row["next_attempt_at"] is not None
    # Backoff means it is not claimable immediately.
    assert service.claim_job(conn) is None


def test_recompute_reuses_pending_job(conn):
    upload(conn, "network", network_csv(STRAIGHT_NETWORK))
    upload(conn, "points", points_csv([["2026-11-18T00:00:00Z", 10, 0]]))
    first = service.enqueue_recompute(conn)
    second = service.enqueue_recompute(conn)
    assert first == second


def test_old_network_segments_remain_for_audit(conn):
    first = upload(conn, "network", network_csv(STRAIGHT_NETWORK))
    upload(conn, "network", network_csv([["NEW", 0, 0, 50, 0]]))
    rows = conn.execute(
        "SELECT * FROM network_segments WHERE revision_id = ?", (first,)
    ).fetchall()
    assert len(rows) == 2


def test_parse_points_normalises_offsets_and_checks_order():
    parsed = parse_points(
        points_csv(
            [
                ["2026-11-18T08:00:00+08:00", 1, 2],
                ["2026-11-18T00:00:05Z", 3, 4],
            ]
        )
    )
    assert parsed.points[0].ts == "2026-11-18T00:00:00Z"
    assert parsed.points[1].ts == "2026-11-18T00:00:05Z"

    with pytest.raises(Exception) as excinfo:
        parse_points(
            points_csv(
                [
                    ["2026-11-18T00:00:10Z", 1, 2],
                    ["2026-11-18T00:00:05Z", 3, 4],
                ]
            )
        )
    assert "升序" in str(excinfo.value)


def _published_valid_result(conn):
    upload(conn, "network", network_csv(STRAIGHT_NETWORK))
    upload(conn, "points", points_csv([["2026-11-18T00:00:00Z", 10, 1],
                                      ["2026-11-18T00:00:10Z", 150, 1]]))
    pass_quality_check(conn)
    _, result_id = run_job(conn)
    assert service.current_result(conn) is not None
    return result_id


def test_opening_quality_check_stops_supply_and_retires_jobs(conn):
    result_id = _published_valid_result(conn)
    check_id = service.open_quality_check(conn)
    # The previously published result immediately stops being supplied.
    assert service.current_result(conn) is None
    row = conn.execute("SELECT * FROM results WHERE id = ?", (result_id,)).fetchone()
    assert row["invalidated_reason"] == "NETWORK_QC_OPEN"
    check = conn.execute(
        "SELECT * FROM quality_checks WHERE id = ?", (check_id,)
    ).fetchone()
    assert check["state"] == "pending"
    # Re-opening while pending reuses the same check instead of stacking rows.
    active_network = db.get_active_revision_id(conn, "network")
    before = conn.execute(
        "SELECT COUNT(*) AS n FROM quality_checks"
        " WHERE network_revision_id = ?",
        (active_network,),
    ).fetchone()["n"]
    assert service.open_quality_check(conn) == check_id
    assert conn.execute(
        "SELECT COUNT(*) AS n FROM quality_checks"
        " WHERE network_revision_id = ?",
        (active_network,),
    ).fetchone()["n"] == before


def test_passing_quality_check_requeues_and_republishes(conn):
    result_id = _published_valid_result(conn)
    service.open_quality_check(conn)
    assert service.current_result(conn) is None
    check = service.claim_quality_check(conn)
    assert service.execute_quality_check(conn, check) == "passed"
    # Passing QC schedules the trajectory recompute for this revision.
    job = service.claim_job(conn)
    assert job is not None
    new_id = service.execute_job(conn, job)
    assert new_id != result_id
    assert service.current_result(conn)["id"] == new_id


def test_quality_check_failure_stays_pending_and_retries(conn):
    upload(conn, "network", network_csv(STRAIGHT_NETWORK))
    check_id = service.open_quality_check(conn)
    check = service.claim_quality_check(conn)
    service.fail_quality_check(
        conn, check["id"], check["claim_token"], "boom", check["attempts"]
    )
    row = conn.execute(
        "SELECT * FROM quality_checks WHERE id = ?", (check_id,)
    ).fetchone()
    assert row["state"] == "pending"
    assert row["error"] == "boom"
    assert row["attempts"] == 1
    # Back-off: not immediately claimable.
    assert service.claim_quality_check(conn) is None
    conn.execute(
        "UPDATE quality_checks SET next_attempt_at = NULL WHERE id = ?",
        (check_id,),
    )
    again = service.claim_quality_check(conn)
    assert again["id"] == check_id
    assert service.execute_quality_check(conn, again) == "passed"


def test_restart_resets_running_quality_check_to_pending(conn):
    upload(conn, "network", network_csv(STRAIGHT_NETWORK))
    check_id = service.open_quality_check(conn)
    check = service.claim_quality_check(conn)
    assert check["state"] == "running"
    assert service.reset_running_quality_checks(conn) == 1
    row = conn.execute(
        "SELECT * FROM quality_checks WHERE id = ?", (check_id,)
    ).fetchone()
    assert row["state"] == "pending"
    assert row["claim_token"] is None
    again = service.claim_quality_check(conn)
    assert again["id"] == check_id
    assert service.execute_quality_check(conn, again) == "passed"


def test_new_network_revision_revokes_old_quality_checks(conn):
    upload(conn, "network", network_csv(STRAIGHT_NETWORK))
    first_check = service.open_quality_check(conn)
    check = service.claim_quality_check(conn)
    service.execute_quality_check(conn, check)
    # A rejected check on a defective revision.
    upload(conn, "network", network_csv([["X", 0, 0, 100, 0]]))
    bad_check = service.open_quality_check(conn)
    bad = service.claim_quality_check(conn)
    service.execute_quality_check(conn, bad)
    # Upload another revision: every earlier check must be revoked.
    upload(conn, "network", network_csv(STRAIGHT_NETWORK))
    rows = conn.execute(
        "SELECT id, state FROM quality_checks ORDER BY id"
    ).fetchall()
    states = {r["id"]: r["state"] for r in rows}
    assert states[first_check] == "revoked"
    assert states[bad_check] == "revoked"
    # The new revision starts with no QC conclusion at all.
    assert service.latest_quality_check(conn) is None
    assert service.open_quality_check(conn) not in states
    latest = service.latest_quality_check(conn)
    assert latest["state"] == "pending"
    # A revoked/passed-but-revoked check can never gate a publication.
    assert service.claim_job(conn) is None


def test_qc_reopened_after_rejection_then_passed_allows_publication(conn):
    crossing = [["A", 0, 0, 100, 0], ["B", 50, -50, 50, 50]]
    upload(conn, "network", network_csv(crossing))
    upload(conn, "points", points_csv([["2026-11-18T00:00:00Z", 50, 0]]))
    first = service.open_quality_check(conn)
    check = service.claim_quality_check(conn)
    assert service.execute_quality_check(conn, check) == "rejected"
    # Corrected network uploaded as a new revision: old check revoked, new one
    # starts from a clean state and, once passed, releases the trajectory.
    upload(conn, "network", network_csv(STRAIGHT_NETWORK))
    check2 = service.claim_quality_check(conn)
    assert check2 is None  # no QC opened for the new revision yet
    second = service.open_quality_check(conn)
    assert second != first
    check2 = service.claim_quality_check(conn)
    assert service.execute_quality_check(conn, check2) == "passed"
    job = service.claim_job(conn)
    assert job is not None
    assert service.execute_job(conn, job) is not None
    assert service.current_result(conn) is not None


def test_reopening_qc_while_job_running_prevents_publication(conn):
    upload(conn, "network", network_csv(STRAIGHT_NETWORK))
    upload(conn, "points", points_csv([["2026-11-18T00:00:00Z", 10, 0]]))
    pass_quality_check(conn)
    job = service.claim_job(conn)
    assert job is not None
    # QC reopened (and not re-passed) while the worker holds the job. Opening
    # the check already retires queued work; publishing must still be refused.
    service.open_quality_check(conn)
    assert service.execute_job(conn, job) is None
    row = conn.execute(
        "SELECT state, error FROM jobs WHERE id = ?", (job["id"],)
    ).fetchone()
    assert row["state"] == "stale"
    assert row["error"] in ("NETWORK_QC_OPEN", "NETWORK_QC_REQUIRED")
    assert service.current_result(conn) is None


def test_quality_check_replaced_while_running_is_revoked_not_stuck(conn):
    # Regression: the replacement signal used to be raised inside the finish
    # transaction, rolling the revocation back and leaving a "running" row.
    upload(conn, "network", network_csv(STRAIGHT_NETWORK))
    check_id = service.open_quality_check(conn)
    check = service.claim_quality_check(conn)
    upload(conn, "network", network_csv([["X", 0, 0, 50, 0]]))
    result = service.execute_quality_check(conn, check)
    assert result is None
    row = conn.execute(
        "SELECT state, claim_token FROM quality_checks WHERE id = ?",
        (check_id,),
    ).fetchone()
    assert row["state"] == "revoked"
    assert row["claim_token"] is None
    # The worker loop must not find the dead check again.
    assert service.claim_quality_check(conn) is None


def test_interleaved_qc_matching_and_corrections_never_use_stale_result(conn):
    # 1) clean revision published.
    result_id = _published_valid_result(conn)
    # 2) QC reopened -> supply stops.
    service.open_quality_check(conn)
    assert service.current_result(conn) is None
    # 3) points correction while QC is still pending: still nothing published.
    upload(conn, "points", points_csv([["2026-11-18T00:00:00Z", 20, 0]]))
    assert service.claim_job(conn) is None
    # 4) QC passes on the same network -> recompute against newest points.
    check = service.claim_quality_check(conn)
    assert service.execute_quality_check(conn, check) == "passed"
    job = service.claim_job(conn)
    new_id = service.execute_job(conn, job)
    assert new_id != result_id
    assert service.current_result(conn)["id"] == new_id
    # 5) radius change invalidates again and requeues.
    service.set_radius(conn, 20.0)
    assert service.current_result(conn) is None
    job2 = service.claim_job(conn)
    rid = service.execute_job(conn, job2)
    assert rid is not None
    assert service.current_result(conn)["radius"] == pytest.approx(20.0)
    # 6) a new network upload revokes QC; publication blocked until re-check.
    upload(conn, "network", network_csv(STRAIGHT_NETWORK))
    assert service.current_result(conn) is None
    assert service.claim_job(conn) is None
    pass_quality_check(conn)
    job3 = service.claim_job(conn)
    assert service.execute_job(conn, job3) is not None
    assert service.current_result(conn) is not None
