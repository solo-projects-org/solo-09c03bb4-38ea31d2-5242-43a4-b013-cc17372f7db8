"""Revision, job and result state transitions.

Invariants enforced here:
* each upload is accepted as a new immutable revision and becomes active
  atomically; all results that depend on the replaced revision are invalidated
  in the same transaction;
* uploading a network revision revokes every earlier quality check, and no
  trajectory can be claimed or published for a revision whose latest quality
  check is not ``passed``;
* opening a quality check immediately stops trajectory supply for that
  revision (results invalidated, matching jobs retired); a failed check stays
  pending review and can be retried, a rejected check lists its defects;
* a worker may only publish a result if its network revision, points revision
  and radius are still the active ones at publish time;
* a job or quality check interrupted by a restart or an exception returns to
  ``pending``, so recomputation is attempted again rather than silently lost.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
import uuid
from datetime import datetime, timedelta, timezone

from . import db
from .matching import (
    Segment,
    TrackPoint,
    compress_segment_sequence,
    match_points,
)
from .parsing import ParsedNetwork, ParsedPoints
from .quality import run_quality_check


class SourceChanged(RuntimeError):
    """Raised when the active sources changed while a job was computing."""


class QualityCheckRequired(RuntimeError):
    """Raised when publishing is blocked by the road-network quality gate."""


class QualityCheckChanged(RuntimeError):
    """Raised when a newer quality check superseded the running one."""


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _next_revision_no(conn: sqlite3.Connection, kind: str) -> int:
    table = "network_revisions" if kind == "network" else "points_revisions"
    row = conn.execute(f"SELECT COALESCE(MAX(revision_no), 0) AS n FROM {table}")
    return int(row.fetchone()["n"]) + 1


def _invalidate_results_for_source(
    conn: sqlite3.Connection, kind: str, reason: str
) -> int:
    column = (
        "network_revision_id" if kind == "network" else "points_revision_id"
    )
    active = db.get_active_revision_id(conn, kind)
    if active is None:
        return 0
    cur = conn.execute(
        f"""
        UPDATE results SET invalidated_at = ?, invalidated_reason = ?
        WHERE invalidated_at IS NULL AND {column} = ?
        """,
        (db.utc_now(), reason, active),
    )
    return cur.rowcount


def _stale_pending_jobs(conn: sqlite3.Connection, kind: str, reason: str) -> None:
    column = (
        "network_revision_id" if kind == "network" else "points_revision_id"
    )
    conn.execute(
        f"""
        UPDATE jobs SET state = 'stale', finished_at = ?, error = ?
        WHERE state IN ('pending', 'running')
          AND {column} <> COALESCE(
              (SELECT revision_id FROM active_sources WHERE kind = ?), -1
          )
        """,
        (db.utc_now(), reason, kind),
    )


def accept_network_revision(
    conn: sqlite3.Connection, parsed: ParsedNetwork, filename: str,
    note: str, raw: bytes,
) -> int:
    now = db.utc_now()
    with db.write_txn(conn):
        revision_no = _next_revision_no(conn, "network")
        cur = conn.execute(
            """
            INSERT INTO network_revisions(
                revision_no, filename, content_sha256, segment_count,
                note, created_at
            ) VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                revision_no,
                filename,
                sha256_bytes(raw),
                len(parsed.segments),
                note,
                now,
            ),
        )
        revision_id = int(cur.lastrowid)
        conn.executemany(
            """
            INSERT INTO network_segments(
                revision_id, seg_index, segment_id, ax, ay, bx, by
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            [
                (revision_id, idx, seg.segment_id, seg.ax, seg.ay, seg.bx, seg.by)
                for idx, seg in enumerate(parsed.segments)
            ],
        )
        _invalidate_results_for_source(conn, "network", "NETWORK_REPLACED")
        conn.execute(
            """
            INSERT INTO active_sources(kind, revision_id) VALUES ('network', ?)
            ON CONFLICT(kind) DO UPDATE SET revision_id = excluded.revision_id
            """,
            (revision_id,),
        )
        _revoke_quality_checks(conn, revision_id)
        _stale_pending_jobs(conn, "network", "NETWORK_REPLACED")
        _enqueue_current_job(conn)
    return revision_id


def accept_points_revision(
    conn: sqlite3.Connection, parsed: ParsedPoints, filename: str,
    note: str, raw: bytes,
) -> int:
    now = db.utc_now()
    with db.write_txn(conn):
        revision_no = _next_revision_no(conn, "points")
        cur = conn.execute(
            """
            INSERT INTO points_revisions(
                revision_no, filename, content_sha256, point_count,
                note, created_at
            ) VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                revision_no,
                filename,
                sha256_bytes(raw),
                len(parsed.points),
                note,
                now,
            ),
        )
        revision_id = int(cur.lastrowid)
        conn.executemany(
            """
            INSERT INTO track_points(revision_id, seq, ts, x, y)
            VALUES (?, ?, ?, ?, ?)
            """,
            [
                (revision_id, p.seq, p.ts, p.x, p.y)
                for p in parsed.points
            ],
        )
        _invalidate_results_for_source(conn, "points", "POINTS_REPLACED")
        conn.execute(
            """
            INSERT INTO active_sources(kind, revision_id) VALUES ('points', ?)
            ON CONFLICT(kind) DO UPDATE SET revision_id = excluded.revision_id
            """,
            (revision_id,),
        )
        _stale_pending_jobs(conn, "points", "POINTS_REPLACED")
        _enqueue_current_job(conn)
    return revision_id


def set_radius(conn: sqlite3.Connection, radius: float) -> None:
    with db.write_txn(conn):
        old = db.get_radius(conn)
        if abs(old - radius) <= 1e-12:
            return
        conn.execute(
            "UPDATE settings SET value = ? WHERE key = 'radius'",
            (repr(radius),),
        )
        conn.execute(
            """
            UPDATE results SET invalidated_at = ?, invalidated_reason = 'RADIUS_CHANGED'
            WHERE invalidated_at IS NULL
            """,
            (db.utc_now(),),
        )
        conn.execute(
            """
            UPDATE jobs SET state = 'stale', finished_at = ?, error = 'RADIUS_CHANGED'
            WHERE state IN ('pending', 'running') AND radius <> ?
            """,
            (db.utc_now(), radius),
        )
        _enqueue_current_job(conn)


def _revoke_quality_checks(
    conn: sqlite3.Connection, new_revision_id: int
) -> int:
    """Retire every quality check of previous network revisions.

    Uploading a new network revision cancels earlier quality checks outright:
    their conclusions never apply to the new geometry and must not be used to
    publish a trajectory.
    """
    cur = conn.execute(
        """
        UPDATE quality_checks
        SET state = 'revoked', claim_token = NULL,
            finished_at = COALESCE(finished_at, ?),
            error = COALESCE(error, 'NETWORK_REPLACED')
        WHERE network_revision_id <> ? AND state IN (
            'pending', 'running', 'passed', 'rejected', 'failed'
        )
        """,
        (db.utc_now(), new_revision_id),
    )
    return cur.rowcount


def latest_quality_check(
    conn: sqlite3.Connection, network_revision_id: int | None = None
):
    if network_revision_id is None:
        network_revision_id = db.get_active_revision_id(conn, "network")
    if network_revision_id is None:
        return None
    return conn.execute(
        """
        SELECT * FROM quality_checks
        WHERE network_revision_id = ?
        ORDER BY id DESC LIMIT 1
        """,
        (network_revision_id,),
    ).fetchone()


def network_quality_passed(conn: sqlite3.Connection, network_revision_id: int) -> bool:
    row = latest_quality_check(conn, network_revision_id)
    return row is not None and row["state"] == "passed"


def _stale_jobs_without_passed_qc(
    conn: sqlite3.Connection, network_revision_id: int, reason: str
) -> None:
    """Retire matching jobs whose network revision has no passing QC."""
    if network_quality_passed(conn, network_revision_id):
        return
    conn.execute(
        """
        UPDATE jobs SET state = 'stale', finished_at = ?, error = ?
        WHERE state IN ('pending', 'running')
          AND network_revision_id = ?
        """,
        (db.utc_now(), reason, network_revision_id),
    )


def open_quality_check(conn: sqlite3.Connection) -> int:
    """Start (or reuse) a QC run for the active network revision.

    Starting QC immediately stops trajectory supply for that revision: current
    results are invalidated and matching jobs without a passing check are
    retired. A pending check is reused so retries share one record.
    """
    now = db.utc_now()
    with db.write_txn(conn):
        network_id = db.get_active_revision_id(conn, "network")
        if network_id is None:
            raise ValueError("请先上传路网修订")
        reusable = conn.execute(
            """
            SELECT * FROM quality_checks
            WHERE network_revision_id = ? AND state IN ('pending', 'running')
            ORDER BY id DESC LIMIT 1
            """,
            (network_id,),
        ).fetchone()
        if reusable:
            check_id = int(reusable["id"])
            # A manual retry clears any error back-off so the worker can claim
            # it immediately (the check still stays "pending review").
            if reusable["next_attempt_at"] is not None:
                conn.execute(
                    "UPDATE quality_checks SET next_attempt_at = NULL, error = NULL"
                    " WHERE id = ?",
                    (check_id,),
                )
        else:
            cur = conn.execute(
                """
                INSERT INTO quality_checks(network_revision_id, state, created_at)
                VALUES (?, 'pending', ?)
                """,
                (network_id, now),
            )
            check_id = int(cur.lastrowid)
        conn.execute(
            """
            UPDATE results SET invalidated_at = ?,
                invalidated_reason = 'NETWORK_QC_OPEN'
            WHERE invalidated_at IS NULL AND network_revision_id = ?
            """,
            (now, network_id),
        )
        _stale_jobs_without_passed_qc(conn, network_id, "NETWORK_QC_OPEN")
    return check_id


def claim_quality_check(conn: sqlite3.Connection) -> sqlite3.Row | None:
    with db.write_txn(conn):
        row = conn.execute(
            """
            SELECT * FROM quality_checks
            WHERE state = 'pending'
              AND (next_attempt_at IS NULL OR next_attempt_at <= ?)
            ORDER BY id LIMIT 1
            """,
            (db.utc_now(),),
        ).fetchone()
        if row is None:
            return None
        token = uuid.uuid4().hex
        conn.execute(
            """
            UPDATE quality_checks SET state = 'running', claim_token = ?,
                started_at = ?, error = NULL
            WHERE id = ? AND state = 'pending'
            """,
            (token, db.utc_now(), row["id"]),
        )
        return conn.execute(
            "SELECT * FROM quality_checks WHERE id = ?", (row["id"],)
        ).fetchone()


def reset_running_quality_checks(conn: sqlite3.Connection) -> int:
    """A restarted process inherits no live QC worker: retry interrupted runs."""
    with db.write_txn(conn):
        cur = conn.execute(
            """
            UPDATE quality_checks SET state = 'pending', claim_token = NULL,
                error = COALESCE(error, 'RESTART_INTERRUPTED')
            WHERE state = 'running'
            """
        )
        return cur.rowcount


def fail_quality_check(
    conn: sqlite3.Connection, check_id: int, token: str, message: str,
    attempts: int,
) -> None:
    # Computation errors keep the check "pending review" and allow retries,
    # mirroring job back-off behaviour.
    delay_seconds = min(60, 2 ** min(attempts, 6))
    next_attempt = (
        datetime.now(timezone.utc) + timedelta(seconds=delay_seconds)
    ).isoformat(timespec="seconds").replace("+00:00", "Z")
    with db.write_txn(conn):
        conn.execute(
            """
            UPDATE quality_checks SET state = 'pending', claim_token = NULL,
                attempts = attempts + 1, error = ?, next_attempt_at = ?
            WHERE id = ? AND state = 'running' AND claim_token = ?
            """,
            (message[:2000], next_attempt, check_id, token),
        )


def _finish_quality_check(
    conn: sqlite3.Connection, check: sqlite3.Row, findings: list[dict]
) -> None:
    with db.write_txn(conn):
        network_id = db.get_active_revision_id(conn, "network")
        if (
            network_id != check["network_revision_id"]
            or latest_quality_check(conn, check["network_revision_id"])["id"]
            != check["id"]
        ):
            # Raising inside the transaction would roll the revocation back,
            # so just signal the caller; the row is revoked separately below.
            raise QualityCheckChanged("质检结论已过期，拒绝写入")
        state = "rejected" if findings else "passed"
        conn.execute(
            """
            UPDATE quality_checks SET state = ?, claim_token = NULL,
                finding_count = ?, findings_json = ?, error = NULL,
                finished_at = ?, next_attempt_at = NULL
            WHERE id = ? AND state = 'running' AND claim_token = ?
            """,
            (
                state,
                len(findings),
                json.dumps(findings, ensure_ascii=False),
                db.utc_now(),
                check["id"],
                check["claim_token"],
            ),
        )
        if state == "passed":
            _enqueue_current_job(conn)
        else:
            # Defects mean the revision must not supply trajectories: retire
            # any matching work that was still queued for it.
            _stale_jobs_without_passed_qc(
                conn, check["network_revision_id"], "NETWORK_QC_REJECTED"
            )


def _revoke_running_quality_check(
    conn: sqlite3.Connection, check: sqlite3.Row
) -> None:
    with db.write_txn(conn):
        conn.execute(
            """
            UPDATE quality_checks SET state = 'revoked', claim_token = NULL,
                error = COALESCE(error, 'NETWORK_REPLACED'),
                finished_at = ?
            WHERE id = ? AND state = 'running' AND claim_token = ?
            """,
            (db.utc_now(), check["id"], check["claim_token"]),
        )


def execute_quality_check(
    conn: sqlite3.Connection, check: sqlite3.Row
) -> str | None:
    """Run one claimed QC run. Returns the final state, or None if revoked."""
    token = check["claim_token"]
    if not token:
        raise RuntimeError("quality check has no claim token")
    segments = _load_network(conn, check["network_revision_id"])
    try:
        findings = run_quality_check(segments)
        _finish_quality_check(conn, check, findings)
    except QualityCheckChanged:
        _revoke_running_quality_check(conn, check)
        return None
    return "rejected" if findings else "passed"


def _enqueue_current_job(conn: sqlite3.Connection) -> int | None:
    """Ensure one pending job exists for the active sources and radius."""
    network_id = db.get_active_revision_id(conn, "network")
    points_id = db.get_active_revision_id(conn, "points")
    if network_id is None or points_id is None:
        return None
    radius = db.get_radius(conn)
    existing = conn.execute(
        """
        SELECT id FROM jobs
        WHERE network_revision_id = ? AND points_revision_id = ?
          AND radius = ? AND state IN ('pending', 'running')
        ORDER BY id LIMIT 1
        """,
        (network_id, points_id, radius),
    ).fetchone()
    if existing:
        return int(existing["id"])
    cur = conn.execute(
        """
        INSERT INTO jobs(
            network_revision_id, points_revision_id, radius, state, created_at
        ) VALUES (?, ?, ?, 'pending', ?)
        """,
        (network_id, points_id, radius, db.utc_now()),
    )
    return int(cur.lastrowid)


def enqueue_recompute(conn: sqlite3.Connection) -> int | None:
    with db.write_txn(conn):
        return _enqueue_current_job(conn)


def claim_job(conn: sqlite3.Connection) -> sqlite3.Row | None:
    with db.write_txn(conn):
        row = conn.execute(
            """
            SELECT j.* FROM jobs j
            WHERE j.state = 'pending'
              AND (j.next_attempt_at IS NULL OR j.next_attempt_at <= ?)
              AND COALESCE((
                  SELECT qc.state FROM quality_checks qc
                  WHERE qc.network_revision_id = j.network_revision_id
                  ORDER BY qc.id DESC LIMIT 1
              ), '') = 'passed'
            ORDER BY j.id LIMIT 1
            """,
            (db.utc_now(),),
        ).fetchone()
        if row is None:
            return None
        token = uuid.uuid4().hex
        conn.execute(
            """
            UPDATE jobs SET state = 'running', claim_token = ?,
                started_at = ?, error = NULL
            WHERE id = ? AND state = 'pending'
            """,
            (token, db.utc_now(), row["id"]),
        )
        claimed = conn.execute(
            "SELECT * FROM jobs WHERE id = ?", (row["id"],)
        ).fetchone()
        return claimed


def reset_running_jobs(conn: sqlite3.Connection) -> int:
    """On process start, running jobs belong to a dead worker: retry them."""
    with db.write_txn(conn):
        cur = conn.execute(
            """
            UPDATE jobs SET state = 'pending', claim_token = NULL,
                error = COALESCE(error, 'RESTART_INTERRUPTED')
            WHERE state = 'running'
            """
        )
        return cur.rowcount


def fail_job(
    conn: sqlite3.Connection, job_id: int, token: str, message: str,
    attempts: int,
) -> None:
    # Keep failed work pending so a later worker or restart retries it, but
    # back off briefly to avoid a hot loop on a persistent error.
    delay_seconds = min(60, 2 ** min(attempts, 6))
    next_attempt = (
        datetime.now(timezone.utc) + timedelta(seconds=delay_seconds)
    ).isoformat(timespec="seconds").replace("+00:00", "Z")
    with db.write_txn(conn):
        conn.execute(
            """
            UPDATE jobs SET state = 'pending', claim_token = NULL,
                attempts = attempts + 1, error = ?, next_attempt_at = ?
            WHERE id = ? AND state = 'running' AND claim_token = ?
            """,
            (message[:2000], next_attempt, job_id, token),
        )


def mark_stale(conn: sqlite3.Connection, job_id: int, token: str, reason: str) -> None:
    with db.write_txn(conn):
        conn.execute(
            """
            UPDATE jobs SET state = 'stale', claim_token = NULL,
                error = ?, finished_at = ?
            WHERE id = ? AND state = 'running' AND claim_token = ?
            """,
            (reason, db.utc_now(), job_id, token),
        )


def _load_network(conn: sqlite3.Connection, revision_id: int) -> list[Segment]:
    rows = conn.execute(
        """
        SELECT segment_id, ax, ay, bx, by FROM network_segments
        WHERE revision_id = ? ORDER BY seg_index
        """,
        (revision_id,),
    ).fetchall()
    return [
        Segment(r["segment_id"], r["ax"], r["ay"], r["bx"], r["by"])
        for r in rows
    ]


def _load_points(conn: sqlite3.Connection, revision_id: int) -> list[TrackPoint]:
    rows = conn.execute(
        """
        SELECT seq, ts, x, y FROM track_points
        WHERE revision_id = ? ORDER BY seq
        """,
        (revision_id,),
    ).fetchall()
    return [
        TrackPoint(r["seq"], r["ts"], r["x"], r["y"]) for r in rows
    ]


def _publish(
    conn: sqlite3.Connection, job: sqlite3.Row, token: str, outcome,
    segments: list[Segment],
) -> int:
    with db.write_txn(conn):
        network_id = db.get_active_revision_id(conn, "network")
        points_id = db.get_active_revision_id(conn, "points")
        radius = db.get_radius(conn)
        if (
            network_id != job["network_revision_id"]
            or points_id != job["points_revision_id"]
            or abs(radius - job["radius"]) > 1e-12
        ):
            raise SourceChanged("来源已换版，拒绝发布旧结果")
        if not network_quality_passed(conn, network_id):
            # Defence in depth: claim time already filters these out, but a QC
            # may have been (re)opened while the job was computing. Stale
            # conclusions must never be released.
            conn.execute(
                """
                UPDATE jobs SET state = 'stale', claim_token = NULL,
                    error = 'NETWORK_QC_REQUIRED', finished_at = ?
                WHERE id = ? AND state = 'running' AND claim_token = ?
                """,
                (db.utc_now(), job["id"], token),
            )
            raise QualityCheckRequired("路网未通过质检，拒绝发布轨迹")

        # A newer job for the same sources may have superseded this one while
        # it was computing. Publishing it is still correct, but avoid stacking
        # duplicate current results.
        conn.execute(
            """
            UPDATE results SET invalidated_at = ?, invalidated_reason = 'SUPERSEDED'
            WHERE invalidated_at IS NULL
              AND network_revision_id = ? AND points_revision_id = ?
              AND radius = ?
            """,
            (
                db.utc_now(),
                job["network_revision_id"],
                job["points_revision_id"],
                job["radius"],
            ),
        )
        segment_by_id = {s.segment_id: s for s in segments}
        assignments = []
        for item in outcome.assignments:
            seg = segment_by_id[item["segment_id"]]
            assignments.append(
                {
                    "point_seq": item["point_seq"],
                    "segment_id": item["segment_id"],
                    "distance": item["distance"],
                    "segment_from": [seg.ax, seg.ay],
                    "segment_to": [seg.bx, seg.by],
                }
            )
        diagnostics = dict(outcome.diagnostics or {})
        diagnostics["candidate_totals"] = outcome.candidate_totals
        cur = conn.execute(
            """
            INSERT INTO results(
                job_id, network_revision_id, points_revision_id, radius,
                status, reason, total_distance, path_count,
                assignments_json, candidates_json, diagnostics_json,
                published_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                job["id"],
                job["network_revision_id"],
                job["points_revision_id"],
                job["radius"],
                outcome.status,
                outcome.reason,
                outcome.total_distance,
                outcome.path_count,
                json.dumps(assignments, ensure_ascii=False),
                json.dumps(outcome.candidates, ensure_ascii=False),
                json.dumps(diagnostics, ensure_ascii=False),
                db.utc_now(),
            ),
        )
        result_id = int(cur.lastrowid)
        conn.execute(
            """
            UPDATE jobs SET state = 'done', claim_token = NULL,
                finished_at = ?, error = NULL
            WHERE id = ? AND state = 'running' AND claim_token = ?
            """,
            (db.utc_now(), job["id"], token),
        )
        return result_id


def execute_job(conn: sqlite3.Connection, job: sqlite3.Row) -> int | None:
    """Run one claimed job. Returns the published result id, if any."""
    token = job["claim_token"]
    if not token:
        raise RuntimeError("job has no claim token")
    segments = _load_network(conn, job["network_revision_id"])
    points = _load_points(conn, job["points_revision_id"])
    try:
        outcome = match_points(segments, points, float(job["radius"]))
        return _publish(conn, job, token, outcome, segments)
    except SourceChanged:
        mark_stale(conn, job["id"], token, "SOURCE_CHANGED_DURING_RECOMPUTE")
        return None
    except QualityCheckRequired:
        # The in-transition stale update was rolled back with the guard
        # exception; retire the job in its own transaction so it never hangs.
        mark_stale(conn, job["id"], token, "NETWORK_QC_REQUIRED")
        return None


def current_result(conn: sqlite3.Connection):
    network_id = db.get_active_revision_id(conn, "network")
    points_id = db.get_active_revision_id(conn, "points")
    if network_id is None or points_id is None:
        return None
    radius = db.get_radius(conn)
    return conn.execute(
        """
        SELECT r.* FROM results r
        WHERE r.invalidated_at IS NULL
          AND r.network_revision_id = ? AND r.points_revision_id = ?
          AND r.radius = ?
          AND COALESCE((
              SELECT qc.state FROM quality_checks qc
              WHERE qc.network_revision_id = r.network_revision_id
              ORDER BY qc.id DESC LIMIT 1
          ), '') = 'passed'
        ORDER BY r.id DESC LIMIT 1
        """,
        (network_id, points_id, radius),
    ).fetchone()


def latest_job(conn: sqlite3.Connection):
    return conn.execute(
        "SELECT * FROM jobs ORDER BY id DESC LIMIT 1"
    ).fetchone()


def result_for_job(conn: sqlite3.Connection, job_id: int):
    return conn.execute(
        """
        SELECT * FROM results
        WHERE job_id = ? AND invalidated_at IS NULL
        ORDER BY id DESC LIMIT 1
        """,
        (job_id,),
    ).fetchone()


def segment_sequence(assignments: list[dict]) -> list[str]:
    return compress_segment_sequence(
        item["segment_id"] for item in assignments
    )
