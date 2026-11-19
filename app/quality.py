"""Geometric quality checks for a planar road network revision.

A pair of distinct directed segments on the same plane is *defective* when
either:

* the two segments intersect at a point that is an interior point of at least
  one of them (an unsplit intersection / T-junction), or
* they are collinear and share a sub-interval of positive length (an
  overlapping road section).

Contact limited to the two segments' endpoints, with no positive-length
overlap, is the normal "end-to-end join" case and is never reported.

All coordinates are planar metres. Tolerances mirror the matcher:

* endpoint membership is quantised at ``ENDPOINT_TOL`` (1e-6 m) so floating
  point noise cannot turn an endpoint join into an interior intersection;
* parallelism / collinearity use strict 1e-9-scale tolerances;
* an overlap must be longer than ``MIN_OVERLAP_LENGTH`` (1e-9 m) to count.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from math import hypot
from typing import Sequence

from .matching import Segment

# Two points closer than this (metres) are treated as the same endpoint.
ENDPOINT_TOL = 1e-6
# Strict absolute tolerance for collinearity / zero-length comparisons (m).
GEOMETRY_EPS = 1e-9
# A collinear overlap must be strictly longer than this to be a defect.
MIN_OVERLAP_LENGTH = 1e-9
# Relative sin-angle threshold below which two directions count as parallel.
PARALLEL_REL_EPS = 1e-10

FINDING_INTERSECTION = "intersection"
FINDING_OVERLAP = "overlap"


@dataclass(frozen=True)
class _Vec:
    x: float
    y: float


def _sub(a: tuple[float, float], b: tuple[float, float]) -> _Vec:
    return _Vec(a[0] - b[0], a[1] - b[1])


def _cross(u: _Vec, v: _Vec) -> float:
    return u.x * v.y - u.y * v.x


def _dot(u: _Vec, v: _Vec) -> float:
    return u.x * v.x + u.y * v.y


def _point(p: tuple[float, float]) -> list[float]:
    return [round(p[0], 9), round(p[1], 9)]


def _is_interior(point: tuple[float, float], seg: Segment) -> bool:
    """True when *point* is strictly inside *seg* (not one of its endpoints)."""
    start = (seg.ax, seg.ay)
    end = (seg.bx, seg.by)
    near_start = hypot(point[0] - start[0], point[1] - start[1]) <= ENDPOINT_TOL
    near_end = hypot(point[0] - end[0], point[1] - end[1]) <= ENDPOINT_TOL
    return not near_start and not near_end


def _pair_finding(s1: Segment, s2: Segment) -> dict | None:
    """Return one defect finding for a segment pair, or None when clean."""
    a1 = (s1.ax, s1.ay)
    b1 = (s1.bx, s1.by)
    a2 = (s2.ax, s2.ay)
    b2 = (s2.bx, s2.by)
    v1 = _sub(b1, a1)
    v2 = _sub(b2, a2)
    length1 = hypot(v1.x, v1.y)
    length2 = hypot(v2.x, v2.y)
    d = _cross(v1, v2)
    parallel_limit = (
        PARALLEL_REL_EPS * length1 * length2
        + GEOMETRY_EPS * max(length1, length2)
    )

    if abs(d) > parallel_limit:
        # Non-parallel supporting lines: solve a1 + t*v1 = a2 + u*v2.
        w = _sub(a2, a1)
        t = _cross(w, v2) / d
        u = _cross(w, v1) / d
        if t < -GEOMETRY_EPS or t > 1.0 + GEOMETRY_EPS:
            return None
        if u < -GEOMETRY_EPS or u > 1.0 + GEOMETRY_EPS:
            return None
        t = min(1.0, max(0.0, t))
        u = min(1.0, max(0.0, u))
        point = (a1[0] + t * v1.x, a1[1] + t * v1.y)
        interior1 = _is_interior(point, s1)
        interior2 = _is_interior(point, s2)
        if not interior1 and not interior2:
            # Endpoint-to-endpoint contact only: normal join.
            return None
        ids = sorted((s1.segment_id, s2.segment_id))
        interior_of = [
            sid
            for sid in ids
            if (sid == s1.segment_id and interior1)
            or (sid == s2.segment_id and interior2)
        ]
        return {
            "kind": FINDING_INTERSECTION,
            "segment_ids": ids,
            "point": _point(point),
            "interior_of": interior_of,
        }

    # Parallel directions. On the same supporting line? The perpendicular
    # distance from a2 to s1's line tells us.
    w = _sub(a2, a1)
    perpendicular_gap = abs(_cross(w, v1)) / length1
    if perpendicular_gap > GEOMETRY_EPS:
        return None

    # Collinear: project s2's endpoints onto s1's parameter axis.
    length1_sq = length1 * length1
    t_a2 = _dot(w, v1) / length1_sq
    t_b2 = _dot(_sub(b2, a1), v1) / length1_sq
    lo = max(0.0, min(t_a2, t_b2))
    hi = min(1.0, max(t_a2, t_b2))
    if hi < lo - GEOMETRY_EPS:
        return None
    overlap_length = max(0.0, hi - lo) * length1
    if overlap_length <= MIN_OVERLAP_LENGTH:
        # A single shared point (an endpoint join) or no contact at all.
        return None
    p_lo = (a1[0] + lo * v1.x, a1[1] + lo * v1.y)
    p_hi = (a1[0] + hi * v1.x, a1[1] + hi * v1.y)
    return {
        "kind": FINDING_OVERLAP,
        "segment_ids": sorted((s1.segment_id, s2.segment_id)),
        "from": _point(p_lo),
        "to": _point(p_hi),
        "length": round(overlap_length, 9),
    }


def _bboxes_separated(s1: Segment, s2: Segment) -> bool:
    x_gap = max(s1.ax, s1.bx) < min(s2.ax, s2.bx) - GEOMETRY_EPS or max(
        s2.ax, s2.bx
    ) < min(s1.ax, s1.bx) - GEOMETRY_EPS
    y_gap = max(s1.ay, s1.by) < min(s2.ay, s2.by) - GEOMETRY_EPS or max(
        s2.ay, s2.by
    ) < min(s1.ay, s1.by) - GEOMETRY_EPS
    return x_gap or y_gap


def run_quality_check(segments: Sequence[Segment]) -> list[dict]:
    """Check every segment pair; return deterministic, sorted findings."""
    findings: list[dict] = []
    for i in range(len(segments)):
        s1 = segments[i]
        for j in range(i + 1, len(segments)):
            s2 = segments[j]
            if _bboxes_separated(s1, s2):
                continue
            finding = _pair_finding(s1, s2)
            if finding is not None:
                findings.append(finding)
    findings.sort(
        key=lambda f: (
            f["segment_ids"][0],
            f["segment_ids"][1],
            f["kind"],
            json.dumps(f, ensure_ascii=False, sort_keys=True),
        )
    )
    return findings
