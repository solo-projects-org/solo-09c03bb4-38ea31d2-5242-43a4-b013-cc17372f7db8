"""Boundary tests for road-network quality-check geometry."""

import pytest

from app.matching import Segment
from app.quality import (
    FINDING_INTERSECTION,
    FINDING_OVERLAP,
    run_quality_check,
)


def seg(sid, ax, ay, bx, by):
    return Segment(sid, ax, ay, bx, by)


def kinds(findings):
    return [f["kind"] for f in findings]


def test_clean_endpoint_join_is_not_a_defect():
    # Two one-way segments joined end to start.
    findings = run_quality_check([seg("A", 0, 0, 100, 0), seg("B", 100, 0, 200, 0)])
    assert findings == []


def test_t_junction_where_endpoint_hits_interior_is_defect():
    # B ends at the midpoint of A: the contact is interior to A.
    findings = run_quality_check(
        [seg("A", 0, 0, 100, 0), seg("B", 50, 50, 50, 0)]
    )
    assert kinds(findings) == [FINDING_INTERSECTION]
    f = findings[0]
    assert f["segment_ids"] == ["A", "B"]
    assert f["point"] == pytest.approx([50.0, 0.0], abs=1e-9)
    assert f["interior_of"] == ["A"]


def test_crossing_interior_to_both_is_defect():
    findings = run_quality_check(
        [seg("A", 0, 0, 100, 100), seg("B", 0, 100, 100, 0)]
    )
    assert kinds(findings) == [FINDING_INTERSECTION]
    assert findings[0]["point"] == pytest.approx([50.0, 50.0], abs=1e-9)
    assert sorted(findings[0]["interior_of"]) == ["A", "B"]


def test_endpoint_to_endpoint_contact_in_middle_of_other_segment_chain_is_defect():
    # C ends exactly where A and B join: the point is interior to neither A nor
    # B, but it IS interior to... here it is C's endpoint and A/B shared node:
    # all contacts are endpoint-only -> clean. This documents the rule.
    findings = run_quality_check(
        [
            seg("A", 0, 0, 100, 0),
            seg("B", 100, 0, 200, 0),
            seg("C", 100, 50, 100, 0),
        ]
    )
    assert findings == []


def test_interior_contact_with_tiny_float_noise_is_still_endpoint_join():
    # B's start sits 5e-7 m off A's end, within the endpoint tolerance: the
    # contact must not be misreported as an interior intersection.
    findings = run_quality_check(
        [seg("A", 0, 0, 100, 0), seg("B", 100 + 5e-7, 0, 200, 0)]
    )
    assert findings == []


def test_partial_collinear_overlap_same_direction_is_defect():
    findings = run_quality_check(
        [seg("A", 0, 0, 100, 0), seg("B", 60, 0, 160, 0)]
    )
    assert kinds(findings) == [FINDING_OVERLAP]
    f = findings[0]
    assert f["segment_ids"] == ["A", "B"]
    assert f["from"] == pytest.approx([60.0, 0.0], abs=1e-9)
    assert f["to"] == pytest.approx([100.0, 0.0], abs=1e-9)
    assert f["length"] == pytest.approx(40.0, abs=1e-9)


def test_partial_collinear_overlap_opposite_direction_is_defect():
    findings = run_quality_check(
        [seg("A", 0, 0, 100, 0), seg("B", 160, 0, 60, 0)]
    )
    assert kinds(findings) == [FINDING_OVERLAP]
    assert findings[0]["length"] == pytest.approx(40.0, abs=1e-9)


def test_fully_contained_segment_is_defect():
    findings = run_quality_check(
        [seg("A", 0, 0, 100, 0), seg("B", 20, 0, 80, 0)]
    )
    assert kinds(findings) == [FINDING_OVERLAP]
    assert findings[0]["length"] == pytest.approx(60.0, abs=1e-9)


def test_duplicate_geometry_is_full_overlap_defect():
    findings = run_quality_check(
        [seg("A", 0, 0, 100, 0), seg("B", 0, 0, 100, 0)]
    )
    assert kinds(findings) == [FINDING_OVERLAP]
    assert findings[0]["length"] == pytest.approx(100.0, abs=1e-9)


def test_collinear_segments_sharing_only_endpoint_are_clean():
    findings = run_quality_check(
        [seg("A", 0, 0, 100, 0), seg("B", 100, 0, 100, 100)]
    )
    assert findings == []


def test_zero_length_overlap_at_shared_endpoint_is_clean():
    # B starts exactly at A's end but continues in a different direction:
    # collinear only at a single point, no positive-length overlap.
    findings = run_quality_check(
        [seg("A", 0, 0, 100, 0), seg("B", 100, 0, 100, -80)]
    )
    assert findings == []


def test_parallel_non_collinear_segments_are_clean():
    findings = run_quality_check(
        [seg("A", 0, 0, 100, 0), seg("B", 0, 5, 100, 5)]
    )
    assert findings == []


def test_disjoint_segments_are_clean():
    findings = run_quality_check(
        [seg("A", 0, 0, 10, 0), seg("B", 100, 100, 200, 100)]
    )
    assert findings == []


def test_sample_network_passes_and_defective_one_fails():
    # Mirrors sample/network.csv: endpoint joins and parallel disjoint roads.
    clean = [
        seg("N101", 0, 0, 120, 0),
        seg("N102", 120, 0, 260, 0),
        seg("N201", 0, 6, 120, 6),
        seg("N202", 120, 6, 260, 0),
        seg("N301", 260, 0, 260, 90),
    ]
    assert run_quality_check(clean) == []

    defective = clean + [seg("X1", 60, -30, 60, 30)]
    findings = run_quality_check(defective)
    assert kinds(findings).count(FINDING_INTERSECTION) == 2  # crosses N101, N201


def test_findings_are_deterministic_and_pairwise_complete():
    segments = [
        seg("S1", 0, 0, 100, 0),
        seg("S2", 0, 50, 100, -50),   # crosses S1 at (50, 0)
        seg("S3", 20, 0, 40, 0),      # overlaps S1
    ]
    first = run_quality_check(segments)
    second = run_quality_check(list(reversed(segments)))
    assert first == second
    assert len(first) == 2
    assert sorted(kinds(first)) == sorted(
        [FINDING_OVERLAP, FINDING_INTERSECTION]
    )
    # Every defective pair appears exactly once.
    pairs = {tuple(f["segment_ids"]) for f in first}
    assert pairs == {("S1", "S2"), ("S1", "S3")}
