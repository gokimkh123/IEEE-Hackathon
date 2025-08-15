
"""
postprocess_adjust.py

Self-contained post-processing utilities for anomaly predictions.

Key function:
- adjust_predicts_v2(scores, threshold, labels=None, max_backfill=None, min_duration=1, merge_gap=0, cooldown=0, return_latency=False)

This module does NOT rely on project-specific args or packages.
"""

from typing import Optional, Tuple, List
import numpy as np

def _find_intervals(mask: np.ndarray) -> List[Tuple[int, int]]:
    """
    Find [start, end] (inclusive) intervals where mask is True.
    """
    mask = np.asarray(mask, dtype=bool)
    if mask.ndim != 1:
        raise ValueError("mask must be 1D")
    T = mask.size
    if T == 0:
        return []
    on = np.flatnonzero(mask)
    if on.size == 0:
        return []
    # split by gaps > 1
    gaps = np.where(np.diff(on) > 1)[0] + 1
    groups = np.split(on, gaps)
    return [(g[0], g[-1]) for g in groups]

def _apply_intervals(T: int, intervals: List[Tuple[int, int]]) -> np.ndarray:
    out = np.zeros(T, dtype=bool)
    for s, e in intervals:
        s = int(max(0, s))
        e = int(min(T - 1, e))
        if s <= e:
            out[s:e+1] = True
    return out

def _merge_and_filter(intervals: List[Tuple[int, int]], min_duration: int = 1, merge_gap: int = 0) -> List[Tuple[int, int]]:
    """
    Merge intervals closer than or equal to merge_gap, then remove intervals shorter than min_duration.
    """
    if not intervals:
        return []
    intervals = sorted(intervals)
    merged = []
    cur_s, cur_e = intervals[0]
    for s, e in intervals[1:]:
        if s - cur_e - 1 <= merge_gap:
            cur_e = max(cur_e, e)
        else:
            if (cur_e - cur_s + 1) >= min_duration:
                merged.append((cur_s, cur_e))
            cur_s, cur_e = s, e
    if (cur_e - cur_s + 1) >= min_duration:
        merged.append((cur_s, cur_e))
    return merged

def adjust_predicts_v2(
    scores: np.ndarray,
    threshold: float,
    labels: Optional[np.ndarray] = None,
    max_backfill: Optional[int] = None,
    min_duration: int = 1,
    merge_gap: int = 0,
    cooldown: int = 0,
    return_latency: bool = False
):
    """
    Convert scores -> binary predictions and stabilize them on a time-axis with latency backfill and interval rules.

    Args:
        scores: (T,) scores.
        threshold: float threshold applied to scores.
        labels: optional (T,) ground-truth binary labels for latency backfill within true segments.
        max_backfill: if set, limit how far we backfill into a true segment (in steps).
        min_duration: minimum length of a predicted segment to keep.
        merge_gap: merge two predicted segments if the gap between them is <= merge_gap.
        cooldown: ensure the prediction stays False for at least `cooldown` steps after the end of a predicted segment.
        return_latency: if True and labels are provided, return (pred, avg_latency); otherwise return pred only.

    Returns:
        pred or (pred, avg_latency)
    """
    s = np.asarray(scores).astype(float).reshape(-1)
    T = s.size
    pred = (s > float(threshold)).astype(bool)

    # Cooldown: prevent too-frequent toggling after a True interval ends
    if cooldown and T > 0:
        intervals = _find_intervals(pred)
        cd_intervals = []
        last_end_with_cd = -10**9
        for (st, en) in intervals:
            # Respect cooldown from previous interval's end
            if st <= last_end_with_cd + cooldown and cd_intervals:
                # merge with previous when within cooldown window
                prev_s, prev_e = cd_intervals[-1]
                cd_intervals[-1] = (prev_s, max(prev_e, en))
            else:
                cd_intervals.append((st, en))
            last_end_with_cd = cd_intervals[-1][1]
        pred = _apply_intervals(T, cd_intervals)

    intervals = _find_intervals(pred)

    # Latency backfill within true labels (if provided)
    latency_sum = 0
    seg_cnt = 0
    if labels is not None and T > 0 and intervals:
        y = np.asarray(labels).astype(bool).reshape(-1)
        if y.size != T:
            raise ValueError("labels must be the same length as scores.")
        new_intervals = []
        for (st, en) in intervals:
            # backtrack into the same GT-positive region up to max_backfill
            j = st
            step = 0
            while j - 1 >= 0 and y[j - 1] and (max_backfill is None or step < max_backfill):
                j -= 1
                step += 1
            latency_sum += (st - j)
            seg_cnt += 1
            new_intervals.append((j, en))
        intervals = new_intervals

    # Merge and minimum duration filtering
    intervals = _merge_and_filter(intervals, min_duration=min_duration, merge_gap=merge_gap)

    # Reconstruct prediction
    pred = _apply_intervals(T, intervals)

    if return_latency and labels is not None and seg_cnt > 0:
        return pred, latency_sum / seg_cnt
    if return_latency:
        return pred, 0.0
    return pred

def boolean_intervals(mask: np.ndarray) -> List[Tuple[int, int]]:
    """
    Public helper to extract (start, end) intervals from a boolean mask.
    """
    return _find_intervals(np.asarray(mask, dtype=bool))
