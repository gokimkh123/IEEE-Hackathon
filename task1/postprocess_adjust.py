from typing import Optional, Tuple, List
import numpy as np

def _find_intervals(mask: np.ndarray) -> List[Tuple[int, int]]:
    mask = np.asarray(mask, dtype=bool)
    if mask.ndim != 1:
        raise ValueError("mask must be 1D")
    T = mask.size
    if T == 0:
        return []
    on = np.flatnonzero(mask)
    if on.size == 0:
        return []

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
    s = np.asarray(scores).astype(float).reshape(-1)
    T = s.size
    pred = (s > float(threshold)).astype(bool)

    if cooldown and T > 0:
        intervals = _find_intervals(pred)
        cd_intervals = []
        last_end_with_cd = -10**9
        for (st, en) in intervals:
            if st <= last_end_with_cd + cooldown and cd_intervals:
                prev_s, prev_e = cd_intervals[-1]
                cd_intervals[-1] = (prev_s, max(prev_e, en))
            else:
                cd_intervals.append((st, en))
            last_end_with_cd = cd_intervals[-1][1]
        pred = _apply_intervals(T, cd_intervals)

    intervals = _find_intervals(pred)

    latency_sum = 0
    seg_cnt = 0
    if labels is not None and T > 0 and intervals:
        y = np.asarray(labels).astype(bool).reshape(-1)
        if y.size != T:
            raise ValueError("labels must be the same length as scores.")
        new_intervals = []
        for (st, en) in intervals:
            j = st
            step = 0
            while j - 1 >= 0 and y[j - 1] and (max_backfill is None or step < max_backfill):
                j -= 1
                step += 1
            latency_sum += (st - j)
            seg_cnt += 1
            new_intervals.append((j, en))
        intervals = new_intervals

    intervals = _merge_and_filter(intervals, min_duration=min_duration, merge_gap=merge_gap)

    pred = _apply_intervals(T, intervals)

    if return_latency and labels is not None and seg_cnt > 0:
        return pred, latency_sum / seg_cnt
    if return_latency:
        return pred, 0.0
    return pred

def boolean_intervals(mask: np.ndarray) -> List[Tuple[int, int]]:
    return _find_intervals(np.asarray(mask, dtype=bool))