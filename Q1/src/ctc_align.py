"""标准CTC维特比强制对齐，仅依赖数组计算。
扩展状态在字符间插入空白；相同字符必须经过空白状态。
起点限于首空白或首字符，终点限于末字符或末空白，不可达时明确报错。
负无穷表示不可能发射；拒绝空输入、非数值和正无穷。"""
from __future__ import annotations

import numpy as np

__all__ = ["ctc_viterbi", "tokens_to_spans"]

_NEG_INF = -np.inf


def _validate_log_probs(log_probs):
    """校验并返回形状为 [T, V] 的 float64 对数概率数组。"""
    if log_probs is None:
        raise ValueError("log_probs must not be None")
    arr = np.asarray(log_probs)
    if arr.ndim != 2:
        raise ValueError(
            "log_probs must be a 2-D array [T, V], got shape %r" % (tuple(arr.shape),)
        )
    if arr.size == 0:
        raise ValueError(
            "log_probs must be non-empty, got shape %r" % (tuple(arr.shape),)
        )
    if arr.dtype.kind not in "fiub":
        raise ValueError("log_probs must hold real numbers, got dtype %s" % arr.dtype)
    arr = arr.astype(np.float64, copy=False)
    if np.isnan(arr).any():
        raise ValueError("log_probs contains NaN")
    if np.isposinf(arr).any():
        raise ValueError(
            "log_probs contains +Inf; only -Inf is a valid 'impossible' score"
        )
    return arr


def _validate_blank(blank_id, vocab_size):
    if isinstance(blank_id, bool) or not isinstance(blank_id, (int, np.integer)):
        raise ValueError("blank_id must be an integer, got %r" % (blank_id,))
    blank_id = int(blank_id)
    if not 0 <= blank_id < vocab_size:
        raise ValueError(
            "blank_id=%d out of range [0, %d)" % (blank_id, vocab_size)
        )
    return blank_id


def _validate_target(target, vocab_size=None):
    """校验并返回整数 token 列表；给定词表大小时同时检查取值范围。"""
    if target is None:
        raise ValueError("target must not be None")
    if isinstance(target, (str, bytes)):
        raise ValueError("target must be a sequence of integer token ids")
    try:
        items = list(target)
    except TypeError:
        raise ValueError("target must be a sequence of integer token ids")
    if not items:
        raise ValueError("target must contain at least one token")
    toks = []
    for i, tok in enumerate(items):
        if isinstance(tok, (bool, np.bool_)) or not isinstance(tok, (int, np.integer)):
            raise ValueError(
                "target[%d]=%r is not an integer token id" % (i, tok)
            )
        tok = int(tok)
        if vocab_size is not None and not 0 <= tok < vocab_size:
            raise ValueError(
                "target[%d]=%d out of range [0, %d)" % (i, tok, vocab_size)
            )
        toks.append(tok)
    return toks


def ctc_viterbi(log_probs, target, blank_id=0):
    """用标准 CTC 维特比算法把 target 强制对齐到 log_probs。

    参数
    ----
    log_probs:
        形状为 ``[T, V]`` 的逐帧对数概率；取值为有限实数，
        不可能发生的发射用 ``-inf`` 表示。
    target:
        非空 token 序列，每个 token 都在 ``[0, V)`` 内。
    blank_id:
        CTC 空白符号的索引，需在 ``[0, V)`` 内。

    返回
    ----
    numpy.ndarray
        形状 ``[T]`` 的 int64 数组 path；``path[t]`` 是第 t 帧对应的
        扩展状态索引。扩展序列为 ``blank, target[0], blank, target[1], ..., blank``。

    异常
    -----
    ValueError
        输入非法（空、NaN、+Inf、编号越界、形状错误），或 target 无法在
        T 帧内完成对齐。
    """
    lp = _validate_log_probs(log_probs)
    n_frames, vocab_size = lp.shape
    blank_id = _validate_blank(blank_id, vocab_size)
    toks = _validate_target(target, vocab_size)

    if blank_id in toks:
        raise ValueError("target must not contain blank_id")

    n_tok = len(toks)
    n_states = 2 * n_tok + 1

    # 扩展标签序列：blank, y0, blank, y1, ..., blank。
    ext = np.empty(n_states, dtype=np.int64)
    ext[0] = blank_id
    ext[1::2] = toks
    ext[2::2] = blank_id

    emit = lp[:, ext]  # [T, S]：第 t 帧发射 ext[s] 的得分

    best = np.full((n_frames, n_states), _NEG_INF, dtype=np.float64)
    back = np.full((n_frames, n_states), -1, dtype=np.int64)

    # 第 0 帧：只允许状态 0（空白）或状态 1（首个 token）作为起点。
    best[0, 0] = emit[0, 0]
    best[0, 1] = emit[0, 1]

    # 跨两状态跳跃合法的位置：仅奇数（token）状态，且被跳过的空白两侧
    # 必须是不同 token，即重复字符必须经过空白。
    skip_ok = np.zeros(n_states, dtype=bool)
    for s in range(3, n_states, 2):
        skip_ok[s] = ext[s] != ext[s - 2]

    states = np.arange(n_states)
    for t in range(1, n_frames):
        prev = best[t - 1]

        cur = prev.copy()  # 候选一：停留在同一状态
        bck = states.copy()

        cand = np.full(n_states, _NEG_INF)
        cand[1:] = prev[:-1]  # 候选二：前进一个状态
        take = cand > cur
        cur = np.where(take, cand, cur)
        bck = np.where(take, states - 1, bck)

        cand = np.full(n_states, _NEG_INF)
        cand[2:] = prev[:-2]  # 候选三：跳过中间一个状态
        cand[~skip_ok] = _NEG_INF
        take = cand > cur
        cur = np.where(take, cand, cur)
        bck = np.where(take, states - 2, bck)

        best[t] = cur + emit[t]
        back[t] = bck

    # 第 T-1 帧：只允许最后空白或最后 token 作为路径终点。
    final = -1
    final_score = _NEG_INF
    for s in (n_states - 1, n_states - 2):
        if best[n_frames - 1, s] > final_score:
            final_score = best[n_frames - 1, s]
            final = s
    if final < 0 or not np.isfinite(final_score):
        raise ValueError(
            "target %r cannot be aligned over %d frames (unreachable end state)"
            % (toks, n_frames)
        )

    path = np.empty(n_frames, dtype=np.int64)
    s = final
    for t in range(n_frames - 1, -1, -1):
        path[t] = s
        prev_s = int(back[t, s])
        if t > 0 and prev_s < 0:
            raise ValueError(
                "backtracking broke at frame %d (state %d); this is a bug" % (t, s)
            )
        s = prev_s
    if s != -1:
        raise ValueError("backtracking did not terminate at the first frame")

    return path


def tokens_to_spans(path, target):
    """返回 CTC 路径中每个目标 token 占据的帧区间。

    参数
    ----
    path:
        非空一维扩展状态索引序列，即 :func:`ctc_viterbi` 的返回值。
    target:
        该路径所对齐的 token 序列。

    返回
    ----
    list of tuple
        ``[(start_0, end_0), ...]``，左闭右开帧区间，与 target 一一对应。

    异常
    -----
    ValueError
        path 不是合法的非递减状态路径，或某个 token 状态从未被访问。
    """
    if path is None:
        raise ValueError("path must not be None")
    arr = np.asarray(path)
    if arr.ndim != 1 or arr.size == 0:
        raise ValueError("path must be a non-empty 1-D sequence of state indices")
    if arr.dtype.kind not in "iu":
        raise ValueError("path must hold integer state indices, got dtype %s" % arr.dtype)
    states = [int(s) for s in arr.tolist()]

    toks = _validate_target(target)
    n_states = 2 * len(toks) + 1

    for s in states:
        if not 0 <= s < n_states:
            raise ValueError(
                "path state %d out of range [0, %d)" % (s, n_states)
            )
    for a, b in zip(states, states[1:]):
        if b < a:
            raise ValueError(
                "path must be non-decreasing, got %r" % (states,)
            )

    if states[0] not in (0, 1) or states[-1] not in (n_states - 2, n_states - 1):
        raise ValueError("invalid path boundary states")
    for a, b in zip(states, states[1:]):
        if b - a > 2 or (b - a == 2 and (b % 2 == 0 or toks[b // 2] == toks[a // 2])):
            raise ValueError("illegal CTC state transition")
    spans = []
    for i in range(len(toks)):
        s = 2 * i + 1
        frames = [t for t, st in enumerate(states) if st == s]
        if not frames:
            raise ValueError(
                "token %d (target index %d) is never aligned in path %r"
                % (toks[i], i, states)
            )
        spans.append((frames[0], frames[-1] + 1))
    return spans