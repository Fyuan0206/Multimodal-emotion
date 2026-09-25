"""检验CTC强制对齐的输入约束、重复字符、不可达路径与时间帧回溯。
支持测试框架执行及直接运行。"""
from __future__ import annotations

import os
import sys
import traceback
from contextlib import contextmanager

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
for _p in (_ROOT, os.path.join(_ROOT, "src")):
    if os.path.isdir(_p) and _p not in sys.path:
        sys.path.insert(0, _p)

try:
    from src.ctc_align import ctc_viterbi, tokens_to_spans
except ImportError:  # pragma: no cover - src 未作为包导入时的回退
    from ctc_align import ctc_viterbi, tokens_to_spans


@contextmanager
def _raises(exc):
    """``pytest.raises`` 的最小替身，使测试不依赖 pytest 也能直接运行。"""
    try:
        yield
    except exc:
        return
    except Exception as other:  # 异常类型不符
        raise AssertionError(
            "期望 %s，实际 %s：%s" % (exc.__name__, type(other).__name__, other)
        )
    raise AssertionError("期望 %s，但没有任何异常抛出" % exc.__name__)


def _logprobs(frames, vocab, blank_id=0, hot=0.0, cold=-30.0):
    """合成对数概率矩阵，使其最优解恰为给定标签序列。"""
    lp = np.full((len(frames), vocab), cold, dtype=np.float64)
    for t, label in enumerate(frames):
        lp[t, label] = hot
    return lp


def _collapse(path, target, blank_id=0):
    """把对齐路径按CTC规则折叠回标签序列。"""
    ext = [blank_id]
    for tok in target:
        ext.extend([tok, blank_id])
    labels = [ext[s] for s in path]
    out = []
    prev = None
    for lab in labels:
        if lab != prev and lab != blank_id:
            out.append(lab)
        prev = lab
    return out


# --------------------------------------------------------------------------
# 普通路径
# --------------------------------------------------------------------------

def test_blank_token_blank():
    path = ctc_viterbi(_logprobs([0, 3, 0], vocab=5), [3])
    assert path.tolist() == [0, 1, 2]
    assert tokens_to_spans(path, [3]) == [(1, 2)]


def test_token_then_blank():
    path = ctc_viterbi(_logprobs([3, 0], vocab=5), [3])
    assert path.tolist() == [1, 2]
    assert tokens_to_spans(path, [3]) == [(0, 1)]


def test_repeated_frames_stay_in_state():
    path = ctc_viterbi(_logprobs([2, 2], vocab=4), [2])
    assert path.tolist() == [1, 1]
    assert tokens_to_spans(path, [2]) == [(0, 2)]


def test_multi_frame_token_span_is_half_open():
    path = ctc_viterbi(_logprobs([0, 3, 3, 0], vocab=5), [3])
    assert path.tolist() == [0, 1, 1, 2]
    assert tokens_to_spans(path, [3]) == [(1, 3)]


def test_different_tokens_may_skip_blank():
    path = ctc_viterbi(_logprobs([0, 1, 2, 0], vocab=4), [1, 2])
    assert path.tolist() == [0, 1, 3, 4]
    assert tokens_to_spans(path, [1, 2]) == [(1, 2), (2, 3)]


def test_minimum_length_alignment():
    path = ctc_viterbi(_logprobs([4, 0, 4], vocab=6), [4, 4])
    assert path.tolist() == [1, 2, 3]
    assert tokens_to_spans(path, [4, 4]) == [(0, 1), (2, 3)]


def test_nonzero_blank_id():
    path = ctc_viterbi(_logprobs([2, 3, 2], vocab=5, blank_id=2), [3], blank_id=2)
    assert path.tolist() == [0, 1, 2]
    assert tokens_to_spans(path, [3]) == [(1, 2)]


def test_single_frame_single_token():
    path = ctc_viterbi(_logprobs([2], vocab=4), [2])
    assert path.tolist() == [1]


# --------------------------------------------------------------------------
# 重复字符
# --------------------------------------------------------------------------

def test_repeated_token_requires_blank_between():
    path = ctc_viterbi(_logprobs([0, 4, 0, 4, 0], vocab=6), [4, 4])
    assert path.tolist() == [0, 1, 2, 3, 4]
    assert tokens_to_spans(path, [4, 4]) == [(1, 2), (3, 4)]


def test_repeated_token_cannot_skip_blank():
    # 两个4占两帧需要状态1直接跳到3，即跳过两个相同字符之间的空白：
    # 该转移非法，因此目标不可达。
    with _raises(ValueError):
        ctc_viterbi(_logprobs([4, 4], vocab=6), [4, 4])


def test_repeated_triple_token():
    path = ctc_viterbi(_logprobs([0, 4, 0, 4, 0, 4, 0], vocab=6), [4, 4, 4])
    assert path.tolist() == [0, 1, 2, 3, 4, 5, 6]
    assert tokens_to_spans(path, [4, 4, 4]) == [(1, 2), (3, 4), (5, 6)]


# --------------------------------------------------------------------------
# 不可达与非法输入
# --------------------------------------------------------------------------

def test_unreachable_too_short():
    with _raises(ValueError):
        ctc_viterbi(_logprobs([0, 1], vocab=4), [1, 2, 3])


def test_unreachable_single_frame_two_tokens():
    with _raises(ValueError):
        ctc_viterbi(_logprobs([1], vocab=4), [1, 2])


def test_empty_target():
    with _raises(ValueError):
        ctc_viterbi(_logprobs([0, 1], vocab=4), [])


def test_nan_input_rejected():
    lp = _logprobs([0, 1], vocab=4)
    lp[0, 1] = np.nan
    with _raises(ValueError):
        ctc_viterbi(lp, [1])


def test_posinf_input_rejected():
    lp = _logprobs([0, 1], vocab=4)
    lp[1, 1] = np.inf
    with _raises(ValueError):
        ctc_viterbi(lp, [1])


def test_neginf_is_a_valid_impossible_transition():
    lp = np.array(
        [[0.0, -np.inf, -np.inf], [-np.inf, 0.0, -np.inf]], dtype=np.float64
    )
    path = ctc_viterbi(lp, [1])
    assert path.tolist() == [0, 1]


def test_all_neginf_is_unreachable():
    lp = np.full((3, 4), -np.inf)
    with _raises(ValueError):
        ctc_viterbi(lp, [1])


def test_bad_shapes_rejected():
    with _raises(ValueError):
        ctc_viterbi(np.zeros((0, 3)), [1])
    with _raises(ValueError):
        ctc_viterbi(np.zeros((3, 0)), [1])
    with _raises(ValueError):
        ctc_viterbi(np.zeros(3), [1])
    with _raises(ValueError):
        ctc_viterbi(None, [1])


def test_bad_ids_rejected():
    lp = _logprobs([0, 1], vocab=4)
    with _raises(ValueError):
        ctc_viterbi(lp, [7])          # 字符编号越界
    with _raises(ValueError):
        ctc_viterbi(lp, [-1])         # 负编号
    with _raises(ValueError):
        ctc_viterbi(lp, [1], blank_id=9)
    with _raises(ValueError):
        ctc_viterbi(lp, [1.5])        # 非整数编号


def test_spans_reject_invalid_path():
    with _raises(ValueError):
        tokens_to_spans([0, 0, 2], [3])   # 字符状态1从未被访问
    with _raises(ValueError):
        tokens_to_spans([1, 0], [3])      # 路径不是非递减
    with _raises(ValueError):
        tokens_to_spans([9], [3])         # 状态编号越界
    with _raises(ValueError):
        tokens_to_spans([], [3])          # 空路径


# --------------------------------------------------------------------------
# 随机输入下的不变量
# --------------------------------------------------------------------------

def test_random_logprobs_path_is_valid_and_collapses_to_target():
    rng = np.random.default_rng(20260924)
    target = [1, 3, 3, 5]
    for _ in range(20):
        lp = rng.normal(loc=0.0, scale=3.0, size=(14, 6))
        path = ctc_viterbi(lp, target)
        assert path.shape == (14,)
        assert path.dtype == np.int64
        assert np.all(np.diff(path) >= 0), "路径必须非递减"
        assert path[0] in (0, 1), "首帧必须是空白或首个字符"
        assert path[-1] in (2 * len(target), 2 * len(target) - 1)
        for i in range(len(target)):
            assert np.any(path == 2 * i + 1), "每个字符状态都必须被访问"
        assert _collapse(path, target) == target
        spans = tokens_to_spans(path, target)
        assert len(spans) == len(target)
        for start, end in spans:
            assert 0 <= start < end <= 14


def test_random_logprobs_repeated_only_target():
    rng = np.random.default_rng(7)
    target = [2, 2]
    lp = rng.normal(size=(10, 5))
    path = ctc_viterbi(lp, target)
    assert _collapse(path, target) == target
    assert np.any(path == 2)  # 两个相同字符之间的空白状态必须被访问


def _main():
    tests = [
        (name, obj)
        for name, obj in sorted(globals().items())
        if name.startswith("test_") and callable(obj)
    ]
    failures = 0
    for name, fn in tests:
        try:
            fn()
        except Exception:
            failures += 1
            print("失败 %s" % name)
            traceback.print_exc()
        else:
            print("通过 %s" % name)
    print("\n%d 项通过，%d 项失败" % (len(tests) - failures, failures))
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(_main())