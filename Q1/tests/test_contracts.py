"""独立验证最优CTC路径、左闭右开池化、缺失观测、子词映射及缓存失效行为。"""
import itertools
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from ctc_align import ctc_viterbi
from pipeline import pool_interval, token_word_map, source_words


def test_viterbi_matches_exhaustive_ctc_search():
    rng = np.random.default_rng(4)
    for target in ([1, 2], [1, 1], [2]):
        scores = rng.normal(size=(5, 3))
        candidates = []
        for sequence in itertools.product(range(3), repeat=5):
            collapsed = [v for i, v in enumerate(sequence) if v != 0 and (i == 0 or v != sequence[i-1])]
            if collapsed == target:
                candidates.append(sum(scores[t, x] for t, x in enumerate(sequence)))
        path = ctc_viterbi(scores, target)
        extended = [0]
        for v in target: extended.extend([v, 0])
        result = sum(scores[t, extended[s]] for t, s in enumerate(path))
        assert np.isclose(result, max(candidates))


def test_pool_boundary_and_missing_nearest():
    values = np.array([[2.], [4.], [8.]], np.float32)
    times = np.array([.5, 1., 1.5])
    value, valid, idx, fallback, _ = pool_interval(values, times, np.ones(3, bool), .5, 1.)
    assert idx == [0] and value[0] == 2 and valid and not fallback
    value, valid, idx, fallback, distance = pool_interval(values, times, np.array([True, False, True]), 1.1, 1.2)
    assert idx == [1] and not valid and fallback and value[0] == 0 and distance > 0


def test_wordpiece_offsets_preserve_punctuation_and_specials():
    words = source_words("can't go!")
    offsets = np.array([[0,0],[0,3],[3,4],[4,5],[6,8],[8,9],[0,0]])
    assert token_word_map(offsets, words).tolist() == [-1,0,0,0,1,1,-1]


def test_digits_are_not_fabricated_as_aligned_words():
    words = source_words('in 2026')
    assert words[1]['unsupported'] == '2026' and not words[1]['aligned']


def test_ctc_rejects_blank_and_boolean_targets():
    import pytest
    for target in ([0], [True]):
        with pytest.raises(ValueError):
            ctc_viterbi(np.zeros((3, 3)), target)


def test_spans_reject_illegal_repeated_character_skip():
    import pytest
    from ctc_align import tokens_to_spans
    with pytest.raises(ValueError):
        tokens_to_spans([1, 3], [1, 1])
    with pytest.raises(ValueError):
        tokens_to_spans([1, 5], [1, 2, 3])


def test_resume_requires_both_intact_outputs(tmp_path):
    import json
    from artifact_contract import file_hash, reusable_record
    for folder in ['features', 'alignment', 'records']:
        (tmp_path / folder).mkdir()
    (tmp_path / 'features/a.npz').write_bytes(b'original')
    (tmp_path / 'alignment/a.json').write_text('{}')
    record = dict(sample_id='a', status='extracted_unreviewed', source_fingerprint='current',
                  artifact_sha256={f: file_hash(tmp_path/f/('a'+s)) for f,s in [('features','.npz'),('alignment','.json')]})
    (tmp_path / 'records/a.json').write_text(json.dumps(record))
    assert reusable_record(tmp_path, 'a', 'current')
    assert not reusable_record(tmp_path, 'a', 'other-code-version')
    (tmp_path / 'alignment/a.json').unlink()
    assert not reusable_record(tmp_path, 'a', 'current')
    (tmp_path / 'alignment/a.json').write_text('{}')
    (tmp_path / 'features/a.npz').write_bytes(b'corrupt')
    assert not reusable_record(tmp_path, 'a', 'current')


def test_model_tampering_is_rejected(tmp_path):
    import pytest
    from artifact_contract import file_hash, verify_models
    p=tmp_path/'weight';p.write_bytes(b'abcd')
    manifest=[dict(path='weight',bytes=4,sha256=file_hash(p))]
    verify_models(tmp_path,manifest)
    p.write_bytes(b'ABCD')
    with pytest.raises(ValueError):
        verify_models(tmp_path,manifest)
