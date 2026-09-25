"""Regression: digital silence must never yield lexical times or usable acoustics."""
import sys
from pathlib import Path
import numpy as np
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from pipeline import Extractor, audio_signal_status, acoustic_features

def test_silence_skips_model_and_keeps_all_words_unaligned():
    # No processor/vocab/model exists: reaching inference would fail this test.
    extractor=object.__new__(Extractor)
    words=extractor.align(np.zeros(16000,np.float32),0,'word 2026 example')
    assert len(words)==3
    assert all(not w['aligned'] and w['start'] is None and w['end'] is None and w['score'] is None and w['reason']=='audio_all_zero' for w in words)

def test_silent_acoustics_have_no_valid_frames():
    wave=np.zeros(16000,np.float32)
    values,times,valid,pitch=acoustic_features(wave,np.ones(len(wave),bool),1.)
    assert len(times)==20 and times[0]==1.
    assert not values.any() and not valid.any()

def test_tiny_nonzero_is_not_arbitrarily_discarded():
    x=np.zeros(16000,np.float32); x[2]=1e-12
    assert not audio_signal_status(x)['audio_all_zero']

@pytest.mark.parametrize('x',[np.array([]),np.array([np.nan]),np.array([np.inf])])
def test_invalid_signal_rejected(x):
    with pytest.raises(ValueError): audio_signal_status(x)
