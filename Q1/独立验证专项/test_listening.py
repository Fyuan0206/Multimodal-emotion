"""Synthetic unit fixtures only; never used as paper/reference data."""
import json
import pandas as pd
import pytest
from evaluate_listening import evaluate

@pytest.fixture
def case(tmp_path):
    (tmp_path/'sample.json').write_text(json.dumps(dict(words=[dict(text='word',aligned=True,start=1.,end=2.)],media=dict(audio_start_s=0,audio_end_s=3))))
    d=pd.DataFrame([dict(sample_id='sample',word_index=0,word='word',reference_start_s=1.05,reference_end_s=2.2,review_status='confirmed',confidence='high',reviewer='UNIT_TEST_FIXTURE',review_date='2026-09-25',playback_rate=1,reviewer_note='synthetic fixture')])
    return tmp_path,d

def test_known_errors(case):
    p,d=case; _,m=evaluate(d,p)
    assert m['start_mae_ms']==pytest.approx(50)
    assert m['end_mae_ms']==pytest.approx(200)
    assert m['p90_max_boundary_ms']==pytest.approx(200)

def test_uncertain_excluded(case):
    p,d=case; d['review_status']='uncertain'; _,m=evaluate(d,p)
    assert m['matched_words']==0 and m['status']=='pending'

@pytest.mark.parametrize('field,value',[('word','wrong'),('reference_end_s',4),('reviewer',''),('reference_start_s',''),('review_date','bad'),('word_index',.5)])
def test_invalid_rejected(case,field,value):
    p,d=case; d=d.astype(object); d.loc[0,field]=value
    with pytest.raises(ValueError): evaluate(d,p)

def test_duplicate_rejected(case):
    p,d=case
    with pytest.raises(ValueError): evaluate(pd.concat([d,d]),p)


def test_silent_original_cannot_be_confirmed(case):
    p,d=case; path=p/'sample.json'; j=json.loads(path.read_text()); j['media']['audio_all_zero']=True; path.write_text(json.dumps(j))
    with pytest.raises(ValueError,match='Silent original'): evaluate(d,p)
