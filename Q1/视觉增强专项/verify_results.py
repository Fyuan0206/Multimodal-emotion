import hashlib,json
from pathlib import Path
import numpy as np
import pandas as pd
r=Path(__file__).resolve().parent/'results'; x=pd.read_csv(r/'evaluation/oof_predictions.csv'); m=pd.read_csv(r/'evaluation/metrics.csv')
assert len(x)==100 and x.sample_id.nunique()==100 and x.video_id.nunique()==37
assert (x.groupby('video_id').outer_fold.nunique()==1).all()
p=json.loads((r/'evaluation/provenance.json').read_text())
for sid,h in p['expression_sha256'].items():
    f=r/'features'/f'{sid}.npz'; assert hashlib.sha256(f.read_bytes()).hexdigest()==h
    with np.load(f) as z:
        assert z['scores'].shape==(len(z['valid']),7)
        assert np.isfinite(z['scores']).all() and (z['scores'][~z['valid']]==0).all()
for row in m.to_dict('records'):
    assert np.isclose(abs(x[row['model']]-x.label).mean(),row['mae'])
assert len(list((r/'features').glob('*.npz')))==100
report=dict(passed=True,n=100,groups=37,source_group_leakage=False)
(r/'validation.json').write_text(json.dumps(report,indent=2)); print(report)
text='# 自动运行结果\n\n模型输出为预训练七类表情分数，尚无本数据集人工表情真值。下面只评价情感倾向预测；不能视为七类表情准确率。所有比较是对同一100条数据的探索性后续分析，并非独立最终测试集。\n\n'
text+=m.to_csv(index=False)+'\n配对误差差值（负数为改善）：\n'+(r/'evaluation/paired_comparisons.csv').read_text()
text+='\n区间是固定折外预测的描述性组自助区间，不包含重新训练的不确定性。结果无论是否改善均保留，原Q1特征不自动替换。\n'
(r/'结果摘要.md').write_text(text)
