"""Compare CPU and server artifacts without assuming cross-platform bitwise equality."""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd


def compare(reference, candidate):
    manifests = [pd.read_csv(p / 'manifest.csv').set_index('sample_id') for p in (reference, candidate)]
    assert set(manifests[0].index) == set(manifests[1].index)
    for key in ('source_sha256', 'text_sha256'):
        assert manifests[0][key].sort_index().equals(manifests[1][key].sort_index()), key
    rows = []
    boundary_changes = []
    for sid in sorted(manifests[0].index):
        metrics = {}
        with np.load(reference/'features'/f'{sid}.npz', allow_pickle=False) as a, np.load(candidate/'features'/f'{sid}.npz', allow_pickle=False) as b:
            assert set(a.files) == set(b.files), sid
            for key in a.files:
                assert a[key].shape == b[key].shape and a[key].dtype == b[key].dtype, (sid, key)
                delta = np.abs(a[key].astype(np.float64) - b[key].astype(np.float64))
                metrics[key] = {'max_abs': float(delta.max()) if delta.size else 0.,
                                'changed_elements': int(np.count_nonzero(delta))}
        rows.append({'sample_id': sid, 'arrays': metrics})
        documents = [json.loads((folder/'alignment'/f'{sid}.json').read_text()) for folder in (reference,candidate)]
        assert len(documents[0]['words']) == len(documents[1]['words']), sid
        for wi,(x,y) in enumerate(zip(documents[0]['words'],documents[1]['words'])):
            assert x['text'] == y['text'], (sid,wi)
            if x['aligned'] and y['aligned']:
                delta=max(abs(x['start']-y['start']),abs(x['end']-y['end']))
                if delta>1e-8:
                    boundary_changes.append(dict(sample_id=sid,word_index=wi,word=x['text'],
                        cpu_start=x['start'],cpu_end=x['end'],gpu_start=y['start'],gpu_end=y['end'],
                        delta_s=delta,cpu_score=x['score'],gpu_score=y['score']))
    boundary_changes.sort(key=lambda row:row['delta_s'],reverse=True)
    pd.DataFrame(boundary_changes,columns=['sample_id','word_index','word','cpu_start','cpu_end','gpu_start','gpu_end','delta_s','cpu_score','gpu_score']).to_csv(candidate/'cross_platform_boundary_changes.csv',index=False)
    return {'samples': len(rows), 'changed_boundary_words':len(boundary_changes),
            'changed_boundary_clips':len({r['sample_id'] for r in boundary_changes}),
            'max_boundary_difference_s':max((r['delta_s'] for r in boundary_changes),default=0),
            'boundary_review':'Required; do not infer correctness from agreement or emission score', 'input_hashes_match': True,
            'interpretation': 'Descriptive cross-platform differences, not an accuracy test or equivalence claim.',
            'maximum_absolute_difference_by_array': {k: max(r['arrays'][k]['max_abs'] for r in rows) for k in rows[0]['arrays']},
            'details': rows}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--reference', type=Path, required=True)
    parser.add_argument('--candidate', type=Path, required=True)
    args = parser.parse_args()
    report = compare(args.reference, args.candidate)
    (args.candidate/'cross_platform_comparison.json').write_text(json.dumps(report, indent=2))
    print(json.dumps({k: v for k, v in report.items() if k != 'details'}, indent=2))
