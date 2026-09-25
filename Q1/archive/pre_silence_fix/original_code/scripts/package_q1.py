"""打包指定的已核验 Q1 结果；排除模型、视频、密钥、环境及协作日志。"""
import argparse
import json
import sys
from pathlib import Path
from zipfile import ZipFile, ZIP_DEFLATED
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from artifact_contract import file_hash

p=argparse.ArgumentParser()
p.add_argument('--out',type=Path,default=ROOT/'outputs/server_a100')
p.add_argument('--figures',type=Path,default=ROOT/'figures/server_a100')
p.add_argument('--destination',type=Path,default=ROOT/'Q1_A100_results.zip')
args=p.parse_args()
validation=json.loads((args.out/'validation.json').read_text())
assert validation['samples_checked']==100
assert validation['shape_finiteness_padding_and_time_checks']=='passed'
assert validation['pooled_features_recomputed']=='passed'
repeat=json.loads((args.out/'reproducibility.json').read_text())
assert repeat['repeat_samples']==5 and repeat['all_arrays_identical']
environment=json.loads((args.out/'environment.json').read_text())
for name,digest in environment['code_hashes'].items():
    assert file_hash(ROOT/'src'/name)==digest, f'Extraction source changed: {name}'
assert len(list((args.out/'features').glob('*.npz')))==100
assert len(list((args.out/'alignment').glob('*.json')))==100
figure_sources=json.loads((args.figures/'sources.json').read_text())
assert figure_sources['script_sha256']==file_hash(ROOT/'scripts/plot_paper.py'), 'Plot source changed since export'
assert figure_sources['source_environment_sha256']==file_hash(args.out/'environment.json')
for sid,digest in figure_sources['feature_hashes'].items():
    assert file_hash(args.out/'features'/f'{sid}.npz')==digest, sid
files=[]
for folder in ['src','tests','scripts','configs']:
    files += [(f, f.relative_to(ROOT).as_posix()) for f in (ROOT/folder).rglob('*')
              if f.is_file() and '__pycache__' not in f.parts]
for folder in ['features','alignment','records']:
    files += [(f, 'outputs/'+f.relative_to(args.out).as_posix()) for f in (args.out/folder).glob('*') if f.is_file()]
files += [(f,'outputs/'+f.name) for f in args.out.iterdir() if f.is_file()]
files += [(f,'figures/'+f.name) for f in args.figures.iterdir() if f.is_file() and f.suffix in ['.png','.eps','.pdf','.svg','.json']]
for name in ['README.md','requirements.txt','requirements-lock.txt','requirements-server.txt','requirements-server-lock.txt','requirements-qa.txt','models/manifest.json']:
    f=ROOT/name
    if f.is_file(): files.append((f,name))
for name in ['server_plots.log','server_tests.log','server_validation.log','server_reproducibility.log','server_run.log','tests_migration_local.log','server_figure_audit.json','nature_source_audit.json','tex_unchanged_migration.json']:
    f=ROOT/'logs'/name
    if f.is_file(): files.append((f,'logs/'+name))
args.destination.parent.mkdir(parents=True,exist_ok=True)
with ZipFile(args.destination,'w',ZIP_DEFLATED,compresslevel=6) as z:
    for source,name in sorted(files,key=lambda pair:pair[1]): z.write(source,name)
with ZipFile(args.destination) as z:
    assert z.testzip() is None
    assert len([n for n in z.namelist() if n.startswith('outputs/features/') and n.endswith('.npz')])==100
receipt={'filename':args.destination.name,'bytes':args.destination.stat().st_size,
         'sha256':file_hash(args.destination),'file_count':len(files),
         'status':'Automatically verified Q1 candidate; human word-boundary validation pending',
         'runtime':environment.get('runtime'), 'source_output_directory':str(args.out)}
args.destination.with_suffix('.receipt.json').write_text(json.dumps(receipt,ensure_ascii=False,indent=2))
print(json.dumps(receipt,ensure_ascii=False,indent=2))
