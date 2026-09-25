"""Run installed Nature Figure / ModelViz checks on actual exported artifacts."""
import argparse
import json
import subprocess
import sys
from pathlib import Path

p=argparse.ArgumentParser()
p.add_argument('--figures',type=Path,required=True)
p.add_argument('--report',type=Path,required=True)
p.add_argument('--skills',type=Path,default=Path.home()/'.codex/skills')
a=p.parse_args()
sys.path.insert(0,str(a.skills/'modelviz-skill'))
from src.tools.inspect_generated_image import inspect_generated_image
skill=a.skills/'nature-figure/scripts'
a.report.parent.mkdir(parents=True,exist_ok=True)
rows=[]
for pdf in sorted(a.figures.glob('*.pdf')):
    collision=a.report.parent/f'{pdf.stem}.collision.json'
    r=subprocess.run([sys.executable,str(skill/'audit_figure_collisions.py'),str(pdf),'--json-out',str(collision)],capture_output=True,text=True)
    text=subprocess.run([sys.executable,str(skill/'audit_pdf_text.py'),str(pdf),'--min-pt','5','--json'],capture_output=True,text=True)
    png=inspect_generated_image.invoke({'image_path':str(pdf.with_suffix('.png').resolve())})
    rows.append(dict(figure=pdf.stem,collision_exit_code=r.returncode,collision=json.loads(collision.read_text()),
                     text_exit_code=text.returncode,text=json.loads(text.stdout),png=png,
                     alignment=json.loads(pdf.with_suffix('.alignment.json').read_text())))
report={'figures':rows,'all_pass':bool(rows) and all(r['collision_exit_code']==0 and r['text_exit_code']==0 and r['png']['success'] for r in rows)}
a.report.write_text(json.dumps(report,ensure_ascii=False,indent=2))
print(json.dumps({'figures':len(rows),'all_pass':report['all_pass']}))
raise SystemExit(not report['all_pass'])
