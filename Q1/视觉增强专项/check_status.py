import datetime,json,subprocess
from pathlib import Path
root=Path(__file__).resolve().parent; r=root/'results'
job=json.loads((root/'jobs.json').read_text())['run_job_id']
proc=subprocess.run(['sacct','-j',str(job),'--format=JobID,State,ExitCode,Elapsed','-n'],text=True,capture_output=True)
report=dict(checked_at=datetime.datetime.now().astimezone().isoformat(),job_id=job,sacct=proc.stdout,sacct_error=proc.stderr,success=(r/'SUCCESS.txt').exists())
if (r/'progress.json').exists(): report['progress']=json.loads((r/'progress.json').read_text())
if (r/'job_failed.txt').exists(): report['failure']=(r/'job_failed.txt').read_text()
report['status']='completed' if report['success'] else ('failed' if 'failure' in report else 'not_completed_check_scheduler_and_log')
r.mkdir(exist_ok=True); (r/'status_1230.json').write_text(json.dumps(report,indent=2,ensure_ascii=False)); print(json.dumps(report,indent=2,ensure_ascii=False))
