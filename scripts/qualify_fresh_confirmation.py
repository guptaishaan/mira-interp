#!/usr/bin/env python3
"""Apply the prospectively declared all-eight-clips quality rule to reserved matches."""
import hashlib
import json
from pathlib import Path
from datetime import datetime, timezone
ROOT=Path(__file__).resolve().parents[1]
def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()
registered=ROOT/'data/fresh_confirmation_split_manifest.json'
source=ROOT/'data/fresh_confirmation_clip_manifest.json'
audit_path=ROOT/'results/fresh_confirmation_data_audit.json'
m=json.loads(registered.read_text()); clips=json.loads(source.read_text()); audit=json.loads(audit_path.read_text())
expected={r['match_id'] for r in m['matches'] if r['split']=='confirmation'}
if len(expected)!=24 or set(audit['completed_match_ids'])!=expected or 'finished_at_utc' not in audit:
    raise ValueError('All24 reserved candidate audits must finish first')
if clips['split_manifest_sha256']!=sha(registered) or audit['clip_manifest_sha256']!=sha(source):
    raise ValueError('Source qualification provenance failed')
good={mid for mid in expected if sum(r['match_id']==mid for r in clips['records'])==8}
if len(good)<10:
    raise ValueError('Fewer than10 fully qualified fresh matches; no confirmation permitted')
records=[r for r in clips['records'] if r['match_id'] in good]
if len({r['clip_id'] for r in records})!=8*len(good) or any(r['role']!='confirmation' for r in records):
    raise ValueError('Wrong role or duplicate clips')
out={'status':'passed','scope':'fresh confirmation source qualification only; no performance inspected',
     'registered_manifest_sha256':sha(registered),'source_audit_sha256':sha(audit_path),'parent_clip_manifest_sha256':sha(source),
     'registered_utc':m['registered_utc'],'qualified_utc':datetime.now(timezone.utc).isoformat(),
     'quality_rule':'all eight prescribed clips pass original source gates; no replacements or threshold changes',
     'qualified_matches':sorted(good),'excluded_matches':sorted(expected-good),'n_matches':len(good),
     'parent_errors':audit['errors'],'records':records,'source_revision':m['source_revision']}
path=ROOT/'data/qualified_fresh_confirmation_clip_manifest.json'
if path.exists():
    raise ValueError('Qualification already exists; do not overwrite')
path.write_text(json.dumps(out,indent=2)+'\n')
report={k:v for k,v in out.items() if k!='records'}
report['qualified_clip_manifest_sha256']=sha(path)
(ROOT/'results/fresh_confirmation_qualification.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps({'status':'passed','matches':len(good),'excluded':len(expected-good),'clips':len(records)}))
