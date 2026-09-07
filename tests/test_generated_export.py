"""Artificial cohorts only; no real rollout output or frozen pipeline edits."""
from pathlib import Path
import sys
sys.path[:0]=[str(Path(__file__).resolve().parents[1]/'scripts'),str(Path(__file__).resolve().parent)]
import json
import numpy as np
import pytest
from test_generated_evaluation import fixture_records,config
from evaluate_rollouts import summarize
import export_generated_measurements as export
import run_generated_exports as watcher

FACTORS={'probe_linear':1.,'affine_forward':-.5,'quadratic_forward':2.,'random_norm':0.,'wrong_variable_ball_x':-1.}


def synthetic(clipped=False):
    records,prediction=fixture_records();lookup={r['record_id']:i for i,r in enumerate(records)}
    for i,row in enumerate(records):
        row['seed']+=2026090700
        if row['intervention_type']=='baseline':continue
        baseline=prediction[lookup[row['baseline_record_id']]]
        effective=float(np.clip(row['dose'],-100,400)) if clipped else row['dose']
        row['height']['effective_dose']=effective;row['height']['target_clipped']=effective!=row['dose']
        prediction[i]=baseline+(prediction[i]-baseline)*(effective/row['dose'])*FACTORS[row['intervention_type']]
    protocol={**config(),'paths':list(export.PATHS),'requested_doses':list(export.DOSES)}
    report={**summarize(records,prediction,np.ones(12),protocol),'phase':'synthetic'}
    return report,protocol,prediction,records


def test_all_paths_negative_and_constant_cases_retained_with_equal_match_bands():
    report,protocol,prediction,records=synthetic()
    export.validate_prediction_joins(report,prediction,[r['record_id'] for r in records],records)
    tables,curves,matches=export.make_tables(report,protocol)
    assert matches==['a','b'] and set(curves)==set(export.PATHS)
    assert {r['path'] for r in tables['dose_response']}==set(export.PATHS)
    assert len(tables['dose_response'])==100 and len(tables['per_match_dose'])==200
    assert len(tables['per_pair_dose'])==400 and len(tables['per_pair_time'])==80
    for path,factor in FACTORS.items():
        expected=np.array(export.DOSES)[:,None]*np.arange(1,5)[None,:]*4*factor
        assert np.array_equal(curves[path]['mean'],expected)
        for row in (r for r in tables['dose_response'] if r['path']==path):
            base=row['requested_dose_uu']*(row['time_index']+1)*factor
            assert row['pointwise_match_bootstrap_low_uu']==min(2*base,6*base)
            assert row['pointwise_match_bootstrap_high_uu']==max(2*base,6*base)
    constant=[r for r in tables['per_pair_time'] if r['path']=='random_norm']
    assert all(r['spearman'] is None and not r['nondecreasing_with_nonzero_span'] for r in constant)
    negative=[r for r in tables['per_match_time'] if r['path']=='affine_forward']
    assert all(r['view0_extreme_contrast']<0 for r in negative)


def test_effective_dose_axis_keeps_original_assignment_groups():
    report,protocol,_,_=synthetic(clipped=True)
    tables,curves,_=export.make_tables(report,protocol)
    for path in export.PATHS:
        assert np.array_equal(curves[path]['effective_mean'],[-100,-100,0,300,400])
        rows=[r for r in tables['dose_response'] if r['path']==path and r['time_index']==0]
        assert [r['requested_dose_uu'] for r in rows]==list(export.DOSES)
        assert [r['clipped_pair_fraction'] for r in rows]==[1.,1.,0.,0.,1.]
        assert len(rows)==5


def test_complete_csv_roundtrip_keeps_undefined_and_signed_values(tmp_path):
    report,protocol,_,_=synthetic()
    tables,_,_=export.make_tables(report,protocol)
    for name,rows in tables.items():
        output=tmp_path/(name+'.csv');inventory=export.csv_write(output,rows)
        assert inventory['rows']==len(rows) and inventory['bytes']>0
        assert b'\r\n' not in output.read_bytes()


def test_bad_audit_blocks_export_before_reading_results(tmp_path,monkeypatch):
    (tmp_path/'selection_audit.json').write_text(json.dumps({'status':'failed'}))
    monkeypatch.setattr(sys,'argv',['export','--phase','selection','--evaluation-dir',str(tmp_path)])
    with pytest.raises(ValueError,match='Independent measurement audit'):
        export.main()
    assert not (tmp_path/'selection_figures').exists()


def test_prediction_source_join_corruption_is_rejected():
    report,_,prediction,records=synthetic()
    wrong=prediction.copy();wrong[1,0,0,8]+=1
    with pytest.raises(ValueError,match='Audited response differs'):
        export.validate_prediction_joins(report,wrong,[r['record_id'] for r in records],records)


def audit_fixture(tmp_path):
    path=tmp_path/'selection_audit.json'
    path.write_text(json.dumps({'status':'passed_generated_measurement_audit','phase':'selection',
                              'all_summary_metrics_independently_recomputed':True}))
    row={'stage':'selection_measurement_audit','path':str(path),'sha256':watcher.sha(path)}
    return path,{'status':'running','completed_stages':[row]}


def test_watcher_waits_for_owners_completed_stage_not_filename(tmp_path):
    path,study=audit_fixture(tmp_path)
    assert not watcher.completed_audit({**study,'completed_stages':[]},'selection',path)
    assert watcher.completed_audit(study,'selection',path)


def test_watcher_rejects_failed_study_or_changed_audit(tmp_path):
    path,study=audit_fixture(tmp_path)
    with pytest.raises(RuntimeError,match='study failed'):
        watcher.completed_audit({'status':'failed','completed_stages':[]},'selection',path)
    with pytest.raises(RuntimeError,match='study failed'):
        watcher.completed_audit({**study,'status':'failed'},'selection',path)
    path.write_text(path.read_text()+'\n')
    with pytest.raises(RuntimeError,match='provenance differs'):
        watcher.completed_audit(study,'selection',path)


def test_watcher_rejects_wrong_phase_or_incomplete_audit(tmp_path):
    path,study=audit_fixture(tmp_path)
    for replacement in ({'phase':'pilot'},{'all_summary_metrics_independently_recomputed':False}):
        audit={'status':'passed_generated_measurement_audit','phase':'selection',
               'all_summary_metrics_independently_recomputed':True,**replacement}
        path.write_text(json.dumps(audit));study['completed_stages'][0]['sha256']=watcher.sha(path)
        with pytest.raises(RuntimeError,match='Full independent measurement audit'):
            watcher.completed_audit(study,'selection',path)
