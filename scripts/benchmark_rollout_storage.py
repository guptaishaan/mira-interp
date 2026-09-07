#!/usr/bin/env python3
"""CPU-only storage benchmark on immutable pilot outputs; never changes model inputs."""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import time
import zipfile
os.environ['NUMPY_MADVISE_HUGEPAGE'] = '0'
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
MODES = ('deflate_default', 'stored', 'deflate_level1')


def require(value, message):
    if not value:
        raise ValueError(message)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for data in iter(lambda: stream.read(8*1024**2), b''):
            digest.update(data)
    return digest.hexdigest()


def array_record(array):
    return {'shape': list(array.shape), 'dtype': str(array.dtype), 'bytes': array.nbytes,
            'sha256': hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()}


def write_npz(path, arrays, mode):
    with path.open('xb') as stream:
        if mode == 'deflate_default':
            np.savez_compressed(stream, **arrays)
        elif mode == 'stored':
            np.savez(stream, **arrays)
        elif mode == 'deflate_level1':
            with zipfile.ZipFile(stream, 'w', compression=zipfile.ZIP_DEFLATED,
                                 compresslevel=1, allowZip64=True) as archive:
                for name, value in arrays.items():
                    with archive.open(name+'.npy', 'w', force_zip64=True) as member:
                        np.lib.format.write_array(member, value, allow_pickle=False)
        else:
            raise ValueError('Unregistered benchmark format')
        stream.flush()
        os.fsync(stream.fileno())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, default=ROOT/'results/rollout_steering_v1/pilot.json')
    parser.add_argument('--output-dir', type=Path, default=Path('/data2/ishaangp/mira-interp/storage_benchmark_v1'))
    parser.add_argument('--report', type=Path, default=ROOT/'results/rollout_storage_benchmark.json')
    args = parser.parse_args()
    require(not args.report.exists(), 'Completed benchmark report is immutable')
    args.output_dir.mkdir(parents=True, exist_ok=True)
    free_before = shutil.disk_usage(args.output_dir).free
    require(free_before > 28*1024**3, 'Maintain25GiB disk reserve plus bounded benchmark files')
    manifest = json.loads(args.manifest.read_text())
    require(manifest['status'] == 'passed_rollout_steering_pilot' and len(manifest['records']) == 2,
            'Require completed reserved two-condition pilot')
    manifest_sha = sha(args.manifest)
    inputs, timings = [], []
    started = time.perf_counter()
    for index, row in enumerate(manifest['records']):
        source = Path(row['generated_artifact_path'])
        require(sha(source) == row['generated_sha256'], 'Original pilot artifact changed')
        with np.load(source, allow_pickle=False) as saved:
            arrays = {key: saved[key] for key in saved.files}
        identity = {key: array_record(array) for key,array in arrays.items()}
        require(all(array.dtype == np.float32 and np.isfinite(array).all() for array in arrays.values()),
                'Pilot output must contain finite originalFP32 arrays')
        inputs.append({'record_id':row['record_id'], 'source_path':str(source), 'source_sha256':row['generated_sha256'],
                       'source_compressed_bytes':source.stat().st_size, 'arrays':identity})
        for repetition in range(2):
            # Rotate order across both conditions/repeats; do not alter system cache state.
            offset = (index+repetition)%len(MODES)
            order = MODES[offset:]+MODES[:offset]
            for mode in order:
                path = args.output_dir/f'condition{index}_repeat{repetition}_{mode}.npz'
                before = time.perf_counter()
                write_npz(path,arrays,mode)
                write_seconds = time.perf_counter()-before
                file_sha = sha(path)  # Excluded from timing; also warms filesystem cache.
                before = time.perf_counter()
                with np.load(path,allow_pickle=False) as saved:
                    restored = {key:saved[key] for key in saved.files}
                read_seconds = time.perf_counter()-before
                require(set(restored) == set(arrays), 'Array key set changed')
                for key in arrays:
                    require(array_record(restored[key]) == identity[key] and np.array_equal(restored[key],arrays[key]),
                            'Storage benchmark changed an array bit pattern')
                timings.append({'record_id':row['record_id'],'repetition':repetition,'mode':mode,
                    'path':str(path),'sha256':file_sha,'bytes':path.stat().st_size,
                    'write_seconds_including_flush_fsync':write_seconds,'warm_read_seconds':read_seconds,
                    'all_array_metadata_and_bytes_equal':True})
                print(json.dumps(timings[-1]),flush=True)
                del restored
        require(sha(source) == row['generated_sha256'], 'Original pilot changed during benchmark')
        del arrays
    summary = {}
    for mode in MODES:
        entries = [row for row in timings if row['mode'] == mode]
        summary[mode] = {'trials':len(entries),
            'median_write_seconds':float(np.median([row['write_seconds_including_flush_fsync'] for row in entries])),
            'write_seconds_range':[min(row['write_seconds_including_flush_fsync'] for row in entries),max(row['write_seconds_including_flush_fsync'] for row in entries)],
            'median_warm_read_seconds':float(np.median([row['warm_read_seconds'] for row in entries])),
            'mean_bytes':float(np.mean([row['bytes'] for row in entries])),
            'projected_1344_output_bytes':int(np.ceil(np.mean([row['bytes'] for row in entries])*1344))}
    baseline = summary['deflate_default']['median_write_seconds']
    for row in summary.values():
        row['projected_1344_serial_write_seconds_saved_vs_default'] = (baseline-row['median_write_seconds'])*1344
        row['projected_1344_two_writer_wall_seconds_saved_vs_default'] = (baseline-row['median_write_seconds'])*1344/2
        row['projected_free_bytes_after_1344_outputs'] = free_before-row['projected_1344_output_bytes']
        row['projected25GiB_reserve_preserved'] = row['projected_free_bytes_after_1344_outputs'] >= 25*1024**3
    require(sha(args.manifest) == manifest_sha, 'Pilot manifest changed')
    result = {'status':'passed_storage_benchmark','script_sha256':sha(Path(__file__)),
        'pilot_manifest_sha256':manifest_sha,'numpy_version':np.__version__,'python_zipfile_compression':'zlib DEFLATE or ZIP_STORED',
        'source_artifacts':inputs,'trials':timings,'summary':summary,
        'write_timing':'includes ZIP close, stream flush, and fsync; source arrays already in memory',
        'read_timing':'warm filesystem cache after file hashing; includes complete array materialization, excludes verification',
        'projection_limit':'Extrapolates two reserved conditions only; compression and concurrent throughput may vary on other videos. Two-writer estimate assumes equal independent throughput.',
        'free_bytes_before':free_before,'free_bytes_after':shutil.disk_usage(args.output_dir).free,
        'original_artifacts_and_manifest_unchanged':True,'all_reloaded_arrays_bitwise_equal':True,
        'model_or_protocol_changed':False,'gpu_used':False,'elapsed_seconds':time.perf_counter()-started}
    args.report.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps({'status':result['status'],'report':str(args.report),'report_sha256':sha(args.report),'summary':summary},indent=2),flush=True)


if __name__ == '__main__':
    main()
