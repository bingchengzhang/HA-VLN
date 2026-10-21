"""Run the original closed stage on one exclusively free GPU, preserving all evaluation settings."""
import argparse
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import shutil
import threading
import time


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main():
    zbc, project = Path(os.environ['ZBC']), Path(os.environ['PROJECT'])
    runtime = load('stage', project / 'scripts/stage.py')
    supervisor = load('watch', project / 'scripts/training_watch.py')
    root = zbc / 'workspace/supervision/r0'
    previous = json.loads((root / 'status.json').read_text())
    if previous['status'] != 'WAITING_FOR_GPUS' or previous['phase'] != 'closed' or previous['child_pid']:
        raise RuntimeError('Not an idle closed-stage wait; refusing duplicate work.')
    # The old watcher must already be stopped, with its identity verified by the launcher.
    if Path('/proc/%s' % previous['pid']).exists():
        raise RuntimeError('Old watcher still exists.')
    original_pick = runtime.pick_gpus
    (gpu,) = original_pick(1)
    free = int(subprocess.check_output(['nvidia-smi', '-i', gpu, '--query-gpu=memory.free',
                                       '--format=csv,noheader,nounits'], text=True).strip())
    if free < 22000:
        raise RuntimeError('Single-GPU trial requires at least 22000 MiB free.')
    archive = root.with_name('r0-dual-wait-' + time.strftime('%Y%m%d-%H%M%S'))
    root.rename(archive)
    watch = supervisor.Watch(root)
    for name in ('data-review.json', 'smoke-validation.json', 'train-validation.json',
                 'export-validation.json', 'observe-diagnostic.json'):
        if (archive / name).exists():
            shutil.copy2(archive / name, root / name)
    watch.state['completed'] = previous['completed']
    watch.save(phase='closed', step='single-gpu-closed', gpus=[gpu], allocation='model and simulator share one initially exclusive GPU',
               recovered_from=str(archive), child_pid=os.getpid())
    stage = runtime.Stage('closed', 'r0-nav-single')
    watch.save(closed_dir=str(stage.dir))
    # Only allocation changes: original stage_closed starts/stops services and runs all 20 routes.
    def pick_shared(required):
        if required != 2 or original_pick(1)[0] != gpu:
            raise RuntimeError('GPU availability changed before model startup.')
        return gpu, gpu
    runtime.pick_gpus = pick_shared
    stop = threading.Event()
    def monitor():
        peak = 0
        with (root / 'gpu-memory.jsonl').open('a') as log:
            while not stop.is_set():
                try:
                    used = int(subprocess.check_output(['nvidia-smi', '-i', gpu, '--query-gpu=memory.used',
                                                        '--format=csv,noheader,nounits'], text=True).strip())
                    peak = max(peak, used)
                    log.write(json.dumps(dict(time=time.strftime('%F %T'), gpu=gpu, used_mib=used, peak_mib=peak)) + '\n')
                    log.flush()
                    watch.save(peak_gpu_used_mib=peak)
                except Exception as exc:
                    print('Telemetry error:', str(exc), flush=True)
                stop.wait(10)
    thread = threading.Thread(target=monitor, daemon=True)
    thread.start()
    try:
        args = argparse.Namespace(run='r0', model='nav', n=20,
                                 compare=str(zbc / 'workspace/stages/paired-repair-v6-20261019-052025-1019-052025/run-base'))
        runtime.stage_closed(stage, args)
        stage.finish(True)
        stop.set()
        thread.join()
        watch.verify('closed', stage.dir)
        watch.save(status='COMPLETED', phase='complete', child_pid=None)
    except Exception as exc:
        stop.set()
        thread.join()
        stage.finish(False, str(exc))
        watch.save(status='FAILED', error=str(exc))
        raise


if __name__ == '__main__':
    main()
