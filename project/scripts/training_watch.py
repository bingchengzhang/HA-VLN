"""监督既有 stage.py；保留原始数据，失败时停在可检查的阶段。"""
import argparse
import collections
import fcntl
import hashlib
import importlib.util
import json
import os
import re
from pathlib import Path
import select
import subprocess
import sys
import time

PROJECT = Path(os.environ['PROJECT'])
ZBC = Path(os.environ['ZBC'])
HPY = os.environ['HPY']
sys.path.insert(0, str(PROJECT))
QUARANTINE = {'7901': 'Outdoor instruction conflicts with verified indoor official goal.'}


def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix('.partial')
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False))
    temporary.replace(path)


def load_script(name):
    spec = importlib.util.spec_from_file_location(name, PROJECT / 'scripts' / (name + '.py'))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def stationary_endpoints(rows, hashes):
    """Keep original first/last rows when only the current step counter changes."""
    groups = collections.defaultdict(list)
    for index, row in enumerate(rows):
        messages = [dict(message) for message in row['messages']]
        for message in messages:
            if message['role'] == 'user':
                message['content'] = re.sub(
                    r'(?m)^(Progress: step )\d+( of \d+, walked [0-9.]+ m so far\.)$',
                    r'\g<1><STEP>\2', message['content'])
        key = (str(row['episode_id']), json.dumps(messages, sort_keys=True, ensure_ascii=False),
               tuple(hashes[path] for path in row['images']))
        groups[key].append(index)
    selected = {index for indices in groups.values() for index in (indices[0], indices[-1])}
    return [row for index, row in enumerate(rows) if index in selected]


def curate(raw, reviewed, split):
    """保留原始数据；检查标签并去除完全重复和仅计步变化的静止重复。"""
    raw, reviewed, split = map(Path, (raw, reviewed, split))
    if reviewed.exists() or split.exists():
        raise RuntimeError('Reviewed output exists; inspect it instead of overwriting.')
    checker = load_script('label_sheet')
    generator = load_script('gen_decision_data')
    shards, hashes, seen = {}, {}, set()
    episodes, kinds = collections.Counter(), collections.Counter()
    removed = collections.Counter()
    total = 0
    rejected = []
    issue_counts = collections.Counter()
    for source in sorted((raw / 'shards').glob('*.jsonl')):
        kept = []
        for line_number, line in enumerate(source.read_text().splitlines(), 1):
            if not line.strip():
                continue
            row = json.loads(line)
            total += 1
            if str(row['episode_id']) in QUARANTINE:
                removed['quarantined'] += 1
                continue
            issues = checker.check(row)
            if issues:
                # 原始采集保留；无框停止题等无法监督的行不进入训练。
                removed['invalid_labels'] += 1
                issue_counts.update(issues)
                rejected.append(dict(source=str(source), line=line_number, issues=issues, row=row))
                continue
            row['images'] = [str((raw / p).resolve()) for p in row['images']]
            for path in row['images']:
                if path not in hashes:
                    hashes[path] = hashlib.sha256(Path(path).read_bytes()).hexdigest()
            key = (json.dumps(row['messages'], sort_keys=True, ensure_ascii=False),
                   tuple(hashes[p] for p in row['images']))
            if key in seen:
                removed['exact_duplicates'] += 1
                continue
            seen.add(key)
            kept.append(row)
        endpoints = stationary_endpoints(kept, hashes)
        removed['stationary_repeats'] += len(kept) - len(endpoints)
        for row in endpoints:
            episodes[str(row['episode_id'])] += 1
            label = row['raw']['label']
            kinds['stop' if label['stop'] else 'observe' if label.get('observe') else 'choose'] += 1
        shards[source.name] = endpoints
    invalid_rate = removed['invalid_labels'] / max(1, total - removed['quarantined'])
    if invalid_rate > 0.05:
        raise RuntimeError('Invalid label rate %.3f exceeds 0.05: %s' % (invalid_rate, dict(issue_counts)))
    if not episodes or any(kinds[k] == 0 for k in ('choose', 'observe', 'stop')):
        raise RuntimeError('No usable examples or a required decision class is absent.')
    (reviewed / 'shards').mkdir(parents=True)
    (reviewed / 'rejected-labels.jsonl').write_text(
        '\n'.join(json.dumps(r, ensure_ascii=False) for r in rejected))
    for name, rows in shards.items():
        (reviewed / 'shards' / name).write_text('\n'.join(json.dumps(r, ensure_ascii=False) for r in rows))
    generator.write_split([reviewed], split)
    heldout = set(json.loads((split / 'heldout_scans.json').read_text()))
    train_rows = json.loads((split / 'decision_train.json').read_text())
    val_rows = json.loads((split / 'decision_val_full.json').read_text())
    image_scene = {p: generator._scene(r) for rows in shards.values() for r in rows for p in r['images']}
    train_scenes = {image_scene[p] for r in train_rows for p in r['images']}
    if train_scenes & heldout or not val_rows or not train_rows:
        raise RuntimeError('Scene split validation failed.')
    audit = dict(raw=str(raw), raw_samples=total, kept_samples=sum(episodes.values()),
                 kept_episodes=len(episodes), label_kinds=dict(kinds), removed=dict(removed),
                 quarantine=QUARANTINE, maximum_samples_per_episode=max(episodes.values()),
                 stationary_rule='Keep original first and last; normalize only current Progress step; same episode, all messages/labels and all image bytes otherwise identical.',
                 invalid_label_rate=invalid_rate, invalid_label_issues=dict(issue_counts),
                 train=len(train_rows), val=len(val_rows), heldout_scans=sorted(heldout),
                 scene_overlap=[], image_paths=len(hashes))
    write_json(reviewed / 'review.json', audit)
    return audit


class Watch:
    def __init__(self, root):
        self.root = root
        root.mkdir(parents=True, exist_ok=True)
        self.lock = (root / 'watch.lock').open('w')
        fcntl.flock(self.lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if (root / 'status.json').exists():
            raise RuntimeError('Existing supervision state must be inspected before restarting.')
        self.state = dict(status='RUNNING', pid=os.getpid(), completed={}, started=time.strftime('%F %T'))
        self.save()

    def save(self, **changes):
        self.state.update(changes, updated=time.strftime('%F %T'))
        write_json(self.root / 'status.json', self.state)
        print(json.dumps(self.state, ensure_ascii=False), flush=True)

    def observe(self, process=None, pid=None, interval=50):
        # Python3.8 没有 pidfd_open；用启动时间区分同一个 PID 的不同进程。
        def identity():
            try:
                fields = Path('/proc/%s/stat' % pid).read_text().rsplit(')', 1)[1].split()
                return None if fields[0] == 'Z' else fields[19]
            except FileNotFoundError:
                return None
        descriptor = os.pidfd_open(pid) if pid is not None and hasattr(os, 'pidfd_open') else None
        started = identity() if pid is not None and descriptor is None else None
        try:
            while True:
                if descriptor is not None:
                    if select.select([descriptor], [], [], interval)[0]:
                        break
                elif pid is not None:
                    if started is None or identity() != started:
                        break
                    time.sleep(interval)
                else:
                    try:
                        return process.wait(timeout=interval)
                    except subprocess.TimeoutExpired:
                        pass
                self.save(child_pid=pid if pid is not None else process.pid)
        finally:
            if descriptor is not None:
                os.close(descriptor)

    def verify(self, phase, directory):
        status = json.loads((directory / 'status.json').read_text())
        if status['stage'] != 'DONE':
            raise RuntimeError('%s failed: %s; report %s' % (phase, status, directory / 'report.md'))
        self.state['completed'][phase] = str(directory)
        self.save()

    def wait_for_gpus(self, required):
        """沿用 stage.py 的空闲卡判断；资源不足时等待，不启动阶段。"""
        runtime = load_script('stage')
        while True:
            try:
                free = runtime.pick_gpus(required)
            except runtime.StageError as exc:
                self.save(status='WAITING_FOR_GPUS', required_gpus=required, resource_message=str(exc))
                time.sleep(50)
                continue
            self.save(status='RUNNING', required_gpus=required, resource_message=None)
            return free

    def run_stage(self, phase, arguments):
        self.save(phase=phase, child_pid=None)
        self.wait_for_gpus(1 if arguments[0] == 'offline' else 2)
        before = set((ZBC / 'workspace/stages').iterdir())
        with (self.root / (phase + '.log')).open('w') as log:
            process = subprocess.Popen([HPY, 'scripts/stage.py'] + arguments,
                                       cwd=PROJECT, stdout=log, stderr=subprocess.STDOUT)
            self.save(child_pid=process.pid)
            code = self.observe(process=process)
        new = set((ZBC / 'workspace/stages').iterdir()) - before
        candidates = [p for p in new if p.name.startswith(arguments[0] + '-')]
        if code != 0 or len(candidates) != 1:
            raise RuntimeError('%s exited %s; inspect %s' % (phase, code, self.root / (phase + '.log')))
        self.verify(phase, candidates[0])
        return candidates[0]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--collect-pid', type=int)
    parser.add_argument('--collect-dir', type=Path)
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--audit-pilot', action='store_true')
    args = parser.parse_args()
    dataset = ZBC / 'workspace/decision_data'
    if args.audit_pilot:
        print(json.dumps(curate(dataset / 'pilot-repair-v6-20261019',
                               ZBC / 'workspace/diag/pilot-reviewed',
                               ZBC / 'workspace/diag/pilot-reviewed-split'), indent=2))
        return
    commands = [
        ('base-offline', ['offline', '--run', 'r0', '--model', 'base', '--data', 'r0', '--tag', 'r0-base-reviewed']),
        ('train', ['train', '--run', 'r0']),
        ('nav-offline', ['offline', '--run', 'r0', '--model', 'nav', '--data', 'r0', '--tag', 'r0-nav-reviewed']),
        ('closed', ['closed', '--run', 'r0', '--model', 'nav', '--n', '20', '--tag', 'r0-nav-reviewed',
                    '--compare', str(ZBC / 'workspace/stages/paired-repair-v6-20261019-052025-1019-052025/run-base')])]
    if args.dry_run:
        print(json.dumps(dict(adopt_collect_pid=args.collect_pid, collect_dir=str(args.collect_dir),
                              quarantine=QUARANTINE, commands=commands), indent=2))
        return
    if not args.collect_pid or not args.collect_dir:
        parser.error('Provide the existing collection PID and directory.')
    watch = Watch(ZBC / 'workspace/supervision/r0')
    try:
        watch.save(phase='collect', collection_dir=str(args.collect_dir))
        if json.loads((args.collect_dir / 'status.json').read_text())['stage'] != 'DONE':
            watch.observe(pid=args.collect_pid)
        watch.verify('collect', args.collect_dir)
        watch.save(phase='curate')
        audit = curate(dataset / 'r0-raw', dataset / 'r0-reviewed', dataset / 'r0-split')
        write_json(watch.root / 'data-review.json', audit)
        for phase, arguments in commands:
            watch.run_stage(phase, arguments)
        watch.save(status='COMPLETED', phase='complete', child_pid=None)
    except Exception as exc:
        watch.save(status='FAILED', error=str(exc))
        raise


if __name__ == '__main__':
    main()
