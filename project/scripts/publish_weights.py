"""Publish only inference files to Hugging Face; credentials stay in the local HF store."""
import argparse
import fnmatch
import hashlib
import json
from pathlib import Path
import tempfile


COMMON = ('config.json', '*token*.json', 'chat_template.json', 'merges.txt',
          'vocab.json', 'preprocessor_config.json', 'generation_config.json',
          'LICENSE', 'LICENSE.txt', 'NOTICE')
WEIGHTS = {
    'merged': ('model*.safetensors', 'model.safetensors.index.json'),
    'adapter': ('adapter_model.safetensors', 'adapter_config.json'),
}


def checksum(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--kind', choices=WEIGHTS, required=True)
    parser.add_argument('--folder', type=Path, required=True)
    parser.add_argument('--repo', required=True, help='Hugging Face owner/model-name')
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    if len(args.repo.split('/')) != 2 or not all(args.repo.split('/')):
        parser.error('--repo must be owner/model-name')
    folder = args.folder.resolve(strict=True)
    patterns = COMMON + WEIGHTS[args.kind]
    files = sorted(p for p in folder.iterdir() if p.is_file() and not p.is_symlink()
                   and any(fnmatch.fnmatchcase(p.name, pattern) for pattern in patterns))
    required = ('config.json', 'model.safetensors.index.json') if args.kind == 'merged' else (
        'adapter_config.json', 'adapter_model.safetensors')
    names = {p.name for p in files}
    if not set(required) <= names:
        parser.error('Missing required model files: ' + ', '.join(set(required) - names))
    if args.kind == 'merged':
        index = json.loads((folder / 'model.safetensors.index.json').read_text())
        if not set(index['weight_map'].values()) <= names:
            parser.error('Model index references missing shards')
    for path in files:
        print(f'{path.name}\t{path.stat().st_size}')
    print(f'{len(files)} files, {sum(p.stat().st_size for p in files)} bytes')
    if args.dry_run:
        return

    from huggingface_hub import HfApi
    api = HfApi()
    api.whoami()  # Fail before hashing or creating a repository when login is missing.
    before = {p.name: (p.stat().st_size, p.stat().st_mtime_ns) for p in files}
    hashes = {p.name: checksum(p) for p in files}
    if any(before[p.name] != (p.stat().st_size, p.stat().st_mtime_ns) for p in files):
        raise RuntimeError('Source model changed during hashing; retry after export completes')
    card = f'''---
base_model: Qwen/Qwen2.5-VL-7B-Instruct
tags:
- ha-vln
- vision-language-navigation
- {args.kind}
---
# HA-VLN nav r0 ({args.kind})

Trained decision model for hierarchical human-aware vision-language navigation.
Code and architecture: https://github.com/bingchengzhang/HA-VLN

This is a research checkpoint. The original 20-episode closed-loop r0 evaluation
had SR=0; offline improvement and successful training do not establish navigation success.
Use the adapter with Qwen/Qwen2.5-VL-7B-Instruct, or use the merged model directly.

Follow the base model terms and the Matterport3D academic-use agreement, which
also covers models derived from its data:
https://kaldir.vc.in.tum.de/matterport/MP_TOS.pdf

SHA256SUMS contains the checksums of the published inference files.
'''
    api.create_repo(args.repo, repo_type='model', private=False, exist_ok=True)
    # Pass exact names so training logs and checkpoints cannot enter the model repository.
    api.upload_folder(repo_id=args.repo, repo_type='model', folder_path=folder,
                      allow_patterns=[p.name for p in files])
    with tempfile.TemporaryDirectory() as temporary:
        metadata = Path(temporary)
        (metadata / 'README.md').write_text(card, encoding='utf-8')
        (metadata / 'SHA256SUMS').write_text(
            ''.join(f'{hashes[p.name]}  {p.name}\n' for p in files), encoding='utf-8')
        api.upload_folder(repo_id=args.repo, repo_type='model', folder_path=metadata)
    remote = {p.rfilename: p for p in api.model_info(args.repo, files_metadata=True).siblings}
    for path in files:
        item = remote.get(path.name)
        if item is None or item.size != path.stat().st_size:
            raise RuntimeError(f'Remote size verification failed: {path.name}')
        if item.lfs and item.lfs.sha256 != hashes[path.name]:
            raise RuntimeError(f'Remote SHA256 verification failed: {path.name}')
        if before[path.name] != (path.stat().st_size, path.stat().st_mtime_ns):
            raise RuntimeError(f'Source changed during upload: {path.name}')
    print('VERIFIED https://huggingface.co/' + args.repo)


if __name__ == '__main__':
    main()
