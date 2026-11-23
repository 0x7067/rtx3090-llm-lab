"""Stage the pinned adapter and overlay without fetching or copying weights."""
import argparse
import hashlib
import io
import json
from pathlib import Path
import shutil
import subprocess
import tarfile
import tempfile


def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def safe_relative(relative):
    path = Path(relative)
    if path.is_absolute() or '..' in path.parts or not path.parts:
        raise ValueError(f'Invalid relative path: {relative}')
    return path


def verified_files(root, files):
    for relative, expected in files.items():
        path = safe_relative(relative)
        source = root / path
        if digest(source) != expected:
            raise ValueError(f'Checksum mismatch: {source}')
        yield path, source


def main():
    here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--adapter', type=Path, required=True)
    parser.add_argument('--llama-source', type=Path, required=True)
    parser.add_argument('--overlay', type=Path, required=True)
    parser.add_argument('--model-config', type=Path, required=True)
    parser.add_argument('--chat-template', type=Path, required=True)
    parser.add_argument('--short-runtime', type=Path, required=True)
    parser.add_argument('--short-draft', type=Path, required=True)
    parser.add_argument('--output', type=Path, default=here / '.payload')
    args = parser.parse_args()
    lock = json.loads((here / 'sources.json').read_text())
    output = args.output
    if output.exists() or output.is_symlink():
        raise FileExistsError(f'Existing payload must be inspected before replacement: {output}')

    adapter = args.adapter / 'manifest.json'
    if digest(adapter) != lock['adapter_manifest_sha256']:
        raise ValueError('Adapter manifest differs from reviewed sources.json')
    plugin_files = list(verified_files(args.adapter / 'package', json.loads(adapter.read_text())['files']))
    model_files = list(verified_files(args.model_config, lock['model_metadata']))
    if digest(args.chat_template) != lock['chat_template_sha256']:
        raise ValueError('Chat template differs from reviewed sources.json')
    overlay = json.loads((args.overlay / 'manifest.json').read_text())
    if overlay != lock['overlay']:
        raise ValueError('Overlay manifest differs from reviewed sources.json')
    overlay_files = list(verified_files(args.overlay, {p: h['patched_sha256'] for p, h in overlay.items()}))
    # These libraries are read-only host mounts. Verify the same bytes used
    # during qualification without adding a second CUDA runtime to the image.
    list(verified_files(args.adapter.parent, lock['native_host_libraries']))
    short = lock['short_profile']
    if {file.name for file in args.short_runtime.iterdir()} != set(short['runtime_files']):
        raise ValueError('Short runtime file set differs from reviewed sources.json')
    list(verified_files(args.short_runtime, short['runtime_files']))
    if digest(args.short_draft) != short['draft_sha256']:
        raise ValueError('Short draft differs from reviewed sources.json')

    archive = subprocess.check_output([
        'git', '-C', str(args.llama_source), 'archive', lock['gguf_reader_commit'],
        'gguf-py/gguf', 'LICENSE',
    ])
    with tempfile.TemporaryDirectory(dir=output.parent, prefix='.orca-payload-') as temporary:
        payload = Path(temporary) / 'payload'
        for prefix, entries in (('plugin', plugin_files), ('overlay', overlay_files), ('model', model_files)):
            for relative, source in entries:
                target = payload / prefix / relative
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(source, target)
        shutil.copyfile(args.chat_template, payload / 'chat_template.jinja')
        with tarfile.open(fileobj=io.BytesIO(archive)) as source:
            for member in source.getmembers():
                if not member.isfile():
                    continue
                path = safe_relative(member.name)
                relative = path if path == Path('LICENSE') else path.relative_to('gguf-py')
                target = payload / 'gguf-source' / relative
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(source.extractfile(member).read())
        files = {str(path.relative_to(payload)): digest(path)
                 for path in sorted(payload.rglob('*')) if path.is_file()}
        (payload / 'payload-sha256.json').write_text(json.dumps(files, indent=2) + '\n')
        payload.rename(output)
    print(json.dumps({'payload': str(output), 'files': len(files),
                      'bytes': sum(p.stat().st_size for p in output.rglob('*') if p.is_file())}))


if __name__ == '__main__':
    main()
