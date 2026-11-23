"""Reuse a deep artifact check only while its complete filesystem identity holds.

This is an operator-owned verification cache, not a signature or a new trust
boundary. Artifact owners remain trusted. Content, inode, metadata, membership,
link and pin changes invalidate it. Source verification is never cached here.
"""
from __future__ import annotations
import hashlib
import json
import os
from pathlib import Path
import stat
import tempfile

from preflight import PreflightError, require

MAX_CACHE_BYTES = 32 * 1024 * 1024


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=True).encode()


def artifact_identity(manifest):
    # Interpreter/source revisions do not change already-verified model/SDK
    # bytes. The caller independently validates the entire current manifest,
    # source tree, package versions and native executable on every startup.
    return {
        'image': manifest['artifacts']['image'],
        'sdkLockSha256': manifest['artifacts']['sdkLockSha256'],
        'runtimeTreeSha256': manifest['artifacts']['runtimeTreeSha256'],
        'models': manifest['models'], 'pythonVersion': manifest['pythonVersion'],
        'platform': manifest['platform'], 'inference': manifest['inference'],
        'workerConfig': manifest['workerConfig'],
    }


def metadata_snapshot(roots):
    records = {}
    for label, root in sorted(roots.items()):
        root = Path(root)
        require(root.is_absolute() and root.is_dir() and not root.is_symlink(),
                'Missing regular artifact directory')
        for path in [root, *sorted(root.rglob('*'))]:
            s = path.lstat()
            require(stat.S_ISREG(s.st_mode) or stat.S_ISDIR(s.st_mode) or stat.S_ISLNK(s.st_mode),
                    'Artifact contains a non-regular entry')
            relative = path.relative_to(root).as_posix()
            records[label + '/' + relative] = [s.st_dev, s.st_ino, s.st_mode,
                s.st_uid, s.st_gid, s.st_nlink, s.st_size, s.st_mtime_ns, s.st_ctime_ns,
                str(path.readlink()) if stat.S_ISLNK(s.st_mode) else None]
    require(bool(records), 'Empty artifact metadata')
    return records


def checked_path(filename):
    path = Path(filename)
    require(path.is_absolute() and path.resolve() == path, 'Cache path must be absolute without links')
    parent = path.parent.stat()
    require(stat.S_ISDIR(parent.st_mode) and parent.st_uid == os.geteuid() and
            parent.st_mode & 0o022 == 0, 'Cache parent must be owner-controlled')
    require(not path.is_symlink(), 'Cache cannot be a symbolic link')
    return path


def unique_object(pairs):
    result = {}
    for key, value in pairs:
        require(key not in result, 'Duplicate cache key')
        result[key] = value
    return result


def verify_cache(filename, identity, roots):
    path = checked_path(filename)
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
    with os.fdopen(fd, 'rb') as stream:
        info = os.fstat(stream.fileno())
        require(stat.S_ISREG(info.st_mode) and info.st_uid == os.geteuid() and
                info.st_mode & 0o777 == 0o600 and info.st_nlink == 1,
                'Cache must be a private owned regular file')
        require(info.st_size <= MAX_CACHE_BYTES, 'Artifact cache is too large')
        body = stream.read(MAX_CACHE_BYTES + 1)
    require(len(body) <= MAX_CACHE_BYTES, 'Artifact cache is too large')
    try:
        record = json.loads(body, object_pairs_hook=unique_object)
    except (ValueError, UnicodeError):
        raise PreflightError('Invalid artifact verification cache') from None
    require(isinstance(record, dict) and set(record) == {'cacheVersion', 'identity', 'metadata', 'recordSha256'},
            'Invalid artifact cache structure')
    require(type(record['cacheVersion']) is int and record['cacheVersion'] == 1,
            'Unsupported artifact cache version')
    payload = {key: value for key, value in record.items() if key != 'recordSha256'}
    require(record['recordSha256'] == hashlib.sha256(canonical(payload)).hexdigest(),
            'Artifact cache integrity differs')
    require(record['identity'] == identity, 'Artifact pins changed; deep verification required')
    require(record['metadata'] == metadata_snapshot(roots),
            'Artifact files changed; deep verification required')


def prime_cache(filename, identity, roots, deep_check):
    path = checked_path(filename)
    before = metadata_snapshot(roots)
    deep_check()
    require(before == metadata_snapshot(roots), 'Artifacts changed during deep verification')
    payload = {'cacheVersion': 1, 'identity': identity, 'metadata': before}
    record = {**payload, 'recordSha256': hashlib.sha256(canonical(payload)).hexdigest()}
    body = canonical(record) + b'\n'
    require(len(body) <= MAX_CACHE_BYTES, 'Artifact cache is too large')
    fd, temporary = tempfile.mkstemp(prefix='.glm-verified-', dir=path.parent)
    try:
        with os.fdopen(fd, 'wb') as stream:
            stream.write(body); stream.flush(); os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary): os.unlink(temporary)
