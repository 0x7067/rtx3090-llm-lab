"""Apply only the opt-in short-DFlash launcher gate to the exact deployed source."""
import hashlib
from pathlib import Path
import sys

ORIGINAL_SHA256 = 'e8f4d78344b94410c9ea0a9a16636c6a1e38cd26e82c7ce47840d186b541dcb5'
OLD = '    if [ "${DFLASH_TOKENS:-7}" != 7 ]; then\n'
NEW = '''    # Experimental qualification only: preserve k=7 unless explicitly opted in.
    # All other PTH backend, lookup, dynamic-K and connector guards still apply.
    if [ "${DFLASH_TOKENS:-7}" != 7 ] && {
      [ "${NORMAL120_PTH_SHORT_DFLASH:-0}" != 1 ] || {
        [ "${DFLASH_TOKENS:-7}" != 3 ] && [ "${DFLASH_TOKENS:-7}" != 5 ];
      };
    }; then
'''
MARKER = '    if [ "${LOOKUP:-1}" != 1 ]; then\n'
NOTICE = '''    if [ "${DFLASH_TOKENS:-7}" != 7 ]; then
      echo "[normal120-experimental] PTH DFlash k=$DFLASH_TOKENS explicitly opted in; not qualified for production" >&2
    fi
'''

def patched(data):
    if hashlib.sha256(data).hexdigest() != ORIGINAL_SHA256:
        raise ValueError('Launcher hash mismatch; refusing patch')
    source = data.decode()
    if source.count(OLD) != 1 or source.count(MARKER) != 1:
        raise ValueError('Launcher anchors changed; refusing patch')
    return source.replace(OLD, NEW).replace(MARKER, NOTICE + MARKER).encode()

if __name__ == '__main__':
    path = Path(sys.argv[1])
    path.write_bytes(patched(path.read_bytes()))
