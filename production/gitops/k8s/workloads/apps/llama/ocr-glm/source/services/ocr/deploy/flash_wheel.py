"""Build-only download of one official, hash-pinned FlashAttention wheel."""
import argparse
from email.parser import BytesParser
import hashlib
from pathlib import Path
import urllib.request
import zipfile

VERSION = "2.8.3"
UPSTREAM_FILENAME = "flash_attn-2.8.3+cu12torch2.8cxx11abiTRUE-cp312-cp312-linux_x86_64.whl"
URL = "https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.3/" + UPSTREAM_FILENAME.replace("+", "%2B")
SHA256 = "f25da18657a87fc83dc1bfb8b7751b82246e9db355510226b674fd437c34b5fb"
SIZE = 256040057
# Upstream CI renames the wheel after building; METADATA keeps the public version.
LOCAL_FILENAME = "flash_attn-2.8.3-cp312-cp312-linux_x86_64.whl"


def validate_metadata(path: Path) -> None:
    with zipfile.ZipFile(path) as wheel:
        names = [name for name in wheel.namelist() if name.endswith(".dist-info/METADATA")]
        if len(names) != 1 or wheel.getinfo(names[0]).file_size > 1024 * 1024:
            raise ValueError("Invalid FlashAttention wheel metadata")
        metadata = BytesParser().parsebytes(wheel.read(names[0]))
        if metadata.get("Name", "").replace("_", "-").lower() != "flash-attn" or metadata.get("Version") != VERSION:
            raise ValueError("Unexpected FlashAttention distribution identity")


def download(destination: Path) -> Path:
    destination.mkdir(parents=True, exist_ok=True)
    target = destination / LOCAL_FILENAME
    pending = destination / (LOCAL_FILENAME + ".partial")
    digest = hashlib.sha256()
    size = 0
    try:
        with urllib.request.urlopen(URL, timeout=60) as response, pending.open("xb") as output:
            while chunk := response.read(1024 * 1024):
                size += len(chunk)
                if size > SIZE:
                    raise ValueError("FlashAttention wheel exceeds its recorded size")
                digest.update(chunk)
                output.write(chunk)
        if size != SIZE or digest.hexdigest() != SHA256:
            raise ValueError("FlashAttention wheel differs from its official release digest")
        validate_metadata(pending)
        pending.replace(target)
        return target
    finally:
        pending.unlink(missing_ok=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--destination", type=Path, required=True)
    args = parser.parse_args()
    print(download(args.destination))
