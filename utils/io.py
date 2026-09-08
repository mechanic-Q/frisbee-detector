"""Shared I/O utilities: safe file operations with error handling."""

import shutil
from pathlib import Path


def safe_copy(src: str | Path, dst: str | Path) -> bool:
    """Copy a file with error handling. Returns True on success."""
    try:
        shutil.copy2(str(src), str(dst))
        return True
    except OSError as e:
        print(f"  ERROR copying {src} -> {dst}: {e}")
        return False


def safe_write(path: str | Path, content: str) -> bool:
    """Write content to file with error handling. Returns True on success."""
    try:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content)
        return True
    except OSError as e:
        print(f"  ERROR writing {path}: {e}")
        return False
