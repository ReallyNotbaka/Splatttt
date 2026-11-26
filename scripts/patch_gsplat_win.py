"""Make gsplat 1.5.3 usable on Windows (JIT build + cached .pyd reload).

gsplat 1.5.3 predates any Windows CI, so its ``gsplat/cuda/_backend.py`` makes
three assumptions that only hold on Linux/macOS. This script patches the
*installed* package in place (it never touches the repo or ``pyproject.toml``)
and is safe to re-run.

The three fixes:

1. ``-Wno-attributes`` is a GCC flag. MSVC rejects it with ``D8021``, which
   kills every host ``.cpp`` translation unit.
2. The cached-build probe only looks for ``*.so`` / ``*.lib``. On Windows the
   extension is a ``*.pyd``, so gsplat re-enters the JIT builder (and then
   needs ``cl.exe`` on PATH) even though a good binary already exists.
3. ``cuda_toolkit_available()`` stats ``bin/nvcc``; on Windows the binary is
   ``bin/nvcc.exe``, so the toolkit is reported missing and ``_C`` becomes
   ``None``.

Usage:
    python scripts/patch_gsplat_win.py          # apply / verify
    python scripts/patch_gsplat_win.py --check  # report only, exit 1 if unpatched
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

OLD_CFLAGS = '        extra_cflags = [opt_level, "-Wno-attributes"]'
NEW_CFLAGS = (
    '        extra_cflags = '
    '[opt_level] if os.name == "nt" else [opt_level, "-Wno-attributes"]'
)

OLD_CACHE_BLOCK = '''        if os.path.exists(os.path.join(build_dir, f"{name}.so")) or os.path.exists(
            os.path.join(build_dir, f"{name}.lib")
        ):
            # If the build exists, we assume the extension has been built
            # and we can load it.
            _C = load_extension(
                name=name,
                sources=sources,
                extra_cflags=extra_cflags,
                extra_cuda_cflags=extra_cuda_cflags,
                extra_include_paths=extra_include_paths,
                build_directory=build_dir,
                verbose=VERBOSE,
            )'''

NEW_CACHE_BLOCK = '''        _built_ext = next(
            (
                ext
                for ext in (".so", ".pyd", ".lib")
                if os.path.exists(os.path.join(build_dir, f"{name}{ext}"))
            ),
            None,
        )
        if _built_ext is not None:
            # If the build exists, we assume the extension has been built and we
            # can load it. On Windows the artifact is a .pyd, so import it
            # directly instead of re-entering the JIT builder (which would
            # require a compiler on PATH).
            if _built_ext in (".so", ".pyd"):
                _C = _import_module_from_library(name, build_dir, True)
            else:
                _C = load_extension(
                    name=name,
                    sources=sources,
                    extra_cflags=extra_cflags,
                    extra_cuda_cflags=extra_cuda_cflags,
                    extra_include_paths=extra_include_paths,
                    build_directory=build_dir,
                    verbose=VERBOSE,
                )'''

OLD_NVCC = '''    nvcc_path = os.path.join(cuda_home, "bin", "nvcc")
    if not os.path.isfile(nvcc_path):'''
NEW_NVCC = '''    nvcc_path = os.path.join(cuda_home, "bin", "nvcc")
    if not os.path.isfile(nvcc_path) and not os.path.isfile(nvcc_path + ".exe"):'''

MARKERS = (
    'extra_cflags = [opt_level] if os.name == "nt"',
    "_built_ext = next(",
    'nvcc_path + ".exe"',
)


def _backend_path() -> Path:
    import gsplat  # noqa: WPS433 - deliberately lazy

    return Path(gsplat.__file__).resolve().parent / "cuda" / "_backend.py"


def patch_text(text: str) -> tuple[str, list[str]]:
    """Return (new_text, list_of_applied_fix_labels)."""
    applied: list[str] = []

    if MARKERS[0] not in text:
        if OLD_CFLAGS not in text:
            raise RuntimeError("cannot find extra_cflags line to patch")
        text = text.replace(OLD_CFLAGS, NEW_CFLAGS, 1)
        applied.append("msvc-cflags")

    if MARKERS[1] not in text:
        if OLD_CACHE_BLOCK not in text:
            raise RuntimeError("cannot find cached-build block to patch")
        text = text.replace(OLD_CACHE_BLOCK, NEW_CACHE_BLOCK, 1)
        applied.append("pyd-cache")

    if MARKERS[2] not in text:
        if OLD_NVCC not in text:
            raise RuntimeError("cannot find nvcc probe to patch")
        text = text.replace(OLD_NVCC, NEW_NVCC, 1)
        applied.append("nvcc-exe")

    return text, applied


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check",
        action="store_true",
        help="report status without writing; exit 1 if any fix is missing",
    )
    args = parser.parse_args()

    try:
        path = _backend_path()
    except ImportError:
        print("gsplat is not installed in this environment")
        return 2

    if not path.exists():
        print(f"missing {path}")
        return 2

    original = path.read_text(encoding="utf-8")
    if all(m in original for m in MARKERS):
        print(f"ALREADY_PATCHED / OK  ({path})")
        return 0

    if args.check:
        missing = [name for name, m in zip(
            ("msvc-cflags", "pyd-cache", "nvcc-exe"), MARKERS) if m not in original]
        print(f"UNPATCHED  {path}  missing={missing}")
        return 1

    try:
        patched, applied = patch_text(original)
    except RuntimeError as exc:
        print(f"FAILED: {exc}  ({path})")
        return 2

    path.write_text(patched, encoding="utf-8")
    print(f"PATCHED {applied}  ({path})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
