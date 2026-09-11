# -------------------------------------------------------------
# @file          __init__.py
# @author        Priyangkar Ghosh
# @created       2025-06-18
# @description   Initializes the tlang package
# @license       MIT
# -------------------------------------------------------------

import logging
logger = logging.getLogger(__name__)

from importlib import metadata as _metadata
from pathlib import Path as _Path

try:
    __version__ = _metadata.version('tlang')
except _metadata.PackageNotFoundError:
    # Not installed via pip/uv (e.g. run straight from a checkout with no
    # egg-info) -- there's no distribution metadata to read the version from.
    __version__ = 'unknown'


def build_info() -> dict:
    """Answer "what am I actually running?" -- the question that isn't
    answerable from `__version__` alone. A stale, non-editable copy of this
    package (e.g. left behind in a venv's site-packages after the source
    moved on) can report a version number and still resolve imports, so the
    version alone can't tell you whether an install is current.

    Returns a dict with:
      - "version": same string as `tlang.__version__`.
      - "package_dir": the directory this package was actually imported
        from (``Path(tlang.__file__).parent``) -- the load-bearing fact,
        since two installs can share a version but live in different places.
      - "editable": best-effort guess at whether this is an editable/source
        checkout (True) vs. a copied site-packages install (False), based on
        whether `package_dir` sits under a `site-packages` directory.
    """
    package_dir = _Path(__file__).parent.resolve()
    editable = 'site-packages' not in package_dir.parts
    return {
        'version': __version__,
        'package_dir': package_dir,
        'editable': editable,
    }


from .compiler.shader_manager import ShaderManager
from .compiler.shader import Shader
from .runtime.kernel import Kernel
from .runtime.pipeline import Pipeline
from .runtime.buffer_pool import BufferPool, BufferPoolMetrics, TempHandle, clear_buffer
from .runtime.pinned_buffer import PinnedBuffer, PinnedBufferFallback, buffer_storage_supported
from .runtime.debug_log import DebugLog, DebugLogResult
from .frontend.interface_registry import (
    ExternConst,
    InterfaceDecl,
    InterfaceKind,
    InterfaceMember,
    location_span,
    member_locations,
)
from .errors import (
    SourceLocation,
    TlangError,
    TlangSyntaxError,
    TlangAttributeError,
    TlangDependencyError,
    TlangBindingError,
    TlangCompileError,
    TlangLinkError,
    TlangBuildError,
)

__all__ = [
    "__version__",
    "build_info",
    "ShaderManager",
    "Shader",
    "Kernel",
    "Pipeline",
    "BufferPool",
    "BufferPoolMetrics",
    "TempHandle",
    "clear_buffer",
    "PinnedBuffer",
    "PinnedBufferFallback",
    "buffer_storage_supported",
    "DebugLog",
    "DebugLogResult",
    "ExternConst",
    "InterfaceDecl",
    "InterfaceKind",
    "InterfaceMember",
    "location_span",
    "member_locations",
    "SourceLocation",
    "TlangError",
    "TlangSyntaxError",
    "TlangAttributeError",
    "TlangDependencyError",
    "TlangBindingError",
    "TlangCompileError",
    "TlangLinkError",
    "TlangBuildError",
]