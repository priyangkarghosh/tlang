# -------------------------------------------------------------
# @file          __init__.py
# @author        Priyangkar Ghosh
# @created       2025-06-18
# @description   Initializes the tlang package
# @license       MIT
# -------------------------------------------------------------

import logging
logger = logging.getLogger(__name__)

from .compiler.shader_manager import ShaderManager
from .compiler.shader import Shader
from .runtime.kernel import Kernel
from .runtime.pipeline import Pipeline
from .runtime.buffer_pool import BufferPool, BufferPoolMetrics, TempHandle, clear_buffer
from .frontend.interface_registry import (
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
)

__all__ = [
    "ShaderManager",
    "Shader",
    "Kernel",
    "Pipeline",
    "BufferPool",
    "BufferPoolMetrics",
    "TempHandle",
    "clear_buffer",
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
]