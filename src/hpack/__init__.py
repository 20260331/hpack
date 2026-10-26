"""
HTTP/2 header encoding for Python.
"""
from __future__ import annotations

from .exceptions import (
    HPACKDecodingError,
    HPACKError,
    InvalidContextError,
    InvalidTableIndex,
    InvalidTableIndexError,
    InvalidTableSizeError,
    OversizedHeaderListError,
)
from .hpack import CONTEXT_VERSION, Decoder, Encoder
from .struct import HeaderTuple, NeverIndexedHeaderTuple

__all__ = [
    "CONTEXT_VERSION",
    "Decoder",
    "Encoder",
    "HPACKDecodingError",
    "HPACKError",
    "HeaderTuple",
    "InvalidContextError",
    "InvalidTableIndex",
    "InvalidTableIndexError",
    "InvalidTableSizeError",
    "NeverIndexedHeaderTuple",
    "OversizedHeaderListError",
]

__version__ = "4.3.0+dev"
