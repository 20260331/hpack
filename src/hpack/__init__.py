"""
HTTP/2 header encoding for Python.
"""
from __future__ import annotations

from .exceptions import (
    HPACKDecodingError,
    HPACKError,
    InvalidTableIndex,
    InvalidTableIndexError,
    InvalidTableSizeError,
    OversizedHeaderListError,
    TranscodeAbortedError,
)
from .hpack import Decoder, Encoder
from .struct import HeaderTuple, NeverIndexedHeaderTuple
from .transcoder import Transcoder

__all__ = [
    "Decoder",
    "Encoder",
    "HPACKDecodingError",
    "HPACKError",
    "HeaderTuple",
    "InvalidTableIndex",
    "InvalidTableIndexError",
    "InvalidTableSizeError",
    "NeverIndexedHeaderTuple",
    "OversizedHeaderListError",
    "TranscodeAbortedError",
    "Transcoder",
]

__version__ = "4.3.0+dev"
