"""
HTTP/2 header encoding for Python.
"""
from __future__ import annotations

from .block import (
    DynamicTableSizeUpdate,
    HeaderBlock,
    IndexedHeaderField,
    IndexedName,
    IndexingMode,
    LiteralHeaderField,
    LiteralName,
    Representation,
    WireString,
)
from .exceptions import (
    HPACKDecodingError,
    HPACKError,
    InvalidBlockEditError,
    InvalidTableIndex,
    InvalidTableIndexError,
    InvalidTableSizeError,
    OversizedHeaderListError,
)
from .hpack import Decoder, Encoder
from .struct import HeaderTuple, NeverIndexedHeaderTuple

__all__ = [
    "Decoder",
    "DynamicTableSizeUpdate",
    "Encoder",
    "HPACKDecodingError",
    "HPACKError",
    "HeaderBlock",
    "HeaderTuple",
    "IndexedHeaderField",
    "IndexedName",
    "IndexingMode",
    "InvalidBlockEditError",
    "InvalidTableIndex",
    "InvalidTableIndexError",
    "InvalidTableSizeError",
    "LiteralHeaderField",
    "LiteralName",
    "NeverIndexedHeaderTuple",
    "OversizedHeaderListError",
    "Representation",
    "WireString",
]

__version__ = "4.3.0+dev"
