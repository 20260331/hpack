"""
Lossless in-memory representation of HPACK header blocks.

The regular :class:`hpack.Encoder`/:class:`hpack.Decoder` pair only exposes the
decoded header list: parsing a block and re-encoding it does not reproduce the
original wire format, and there is no way to change a single literal without
re-encoding everything around it.

This module parses an HPACK block (RFC 7541 section 6) into a tree of
:class:`Representation` objects that each remember the exact wire bytes they
came from. Unmodified representations are therefore serialized back
byte-for-byte, while individual literal header fields can be edited and are
re-encoded in place. Every re-serialization is replayed against a private copy
of the dynamic header table to make sure that the resulting block still obeys
the indexing and table-size rules of RFC 7541.
"""
from __future__ import annotations

import enum
from typing import TYPE_CHECKING, Self, cast

from .exceptions import (
    HPACKDecodingError,
    InvalidBlockEditError,
    InvalidTableSizeError,
    OversizedHeaderListError,
)
from .hpack import DEFAULT_MAX_HEADER_LIST_SIZE, decode_integer, encode_integer
from .huffman import HuffmanEncoder
from .huffman_constants import REQUEST_CODES, REQUEST_CODES_LENGTH
from .huffman_table import decode_huffman
from .table import HeaderTable, table_entry_size

if TYPE_CHECKING:
    from collections.abc import Iterator  # pragma: no cover

__all__ = [
    "DynamicTableSizeUpdate",
    "HeaderBlock",
    "IndexedHeaderField",
    "IndexedName",
    "IndexingMode",
    "LiteralHeaderField",
    "LiteralName",
    "Representation",
    "WireString",
]

_HUFFMAN_ENCODER = HuffmanEncoder(REQUEST_CODES, REQUEST_CODES_LENGTH)


class IndexingMode(enum.Enum):
    """
    The indexing behaviour of a literal header field representation
    (RFC 7541 sections 6.2.1 to 6.2.3).
    """

    #: Literal Header Field with Incremental Indexing (section 6.2.1, ``01``).
    INCREMENTAL = "incremental"

    #: Literal Header Field without Indexing (section 6.2.2, ``0000``).
    NEVER_INDEXED = "never_indexed"

    #: Literal Header Field Never Indexed (section 6.2.3, ``0001``).
    WITHOUT_INDEXING = "without_indexing"


#: Value of the first byte for each indexing mode when the header name is sent
#: as a literal (i.e. with a name index of zero).
_MODE_TYPE_BYTE: dict[IndexingMode, int] = {
    IndexingMode.INCREMENTAL: 0x40,
    IndexingMode.WITHOUT_INDEXING: 0x00,
    IndexingMode.NEVER_INDEXED: 0x10,
}


def _classify_literal(first_byte: int) -> tuple[IndexingMode, int]:
    """
    Classifies the first byte of a literal representation, returning the
    indexing mode and the bit width of the name-index prefix.
    """
    if first_byte & 0x40:
        return IndexingMode.INCREMENTAL, 6
    if first_byte & 0x10:
        return IndexingMode.NEVER_INDEXED, 4
    return IndexingMode.WITHOUT_INDEXING, 4


class WireString:
    """
    A single HPACK string (section 5.2) with its wire representation preserved.

    ``payload`` holds the bytes that travel on the wire (already Huffman
    encoded when ``huffman`` is true), while ``value`` holds the decoded bytes.
    ``_length_raw`` remembers the exact length-prefix octets so that an
    unmodified string is restored byte-for-byte.
    """

    __slots__ = ("_dirty", "_length_raw", "huffman", "payload", "value")

    def __init__(self, huffman: bool, length_raw: bytes, payload: bytes, value: bytes) -> None:
        self.huffman = huffman
        self._length_raw = length_raw
        self.payload = payload
        self.value = value
        self._dirty = False

    @classmethod
    def parse(cls, data: bytes, offset: int) -> tuple[Self, int]:
        """
        Reads a string representation starting at the length-prefix byte and
        returns it alongside the offset immediately past the string.
        """
        length, consumed = decode_integer(data[offset:], 7)
        huffman = bool(data[offset] & 0x80)
        length_raw = bytes(data[offset:offset + consumed])

        payload_start = offset + consumed
        payload = bytes(data[payload_start:payload_start + length])
        if len(payload) != length:
            msg = "Truncated header block"
            raise HPACKDecodingError(msg)

        value = decode_huffman(payload) if huffman else payload
        return cls(huffman, length_raw, payload, value), payload_start + length

    @property
    def dirty(self) -> bool:
        """Whether this string was modified after parsing."""
        return self._dirty

    def set(self, value: bytes, huffman: bool | None = None) -> None:
        """
        Replaces the decoded value of this string.

        :param value: The new decoded value as bytes.
        :param huffman: Whether the new value should be Huffman encoded. When
                        ``None`` (the default) the original representation is
                        kept.
        """
        if not isinstance(value, bytes):
            msg = f"Wire string values must be bytes, got {type(value).__name__}"
            raise TypeError(msg)

        if huffman is None:
            huffman = self.huffman
        self.huffman = huffman
        self.payload = _HUFFMAN_ENCODER.encode(value) if huffman else value
        self.value = value
        self._dirty = True

    def serialize(self) -> bytes:
        """Returns the wire bytes of this string."""
        if not self._dirty:
            return self._length_raw + self.payload

        length = encode_integer(len(self.payload), 7)
        if self.huffman:
            length[0] |= 0x80
        return bytes(length) + self.payload

    def __repr__(self) -> str:
        return f"WireString(huffman={self.huffman!r}, value={self.value!r})"


class IndexedName:
    """
    A literal header field name taken from the static or dynamic table
    (RFC 7541 sections 6.2.x with a non-zero name index).
    """

    __slots__ = ("index", "name", "raw")

    def __init__(self, index: int, raw: bytes, name: bytes) -> None:
        self.index = index
        #: The decoded header name referenced by the index, captured at parse
        #: time so that nodes remain self-describing.
        self.name = name
        #: The exact wire bytes of the index prefix, including the first byte
        #: that carries the representation's bit pattern.
        self.raw = raw

    @property
    def literal(self) -> bool:
        """Whether this name is a literal name rather than an index."""
        return False

    def serialize(self) -> bytes:
        """Returns the wire bytes of this name reference."""
        return self.raw

    def __repr__(self) -> str:
        return f"IndexedName(index={self.index!r})"


class LiteralName:
    """
    A literal header field name sent on the wire (a name index of zero).
    """

    __slots__ = ("wire",)

    def __init__(self, wire: WireString) -> None:
        self.wire = wire

    @property
    def literal(self) -> bool:
        """Whether this name is a literal name rather than an index."""
        return True

    @property
    def dirty(self) -> bool:
        """Whether this name was modified after parsing."""
        return self.wire.dirty

    def set(self, value: bytes, huffman: bool | None = None) -> None:
        """Replaces the decoded name, see :meth:`WireString.set`."""
        self.wire.set(value, huffman)

    def serialize(self) -> bytes:
        """Returns the wire bytes of this literal name (without type byte)."""
        return self.wire.serialize()

    def __repr__(self) -> str:
        return f"LiteralName({self.wire!r})"


class Representation:
    """
    Base class for the representations an HPACK header block is made of
    (RFC 7541 section 6).
    """

    __slots__ = ("_dirty", "end", "raw", "start")

    def __init__(self, raw: bytes, start: int) -> None:
        #: The exact wire bytes of this representation.
        self.raw = raw
        #: Offset of the first byte of this representation in the block.
        self.start = start
        #: Offset immediately past this representation in the block.
        self.end = start + len(raw)
        self._dirty = False

    @property
    def dirty(self) -> bool:
        """Whether this representation (or one of its children) was edited."""
        return self._dirty

    def serialize(self) -> bytes:
        """Returns the wire bytes of this representation."""
        raise NotImplementedError()  # noqa: RSE102


class IndexedHeaderField(Representation):
    """
    An indexed header field representation (RFC 7541 section 6.1, ``1``).
    """

    __slots__ = ("index",)

    def __init__(self, index: int, raw: bytes, start: int) -> None:
        super().__init__(raw, start)
        #: The referenced static/dynamic table index (1-based).
        self.index = index

    def serialize(self) -> bytes:
        """Returns the wire bytes of this representation."""
        return self.raw

    def __repr__(self) -> str:
        return f"IndexedHeaderField(index={self.index!r})"


class DynamicTableSizeUpdate(Representation):
    """
    A dynamic table size update (RFC 7541 section 6.3, ``001``).
    """

    __slots__ = ("new_size",)

    def __init__(self, new_size: int, raw: bytes, start: int) -> None:
        super().__init__(raw, start)
        #: The requested maximum dynamic table size.
        self.new_size = new_size

    def serialize(self) -> bytes:
        """Returns the wire bytes of this representation."""
        return self.raw

    def __repr__(self) -> str:
        return f"DynamicTableSizeUpdate(new_size={self.new_size!r})"


class LiteralHeaderField(Representation):
    """
    A literal header field representation in one of the three indexing modes
    (RFC 7541 sections 6.2.1 to 6.2.3).
    """

    __slots__ = ("name_representation", "value_representation")

    def __init__(
        self,
        name_representation: IndexedName | LiteralName,
        value_representation: WireString,
        raw: bytes,
        start: int,
    ) -> None:
        super().__init__(raw, start)
        self.name_representation = name_representation
        self.value_representation = value_representation

    @property
    def mode(self) -> IndexingMode:
        """The indexing mode of this literal, as encoded in the first byte."""
        mode, _ = _classify_literal(self.raw[0])
        return mode

    @property
    def name(self) -> bytes:
        """The decoded header field name."""
        if isinstance(self.name_representation, IndexedName):
            return self.name_representation.name
        return self.name_representation.wire.value

    @property
    def value(self) -> bytes:
        """The decoded header field value."""
        return self.value_representation.value

    def set_name(self, name: bytes, huffman: bool | None = None) -> None:
        """
        Replaces the value of a *literal* header name.

        A name that was sent as a table index cannot be edited in place: doing
        so would change the representation rather than one literal.
        """
        if not isinstance(self.name_representation, LiteralName):
            msg = "Cannot edit an indexed header name; edit the value or use a literal-name field"
            raise InvalidBlockEditError(msg)

        self.name_representation.set(name, huffman)
        self._dirty = True

    def set_value(self, value: bytes, huffman: bool | None = None) -> None:
        """
        Replaces the literal header value.

        :param value: The new decoded value as bytes.
        :param huffman: Whether the new value should be Huffman encoded. When
                        ``None`` (the default) the original representation is
                        kept.
        """
        self.value_representation.set(value, huffman)
        self._dirty = True

    def serialize(self) -> bytes:
        """
        Returns the wire bytes of this representation, rebuilding it only when
        the name or value was edited.
        """
        if not self._dirty:
            return self.raw

        name_repr = self.name_representation
        if isinstance(name_repr, IndexedName):
            # Preserve the original name-index octets verbatim; only the value
            # can have changed for an indexed-name literal.
            return name_repr.serialize() + self.value_representation.serialize()

        mode, _ = _classify_literal(self.raw[0])
        type_byte = _MODE_TYPE_BYTE[mode]
        return bytes([type_byte]) + name_repr.serialize() + self.value_representation.serialize()

    def __repr__(self) -> str:
        return f"LiteralHeaderField(mode={self.mode!r}, name={self.name!r}, value={self.value!r})"


def _parse_literal(
    table: HeaderTable,
    data: bytes,
    start: int,
    offset: int,
) -> tuple[LiteralHeaderField, bytes, bytes, int]:
    """
    Parses one literal header field representation beginning at ``start`` and
    returns the node, the resolved name/value bytes and the offset past it.
    """
    first = data[start]
    mode, name_prefix_bits = _classify_literal(first)
    indexed_bits = first & (0x3F if mode is IndexingMode.INCREMENTAL else 0x0F)

    if indexed_bits:
        index, consumed = decode_integer(data[offset:], name_prefix_bits)
        name = table.get_by_index(index)[0]
        name_repr: IndexedName | LiteralName = IndexedName(
            index, bytes(data[offset:offset + consumed]), name,
        )
        offset += consumed
    else:
        # The type byte is consumed; the literal name follows.
        offset += 1
        name_wire, offset = WireString.parse(data, offset)
        name_repr = LiteralName(name_wire)
        name = name_wire.value

    value_wire, offset = WireString.parse(data, offset)
    value = value_wire.value

    if mode is IndexingMode.INCREMENTAL:
        table.add(name, value)

    raw = bytes(data[start:offset])
    return LiteralHeaderField(name_repr, value_wire, raw, start), name, value, offset


def _clone_header_table(table: HeaderTable) -> HeaderTable:
    """
    Builds an independent copy of a header table's dynamic state so that edit
    validation never touches the parsing context.
    """
    clone = HeaderTable()
    clone.maxsize = table.maxsize
    # ``dynamic_entries`` is ordered newest-first, so entries must be re-added
    # oldest-first to reproduce the exact same table and eviction state.
    for name, value in reversed(table.dynamic_entries):
        clone.add(name, value)
    return clone


def _validate(
    nodes: tuple[Representation, ...],
    initial_table: HeaderTable,
    max_allowed_table_size: int,
    max_header_list_size: int,
) -> None:
    """
    Replays the parsed (and possibly edited) representations against a private
    dynamic table copy, enforcing the same indexing and table-size rules as the
    regular decoder.
    """
    table = _clone_header_table(initial_table)
    headers_seen = 0
    inflated_size = 0

    for node in nodes:
        if isinstance(node, DynamicTableSizeUpdate):
            if headers_seen:
                msg = "Table size update not at the start of the block"
                raise HPACKDecodingError(msg)
            if node.new_size > max_allowed_table_size:
                msg = "Encoder exceeded max allowable table size"
                raise InvalidTableSizeError(msg)
            table.maxsize = node.new_size
            continue

        if isinstance(node, IndexedHeaderField):
            name, value = table.get_by_index(node.index)
        else:
            # The only other header-carrying representation is a literal.
            literal = cast("LiteralHeaderField", node)
            name_repr = literal.name_representation
            if isinstance(name_repr, IndexedName):
                name = table.get_by_index(name_repr.index)[0]
            else:
                name = name_repr.wire.value
            value = literal.value_representation.value

            if literal.mode is IndexingMode.INCREMENTAL:
                table.add(name, value)

        headers_seen += 1
        inflated_size += table_entry_size(name, value)
        if inflated_size > max_header_list_size:
            msg = f"A header list larger than {max_header_list_size} has been received"
            raise OversizedHeaderListError(msg)


class HeaderBlock:
    """
    A parsed HPACK header block: the root of the lossless representation tree.

    Build one with :meth:`from_bytes`, inspect :attr:`nodes` to see which
    representation every byte segment uses, edit individual
    :class:`LiteralHeaderField` nodes, and call :meth:`to_bytes` to serialize
    the result. Unmodified blocks round-trip byte-for-byte; edited blocks are
    validated against the dynamic table rules before they are returned.
    """

    __slots__ = (
        "_initial_table",
        "_max_allowed_table_size",
        "_max_header_list_size",
        "nodes",
        "raw",
    )

    def __init__(
        self,
        raw: bytes,
        nodes: tuple[Representation, ...],
        initial_table: HeaderTable,
        max_allowed_table_size: int,
        max_header_list_size: int,
    ) -> None:
        self.raw = raw
        #: The representations the block consists of, in wire order.
        self.nodes = nodes
        self._initial_table = initial_table
        self._max_allowed_table_size = max_allowed_table_size
        self._max_header_list_size = max_header_list_size

    @classmethod
    def from_bytes(
        cls,
        data: bytes | bytearray | memoryview,
        *,
        header_table: HeaderTable | None = None,
        max_allowed_table_size: int = HeaderTable.DEFAULT_SIZE,
        max_header_list_size: int = DEFAULT_MAX_HEADER_LIST_SIZE,
    ) -> Self:
        """
        Parses an HPACK-encoded header block into a lossless representation
        tree.

        :param data: The complete HPACK-encoded header block.
        :param header_table: The dynamic table state at the start of the block.
                             When omitted a fresh table is used, matching
                             :class:`hpack.Decoder`. The table is advanced as
                             the block is parsed, just like a decoder's.
        :param max_allowed_table_size: Maximum value permitted for dynamic
                                       table size updates.
        :param max_header_list_size: Maximum decompressed header list size.
        :raises HPACKDecodingError: If the block cannot be decoded.
        """
        data = bytes(data)
        table = HeaderTable() if header_table is None else header_table
        initial_table = _clone_header_table(table)

        nodes: list[Representation] = []
        offset = 0
        headers_seen = 0
        inflated_size = 0

        while offset < len(data):
            start = offset
            first = data[offset]

            if first & 0x80:
                # Indexed header field (section 6.1).
                index, consumed = decode_integer(data[offset:], 7)
                name, value = table.get_by_index(index)
                offset += consumed
                node: Representation = IndexedHeaderField(
                    index, bytes(data[start:offset]), start,
                )

            elif not (first & 0x40) and (first & 0x20):
                # Dynamic table size update (section 6.3). Only permitted
                # before any actual header field.
                if headers_seen:
                    msg = "Table size update not at the start of the block"
                    raise HPACKDecodingError(msg)

                new_size, consumed = decode_integer(data[offset:], 5)
                if new_size > max_allowed_table_size:
                    msg = "Encoder exceeded max allowable table size"
                    raise InvalidTableSizeError(msg)
                table.maxsize = new_size
                offset += consumed
                nodes.append(DynamicTableSizeUpdate(
                    new_size, bytes(data[start:offset]), start,
                ))
                continue

            else:
                # Literal header field (sections 6.2.1 to 6.2.3).
                node, name, value, offset = _parse_literal(table, data, start, offset)

            headers_seen += 1
            inflated_size += table_entry_size(name, value)
            if inflated_size > max_header_list_size:
                msg = f"A header list larger than {max_header_list_size} has been received"
                raise OversizedHeaderListError(msg)

            nodes.append(node)

        return cls(
            data,
            tuple(nodes),
            initial_table,
            max_allowed_table_size,
            max_header_list_size,
        )

    def literals(self) -> tuple[LiteralHeaderField, ...]:
        """Returns all literal header field representations in wire order."""
        return tuple(node for node in self.nodes if isinstance(node, LiteralHeaderField))

    def to_bytes(self) -> bytes:
        """
        Serializes the block back to its HPACK wire representation.

        Representations that were not edited are emitted from their original
        wire bytes, so an unmodified block is restored byte-for-byte. When an
        edit was made the rebuilt block is replayed against the dynamic table:
        if the edit would invalidate a table index, violate table-size limits
        or exceed the header list size limit,
        :class:`hpack.InvalidBlockEditError` is raised instead of emitting an
        invalid block.
        """
        if not any(node.dirty for node in self.nodes):
            return self.raw

        try:
            _validate(
                self.nodes,
                self._initial_table,
                self._max_allowed_table_size,
                self._max_header_list_size,
            )
        except HPACKDecodingError as err:
            msg = f"Edited header block would violate HPACK table/indexing rules: {err}"
            raise InvalidBlockEditError(msg) from err

        return b"".join(node.serialize() for node in self.nodes)

    def __iter__(self) -> Iterator[Representation]:
        return iter(self.nodes)

    def __len__(self) -> int:
        return len(self.nodes)

    def __getitem__(self, index: int) -> Representation:
        return self.nodes[index]

    def __repr__(self) -> str:
        return f"HeaderBlock({self.nodes!r})"
