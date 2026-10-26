"""
Tests for the lossless HPACK block representation tree and the local editing
interface in ``hpack.block``.
"""

import json
from binascii import unhexlify
from pathlib import Path

import pytest
from hypothesis import given
from hypothesis.strategies import binary, booleans, lists, tuples

from hpack import (
    Decoder,
    DynamicTableSizeUpdate,
    Encoder,
    HPACKDecodingError,
    HeaderBlock,
    IndexedHeaderField,
    IndexedName,
    IndexingMode,
    InvalidBlockEditError,
    InvalidTableIndex,
    InvalidTableSizeError,
    LiteralHeaderField,
    LiteralName,
    OversizedHeaderListError,
    Representation,
    WireString,
)
from hpack.table import HeaderTable

FIXTURE_DIR = Path(__file__).parent / "test_fixtures"
STORY_FILES = sorted(str(path) for subdir in FIXTURE_DIR.iterdir() if subdir.is_dir() for path in subdir.glob("story_*.json"))


# Hand-crafted wire sequences used across the parsing tests.
WIRE_INDEXED_GET = b"\x82"  # indexed :method GET
WIRE_TABLE_SIZE_0 = b"\x20"  # dynamic table size update to 0
WIRE_TABLE_SIZE_128 = b"\x3f\x61"  # dynamic table size update to 128
WIRE_INCREMENTAL_INDEXED_NAME = b"\x44\x04/foo"  # :path /foo, incremental
WIRE_LITERAL_NO_INDEX = b"\x00\x03foo\x03bar"  # foo: bar, without indexing
WIRE_LITERAL_NEVER_INDEXED = b"\x10\x03foo\x03bar"  # foo: bar, never indexed
WIRE_LITERAL_EMPTY_VALUE = b"\x00\x03foo\x00"  # foo: (empty value)
# incremental literal with Huffman-encoded literal name and value
WIRE_HUFFMAN = unhexlify("408825a849e95ba97d7f8925a849e95bb8e8b4bf")
# block mixing every representation kind (size updates must come first)
WIRE_MIXED = WIRE_TABLE_SIZE_0 + WIRE_INDEXED_GET + WIRE_INCREMENTAL_INDEXED_NAME + WIRE_LITERAL_NEVER_INDEXED


class TestBlockParsing:
    def test_empty_block(self):
        block = HeaderBlock.from_bytes(b"")
        assert block.nodes == ()
        assert len(block) == 0
        assert block.to_bytes() == b""

    def test_indexed_header_field(self):
        block = HeaderBlock.from_bytes(WIRE_INDEXED_GET)
        node = block[0]
        assert isinstance(node, IndexedHeaderField)
        assert node.index == 2
        assert node.raw == b"\x82"
        assert (node.start, node.end) == (0, 1)
        assert not node.dirty

    def test_dynamic_table_size_update(self):
        block = HeaderBlock.from_bytes(WIRE_TABLE_SIZE_128)
        node = block[0]
        assert isinstance(node, DynamicTableSizeUpdate)
        assert node.new_size == 128
        assert node.raw == b"\x3f\x61"

        zero = HeaderBlock.from_bytes(WIRE_TABLE_SIZE_0)[0]
        assert isinstance(zero, DynamicTableSizeUpdate)
        assert zero.new_size == 0

    def test_literal_with_incremental_indexing_and_indexed_name(self):
        block = HeaderBlock.from_bytes(WIRE_INCREMENTAL_INDEXED_NAME)
        node = block[0]
        assert isinstance(node, LiteralHeaderField)
        assert node.mode is IndexingMode.INCREMENTAL
        assert node.name == b":path"
        assert node.value == b"/foo"
        assert isinstance(node.name_representation, IndexedName)
        assert node.name_representation.index == 4
        assert node.name_representation.raw == b"\x44"
        assert not node.name_representation.literal
        assert isinstance(node.value_representation, WireString)
        assert node.value_representation.huffman is False
        assert node.value_representation.payload == b"/foo"

    def test_literal_without_indexing_and_literal_name(self):
        block = HeaderBlock.from_bytes(WIRE_LITERAL_NO_INDEX)
        node = block[0]
        assert isinstance(node, LiteralHeaderField)
        assert node.mode is IndexingMode.WITHOUT_INDEXING
        name_repr = node.name_representation
        assert isinstance(name_repr, LiteralName)
        assert name_repr.literal is True
        assert name_repr.wire.huffman is False
        assert name_repr.wire.value == b"foo"
        assert node.value == b"bar"
        assert node.value_representation.payload == b"bar"
        assert node.raw == WIRE_LITERAL_NO_INDEX
        assert (node.start, node.end) == (0, len(WIRE_LITERAL_NO_INDEX))

    def test_literal_never_indexed(self):
        block = HeaderBlock.from_bytes(WIRE_LITERAL_NEVER_INDEXED)
        node = block[0]
        assert isinstance(node, LiteralHeaderField)
        assert node.mode is IndexingMode.NEVER_INDEXED

    def test_empty_literal_value(self):
        node = HeaderBlock.from_bytes(WIRE_LITERAL_EMPTY_VALUE)[0]
        assert isinstance(node, LiteralHeaderField)
        assert node.name == b"foo"
        assert node.value == b""
        assert node.value_representation.payload == b""

    def test_huffman_strings(self):
        node = HeaderBlock.from_bytes(WIRE_HUFFMAN)[0]
        assert isinstance(node, LiteralHeaderField)
        name_repr = node.name_representation
        value_wire = node.value_representation
        assert isinstance(name_repr, LiteralName)
        assert name_repr.wire.huffman is True
        assert name_repr.wire.value == b"custom-key"
        assert value_wire.huffman is True
        assert value_wire.value == b"custom-value"

    def test_multibyte_name_index_four_bit_prefix(self):
        # Add one dynamic entry (index 62), then reference it from a
        # without-indexing literal whose 4-bit name index needs continuation
        # octets (0x0f 0x2f == 62).
        wire = b"\x40\x01a\x01b" + b"\x0f\x2f\x01z"
        block = HeaderBlock.from_bytes(wire)
        literal = block.literals()[1]
        name_repr = literal.name_representation
        assert isinstance(name_repr, IndexedName)
        assert name_repr.index == 62
        assert name_repr.raw == b"\x0f\x2f"
        assert literal.name == b"a"
        assert literal.value == b"z"
        assert block.to_bytes() == wire

    def test_multibyte_name_index_six_bit_prefix(self):
        # Add three dynamic entries (indices 62, 63, 64), then reference
        # index 64 via an incremental literal (0x7f 0x01).
        wire = b"\x40\x01a\x011" + b"\x40\x01a\x012" + b"\x40\x01a\x013" + b"\x7f\x01\x01z"
        block = HeaderBlock.from_bytes(wire)
        literal = block.literals()[3]
        name_repr = literal.name_representation
        assert isinstance(name_repr, IndexedName)
        assert name_repr.index == 64
        assert name_repr.raw == b"\x7f\x01"
        assert literal.value == b"z"
        assert block.to_bytes() == wire

    def test_byte_spans_partition_the_block(self):
        block = HeaderBlock.from_bytes(WIRE_MIXED)
        assert [type(node).__name__ for node in block] == [
            "DynamicTableSizeUpdate",
            "IndexedHeaderField",
            "LiteralHeaderField",
            "LiteralHeaderField",
        ]
        assert b"".join(node.raw for node in block) == WIRE_MIXED
        assert all(node.raw == WIRE_MIXED[node.start : node.end] for node in block)
        ordered = list(block)
        assert [node.end for node in ordered[:-1]] == [node.start for node in ordered[1:]]

    def test_accepts_bytearray_and_memoryview(self):
        assert HeaderBlock.from_bytes(bytearray(WIRE_INDEXED_GET)).to_bytes() == WIRE_INDEXED_GET
        assert HeaderBlock.from_bytes(memoryview(WIRE_INDEXED_GET)).to_bytes() == WIRE_INDEXED_GET

    def test_continues_external_header_table(self):
        table = HeaderTable()
        first = HeaderBlock.from_bytes(b"\x40\x01a\x01b", header_table=table)
        assert first.to_bytes() == b"\x40\x01a\x01b"

        # The dynamic entry added by the first block is visible to the second.
        second = HeaderBlock.from_bytes(b"\xbe", header_table=table)
        assert isinstance(second[0], IndexedHeaderField)
        assert second[0].index == 62

        # A fresh block cannot resolve that dynamic index.
        with pytest.raises(InvalidTableIndex):
            HeaderBlock.from_bytes(b"\xbe")

    def test_reprs(self):
        block = HeaderBlock.from_bytes(WIRE_MIXED)
        assert "HeaderBlock(" in repr(block)
        assert "DynamicTableSizeUpdate(new_size=0)" == repr(block.nodes[0])
        assert "IndexedHeaderField(index=2)" == repr(block.nodes[1])
        literal = block.nodes[2]
        assert "LiteralHeaderField(" in repr(literal)
        assert "IndexingMode.INCREMENTAL" in repr(literal)
        assert "IndexedName(index=4)" == repr(literal.name_representation)
        never = block.nodes[3]
        assert "LiteralName(" in repr(never.name_representation)
        assert "WireString(" in repr(never.value_representation)
        assert all(mode.value for mode in IndexingMode)


class TestLosslessRoundTrip:
    def test_unmodified_block_is_byte_for_byte(self):
        block = HeaderBlock.from_bytes(WIRE_MIXED)
        assert block.to_bytes() == WIRE_MIXED
        # A second serialization keeps returning the same wire bytes.
        assert block.to_bytes() == WIRE_MIXED

    @pytest.mark.parametrize(
        "story_path",
        STORY_FILES,
        ids=lambda p: str(Path(p).relative_to(FIXTURE_DIR)).replace("/", "-"),
    )
    def test_all_fixture_wire_is_restored_byte_for_byte(self, story_path):
        with Path(story_path).open(encoding="utf-8") as handle:
            story = json.load(handle)
        if story.get("draft") != 9:
            pytest.skip("fixtures target draft 9")

        block_table = HeaderTable()
        decoder = Decoder()

        for case in story["cases"]:
            if "header_table_size" in case:
                block_table.maxsize = case["header_table_size"]
                decoder.header_table_size = case["header_table_size"]

            wire = unhexlify(case["wire"])
            block = HeaderBlock.from_bytes(wire, header_table=block_table)

            # The central lossless guarantee: untouched input round-trips
            # byte-for-byte.
            assert block.to_bytes() == wire

            # And it decodes to exactly the headers the fixture expects.
            decoded = decoder.decode(wire)
            expected = [(name, value) for header in case["headers"] for name, value in header.items()]
            assert decoded == expected

    @given(
        headers=lists(tuples(binary(max_size=200), binary(max_size=200)), max_size=6),
        huffman=booleans(),
    )
    def test_encoder_output_round_trips_losslessly(self, headers, huffman):
        wire = Encoder().encode(headers, huffman=huffman)
        block = HeaderBlock.from_bytes(wire)

        assert block.to_bytes() == wire
        assert b"".join(node.raw for node in block) == wire
        assert Decoder().decode(wire, raw=True) == headers


class TestLiteralEditing:
    def test_edit_value_keeps_other_segments_identical(self):
        block = HeaderBlock.from_bytes(WIRE_MIXED)
        block.literals()[0].set_value(b"/abc")

        prefix = WIRE_TABLE_SIZE_0 + WIRE_INDEXED_GET
        suffix = WIRE_LITERAL_NEVER_INDEXED
        edited = block.to_bytes()
        # Only the incremental literal was rebuilt; the table size update,
        # indexed field and never-indexed literal stay byte-for-byte identical.
        assert edited.startswith(prefix)
        assert edited.endswith(suffix)
        assert edited[len(prefix) : len(edited) - len(suffix)] == b"\x44\x04/abc"

        decoded = Decoder().decode(edited)
        assert decoded == [
            (":method", "GET"),
            (":path", "/abc"),
            ("foo", "bar"),
        ]

    def test_edit_value_preserves_multibyte_name_index(self):
        wire = b"\x40\x01a\x01b" + b"\x0f\x2f\x01z"
        block = HeaderBlock.from_bytes(wire)
        block.literals()[1].set_value(b"zz")
        edited = block.to_bytes()
        # The two continuation octets naming the dynamic entry are preserved.
        assert b"\x0f\x2f" in edited
        assert edited == b"\x40\x01a\x01b" + b"\x0f\x2f\x02zz"
        assert Decoder().decode(edited, raw=True) == [(b"a", b"b"), (b"a", b"zz")]

    def test_edit_literal_name_and_value(self):
        block = HeaderBlock.from_bytes(WIRE_LITERAL_NO_INDEX)
        literal = block.literals()[0]
        literal.set_name(b"baz")
        literal.set_value(b"qux")

        assert literal.name == b"baz"
        assert literal.value == b"qux"
        assert literal.dirty is True
        assert literal.name_representation.dirty is True
        assert literal.value_representation.dirty is True
        assert block.to_bytes() == b"\x00\x03baz\x03qux"

    def test_edit_name_on_indexed_name_is_rejected(self):
        block = HeaderBlock.from_bytes(WIRE_INCREMENTAL_INDEXED_NAME)
        with pytest.raises(InvalidBlockEditError):
            block.literals()[0].set_name(b":method")

    def test_edit_value_with_non_bytes_is_rejected(self):
        block = HeaderBlock.from_bytes(WIRE_LITERAL_NO_INDEX)
        with pytest.raises(TypeError):
            block.literals()[0].set_value("not-bytes")  # type: ignore[arg-type]

    def test_edit_keeps_huffman_flag_by_default(self):
        block = HeaderBlock.from_bytes(WIRE_HUFFMAN)
        block.literals()[0].set_value(b"custom-valu2")
        edited = block.to_bytes()

        reparsed = HeaderBlock.from_bytes(edited).literals()[0]
        assert reparsed.value_representation.huffman is True
        assert reparsed.value_representation.value == b"custom-valu2"
        # The name segment was not touched at all.
        assert edited.startswith(b"\x40\x88\x25\xa8\x49\xe9\x5b\xa9\x7d\x7f")
        assert Decoder().decode(edited, raw=True) == [(b"custom-key", b"custom-valu2")]

    def test_edit_can_disable_and_force_huffman(self):
        huffman_block = HeaderBlock.from_bytes(WIRE_HUFFMAN)
        huffman_block.literals()[0].set_value(b"custom-value", huffman=False)
        plain = huffman_block.to_bytes()
        assert not (plain[-len(b"custom-value") - 1] & 0x80)
        assert Decoder().decode(plain, raw=True) == [(b"custom-key", b"custom-value")]

        plain_block = HeaderBlock.from_bytes(WIRE_LITERAL_NO_INDEX)
        plain_block.literals()[0].set_value(b"bar", huffman=True)
        encoded = plain_block.to_bytes()
        assert encoded != WIRE_LITERAL_NO_INDEX
        assert Decoder().decode(encoded, raw=True) == [(b"foo", b"bar")]

    def test_edit_name_with_huffman_override(self):
        block = HeaderBlock.from_bytes(WIRE_LITERAL_NO_INDEX)
        block.literals()[0].set_name(b"foo", huffman=True)
        edited = block.to_bytes()
        assert Decoder().decode(edited, raw=True) == [(b"foo", b"bar")]

    def test_multiple_edits_in_one_block(self):
        block = HeaderBlock.from_bytes(
            WIRE_LITERAL_NO_INDEX + b"\x00\x03key\x05value",
        )
        literals = block.literals()
        literals[0].set_value(b"BAR")
        literals[1].set_name(b"KEY")

        edited = block.to_bytes()
        assert Decoder().decode(edited, raw=True) == [
            (b"foo", b"BAR"),
            (b"KEY", b"value"),
        ]

    def test_incremental_edit_is_seen_by_later_indexed_reference(self):
        # Add e1=v1 dynamically, then reference the new entry via index 62.
        wire = b"\x40\x02e1\x02v1" + b"\xbe"
        block = HeaderBlock.from_bytes(wire)
        block.literals()[0].set_value(b"v2")
        edited = block.to_bytes()

        # The indexed representation itself is untouched on the wire; both
        # fields now decode to the edited dynamic entry.
        assert edited[-1] == 0xBE
        assert Decoder().decode(edited, raw=True) == [(b"e1", b"v2"), (b"e1", b"v2")]

    def test_incremental_literal_name_edit_updates_dynamic_entry(self):
        # Literal name+value with incremental indexing, then self-reference.
        wire = b"\x40\x03foo\x03bar" + b"\xbe"
        block = HeaderBlock.from_bytes(wire)
        literal = block.literals()[0]
        literal.set_name(b"baz")
        literal.set_value(b"qux")
        edited = block.to_bytes()

        # Only the literal node changed; the trailing index octet is intact.
        assert edited[-1] == 0xBE
        assert Decoder().decode(edited, raw=True) == [(b"baz", b"qux"), (b"baz", b"qux")]

    def test_editing_non_indexing_literal_never_touches_table(self):
        block = HeaderBlock.from_bytes(b"\x00\x03foo\x03bar")
        block.literals()[0].set_value(b"x" * 1000)
        edited = block.to_bytes()
        assert Decoder().decode(edited, raw=True) == [(b"foo", b"x" * 1000)]

        never = HeaderBlock.from_bytes(b"\x10\x03foo\x03bar")
        never.literals()[0].set_value(b"y" * 1000)
        assert Decoder().decode(never.to_bytes(), raw=True) == [(b"foo", b"y" * 1000)]


class TestEditValidation:
    def test_growing_entry_until_referenced_entry_evicted_is_rejected(self):
        # Size 128; e1 (36 bytes) then e2 (36 bytes) both fit; index 63
        # references e1. Growing e2 past the table evicts e1, so index 63
        # would no longer resolve.
        wire = (
            WIRE_TABLE_SIZE_128 + b"\x40\x02e1\x02v1" + b"\x40\x02e2\x02v2" + b"\xbf"  # indexed 63 == e1
        )
        block = HeaderBlock.from_bytes(wire)
        assert block.to_bytes() == wire

        block.literals()[1].set_value(b"v" * 100)
        with pytest.raises(InvalidBlockEditError) as exc_info:
            block.to_bytes()
        assert isinstance(exc_info.value.__cause__, InvalidTableIndex)

    def test_oversized_edited_list_is_rejected(self):
        block = HeaderBlock.from_bytes(b"\x10\x03foo\x03bar")
        block.literals()[0].set_value(b"x" * 70000)
        with pytest.raises(InvalidBlockEditError) as exc_info:
            block.to_bytes()
        assert isinstance(exc_info.value.__cause__, OversizedHeaderListError)

    def test_table_size_update_still_validates_after_edit(self):
        # Manually assemble an otherwise-unreachable tree where a size update
        # follows a header, and confirm validation rejects it on serialize.
        indexed = IndexedHeaderField(2, b"\x82", 0)
        update = DynamicTableSizeUpdate(0, b"\x20", 1)
        block = HeaderBlock(
            b"\x82\x20",
            (indexed, update),
            HeaderTable(),
            HeaderTable.DEFAULT_SIZE,
            2**16,
        )
        indexed._dirty = True
        with pytest.raises(InvalidBlockEditError) as exc_info:
            block.to_bytes()
        assert isinstance(exc_info.value.__cause__, HPACKDecodingError)

    def test_excessive_table_size_update_after_edit_is_rejected(self):
        # An update before any header, but above the allowed maximum.
        update = DynamicTableSizeUpdate(999, b"\x3f\x87\x07", 0)
        indexed = IndexedHeaderField(2, b"\x82", 3)
        block = HeaderBlock(
            b"\x3f\x87\x07\x82",
            (update, indexed),
            HeaderTable(),
            100,
            2**16,
        )
        indexed._dirty = True
        with pytest.raises(InvalidBlockEditError) as exc_info:
            block.to_bytes()
        assert isinstance(exc_info.value.__cause__, InvalidTableSizeError)


class TestParseErrors:
    def test_truncated_literal_name_length(self):
        with pytest.raises(HPACKDecodingError):
            HeaderBlock.from_bytes(b"\x82\x00")

    def test_truncated_literal_name_payload(self):
        with pytest.raises(HPACKDecodingError):
            HeaderBlock.from_bytes(b"\x00\x03ab")

    def test_truncated_literal_value_payload(self):
        with pytest.raises(HPACKDecodingError):
            HeaderBlock.from_bytes(b"\x00\x03foo\x05ab")

    def test_truncated_value_after_indexed_name(self):
        with pytest.raises(HPACKDecodingError):
            HeaderBlock.from_bytes(b"\x44\x04")

    def test_truncated_table_size_update(self):
        with pytest.raises(HPACKDecodingError):
            HeaderBlock.from_bytes(b"\x3f")

    def test_invalid_table_index(self):
        with pytest.raises(InvalidTableIndex):
            HeaderBlock.from_bytes(b"\xbe")

    def test_table_size_update_after_header_is_rejected(self):
        with pytest.raises(HPACKDecodingError):
            HeaderBlock.from_bytes(b"\x82\x20")

    def test_table_size_update_above_limit_is_rejected(self):
        # 5-bit-prefix integer 200: 0x3f 0xa9 0x01.
        with pytest.raises(InvalidTableSizeError):
            HeaderBlock.from_bytes(b"\x3f\xa9\x01", max_allowed_table_size=100)

    def test_oversized_header_list_is_rejected(self):
        with pytest.raises(OversizedHeaderListError):
            HeaderBlock.from_bytes(
                WIRE_LITERAL_NO_INDEX,
                max_header_list_size=10,
            )

    def test_invalid_huffman_payload_is_rejected(self):
        # H=1 name of length 1 with payload 0xff is not decodable Huffman data.
        with pytest.raises(HPACKDecodingError):
            HeaderBlock.from_bytes(b"\x00\x81\xff")


def test_representation_base_is_node_class():
    block = HeaderBlock.from_bytes(WIRE_INDEXED_GET)
    assert isinstance(block[0], Representation)
