"""
Tests for the dual-context HPACK block transcoder.

The tests model a four-party hop:

    downstream encoder -> [Transcoder: input Decoder / output Encoder] -> upstream decoder

The downstream encoder and the ``Transcoder``'s input decoder share one HPACK
context (the input hop); the transcoder's output encoder and the upstream
decoder share a second, independent one (the output hop).
"""
import pytest

from hpack import (
    Decoder,
    Encoder,
    HPACKDecodingError,
    HeaderTuple,
    NeverIndexedHeaderTuple,
    TranscodeAbortedError,
    Transcoder,
)


class TwoHopConnection:
    """A simulated proxy hop with two independent HPACK contexts."""

    def __init__(self, transcoder: Transcoder | None = None) -> None:
        self.downstream = Encoder()
        self.transcoder = transcoder or Transcoder()
        self.upstream = Decoder()

    def send(
        self,
        fields: list,
        callback=None,
        huffman: bool = True,
    ) -> list[HeaderTuple]:
        wire_in = self.downstream.encode(fields, huffman=huffman)
        wire_out = self.transcoder.transcode(wire_in, callback, huffman=huffman)
        return list(self.upstream.decode(wire_out, raw=True))

    def feed_raw(self, data: bytes, callback=None) -> bytes:
        return self.transcoder.transcode(data, callback)


class TestTranscoderConstruction:
    def test_default_contexts_are_independent(self):
        tc = Transcoder()
        assert isinstance(tc.decoder, Decoder)
        assert isinstance(tc.encoder, Encoder)
        assert tc.decoder is not tc.encoder
        assert tc.decoder.header_table is not tc.encoder.header_table

    def test_existing_contexts_are_accepted(self):
        decoder = Decoder()
        encoder = Encoder()
        tc = Transcoder(decoder=decoder, encoder=encoder)
        assert tc.decoder is decoder
        assert tc.encoder is encoder

    def test_shared_context_is_rejected(self):
        shared = Decoder()
        with pytest.raises(ValueError):
            Transcoder(decoder=shared, encoder=shared)  # type: ignore[arg-type]


class TestTranscoderHappyPath:
    def test_passthrough_block(self):
        conn = TwoHopConnection()
        received = conn.send([
            HeaderTuple(b":method", b"GET"),
            HeaderTuple(b"x-custom", b"value"),
        ])
        assert received == [
            HeaderTuple(b":method", b"GET"),
            HeaderTuple(b"x-custom", b"value"),
        ]

    def test_callback_receives_raw_byte_fields(self):
        seen = []

        def observe(field):
            seen.append(field)
            return field

        conn = TwoHopConnection()
        conn.send([HeaderTuple("x-custom", "value")], callback=observe, huffman=True)
        assert seen == [HeaderTuple(b"x-custom", b"value")]

    def test_repeated_blocks_keep_both_contexts_in_sync(self):
        conn = TwoHopConnection()
        for i in range(5):
            received = conn.send([
                HeaderTuple(b":method", b"GET"),
                HeaderTuple(b"x-repeat", f"value-{i}".encode()),
            ])
            assert received == [
                HeaderTuple(b":method", b"GET"),
                HeaderTuple(b"x-repeat", f"value-{i}".encode()),
            ]

        # Both dynamic tables have advanced, independently of each other.
        input_names = [n for n, _ in conn.transcoder.decoder.header_table.dynamic_entries]
        output_names = [n for n, _ in conn.transcoder.encoder.header_table.dynamic_entries]
        assert b"x-repeat" in input_names
        assert b"x-repeat" in output_names

    def test_dropped_field_is_indexed_on_input_but_not_on_output(self):
        def drop_secret(field):
            if field[0] == b"x-secret":
                return None
            return field

        conn = TwoHopConnection()
        # The first block introduces the secret on the input hop; later blocks
        # reference it purely through the input-side dynamic table.
        assert conn.send([
            HeaderTuple(b":method", b"GET"),
            HeaderTuple(b"x-secret", b"tok"),
            HeaderTuple(b"x-public", b"ok"),
        ], callback=drop_secret) == [
            HeaderTuple(b":method", b"GET"),
            HeaderTuple(b"x-public", b"ok"),
        ]
        received = conn.send([
            HeaderTuple(b"x-secret", b"tok"),
            HeaderTuple(b"x-public", b"ok"),
        ], callback=drop_secret)
        assert received == [HeaderTuple(b"x-public", b"ok")]

        input_entries = list(conn.transcoder.decoder.header_table.dynamic_entries)
        output_entries = list(conn.transcoder.encoder.header_table.dynamic_entries)
        assert b"x-secret" in [n for n, _ in input_entries]
        assert b"x-secret" not in [n for n, _ in output_entries]

    def test_dropping_every_field_emits_empty_block_and_keeps_input_in_sync(self):
        conn = TwoHopConnection()
        wire_out = conn.transcoder.transcode(
            conn.downstream.encode([HeaderTuple(b"x-secret", b"tok")]),
            callback=lambda field: None,
        )
        assert wire_out == b""
        assert list(conn.upstream.decode(wire_out, raw=True)) == []

        # The input table kept the dropped entry, so the downstream's next
        # indexed reference still resolves.
        received = conn.send([HeaderTuple(b"x-secret", b"tok")])
        assert received == [HeaderTuple(b"x-secret", b"tok")]

    def test_callback_rewrites_value(self):
        def rewrite(field):
            if field[0] == b"x-user":
                return HeaderTuple(field[0], b"anonymous")
            return field

        conn = TwoHopConnection()
        received = conn.send([
            HeaderTuple(b"x-user", b"alice"),
            HeaderTuple(b"x-keep", b"1"),
        ], callback=rewrite)
        assert received == [
            HeaderTuple(b"x-user", b"anonymous"),
            HeaderTuple(b"x-keep", b"1"),
        ]

    def test_never_indexed_field_is_preserved(self):
        conn = TwoHopConnection()
        received = conn.send([
            HeaderTuple(b":method", b"GET"),
            NeverIndexedHeaderTuple(b"cookie", b"session=abc"),
        ])
        assert received == [
            HeaderTuple(b":method", b"GET"),
            NeverIndexedHeaderTuple(b"cookie", b"session=abc"),
        ]
        output_entries = list(conn.transcoder.encoder.header_table.dynamic_entries)
        input_entries = list(conn.transcoder.decoder.header_table.dynamic_entries)
        assert b"cookie" not in [n for n, _ in output_entries]
        assert b"cookie" not in [n for n, _ in input_entries]

    def test_three_tuple_sensitive_callback_result(self):
        def make_sensitive(field):
            if field[0] == b"x-auth":
                return (field[0], field[1], True)
            return field

        conn = TwoHopConnection()
        received = conn.send([HeaderTuple(b"x-auth", b"bearer")], callback=make_sensitive)
        assert received == [NeverIndexedHeaderTuple(b"x-auth", b"bearer")]
        assert b"x-auth" not in [
            n for n, _ in conn.transcoder.encoder.header_table.dynamic_entries
        ]


class TestTranscoderFailureBoundaries:
    def test_callback_abort_input_committed_output_rolled_back(self):
        conn = TwoHopConnection()
        # Prefill both contexts with history.
        conn.send([
            HeaderTuple(b"x-out-only", b"1"),
            HeaderTuple(b"x-shared", b"a"),
        ])

        output_before = list(conn.transcoder.encoder.header_table.dynamic_entries)

        def abort(field):
            if field[0] == b"x-block":
                raise TranscodeAbortedError("forbidden field")
            return field

        with pytest.raises(TranscodeAbortedError):
            conn.send([
                HeaderTuple(b"x-shared", b"a"),
                HeaderTuple(b"x-block", b"1"),
            ], callback=abort)

        # Input side committed: the complete, valid block advanced the input
        # table, so it keeps every entry the downstream encoder indexed.
        input_entries = list(conn.transcoder.decoder.header_table.dynamic_entries)
        assert b"x-block" in [n for n, _ in input_entries]

        # Output side rolled back: nothing was indexed for the aborted block,
        # pre-existing output history is preserved.
        assert list(conn.transcoder.encoder.header_table.dynamic_entries) == output_before
        assert b"x-block" not in [
            n for n, _ in conn.transcoder.encoder.header_table.dynamic_entries
        ]
        assert (b"x-out-only", b"1") in conn.transcoder.encoder.header_table.dynamic_entries

        # Both contexts remain usable: the next input block may reference the
        # just-committed input entry, and the output encoder still resolves its
        # pre-existing history against the upstream decoder.
        received = conn.send([
            HeaderTuple(b"x-block", b"1"),
            HeaderTuple(b"x-out-only", b"1"),
        ])
        assert HeaderTuple(b"x-block", b"1") in received
        assert HeaderTuple(b"x-out-only", b"1") in received

    def test_arbitrary_callback_exception_uses_same_boundary(self):
        conn = TwoHopConnection()
        conn.send([HeaderTuple(b"x-old", b"1")])
        output_before = list(conn.transcoder.encoder.header_table.dynamic_entries)

        def boom(field):
            raise RuntimeError("boom")

        with pytest.raises(RuntimeError):
            conn.send([HeaderTuple(b"x-new", b"1")], callback=boom)

        assert list(conn.transcoder.encoder.header_table.dynamic_entries) == output_before
        assert b"x-new" in [
            n for n, _ in conn.transcoder.decoder.header_table.dynamic_entries
        ]

    def test_malformed_input_restores_both_sides(self):
        conn = TwoHopConnection()
        conn.send([HeaderTuple(b"x-old", b"1")])

        input_before = list(conn.transcoder.decoder.header_table.dynamic_entries)
        output_before = list(conn.transcoder.encoder.header_table.dynamic_entries)

        # A valid incremental-literal field followed by an indexed
        # representation with a truncated, impossible integer.
        bad_block = Encoder().encode([(b"x-new", b"1")], huffman=False) + b"\xff"

        with pytest.raises(HPACKDecodingError):
            conn.feed_raw(bad_block)

        assert list(conn.transcoder.decoder.header_table.dynamic_entries) == input_before
        assert list(conn.transcoder.encoder.header_table.dynamic_entries) == output_before
        assert b"x-new" not in [
            n for n, _ in conn.transcoder.decoder.header_table.dynamic_entries
        ]

        # The transcoder is still usable at its clean local boundary.
        received = conn.send([HeaderTuple(b"x-after", b"1")])
        assert received == [HeaderTuple(b"x-after", b"1")]

    def test_encode_failure_rolls_back_output_keeps_input(self):
        conn = TwoHopConnection()
        conn.send([HeaderTuple(b"x-old", b"1")])
        output_before = list(conn.transcoder.encoder.header_table.dynamic_entries)

        def invalid_result(field):
            # The first field is valid and gets partially encoded before the
            # second result blows up inside the output encoder.
            if field[0] == b"x-second":
                return 123  # type: ignore[return-value]
            return field

        with pytest.raises(TypeError):
            conn.send([
                HeaderTuple(b"x-first", b"1"),
                HeaderTuple(b"x-second", b"1"),
            ], callback=invalid_result)

        output_entries = list(conn.transcoder.encoder.header_table.dynamic_entries)
        assert output_entries == output_before
        assert b"x-first" not in [n for n, _ in output_entries]

        # Input side stays committed to the fully decoded block.
        assert b"x-second" in [
            n for n, _ in conn.transcoder.decoder.header_table.dynamic_entries
        ]

        received = conn.send([HeaderTuple(b"x-recovery", b"1")])
        assert received == [HeaderTuple(b"x-recovery", b"1")]

    def test_pending_output_table_size_change_is_restored(self):
        conn = TwoHopConnection()
        conn.send([HeaderTuple(b"x-old", b"1")])
        conn.transcoder.encoder.header_table_size = 512

        def abort(field):
            raise TranscodeAbortedError

        with pytest.raises(TranscodeAbortedError):
            conn.send([HeaderTuple(b"x-new", b"1")], callback=abort)

        # The pending size change must neither be lost nor emitted inside the
        # abandoned block.
        assert conn.transcoder.encoder.header_table_size == 512
        assert conn.transcoder.encoder.table_size_changes == [512]
        assert conn.transcoder.encoder.header_table.resized is True

        # The next successful block emits the deferred size update exactly once.
        wire_out = conn.transcoder.transcode(
            conn.downstream.encode([HeaderTuple(b"x-new", b"1")]),
        )
        assert wire_out[0] & 0xE0 == 0x20
        assert conn.transcoder.encoder.table_size_changes == []
        assert list(conn.upstream.decode(wire_out, raw=True)) == [
            HeaderTuple(b"x-new", b"1"),
        ]


class TestTranscoderTableSizes:
    def test_inbound_size_update_is_not_propagated(self):
        conn = TwoHopConnection()
        conn.downstream.header_table_size = 256

        wire_out = conn.transcoder.transcode(
            conn.downstream.encode([HeaderTuple(b"x-small", b"v")]),
        )

        assert conn.transcoder.decoder.header_table_size == 256
        assert conn.transcoder.encoder.header_table_size == 4096
        # The output block starts with a header, not a table size update.
        assert wire_out[0] & 0xE0 != 0x20
        assert list(conn.upstream.decode(wire_out, raw=True)) == [
            HeaderTuple(b"x-small", b"v"),
        ]

    def test_output_context_with_smaller_table(self):
        encoder = Encoder()
        encoder.header_table_size = 256
        upstream = Decoder()
        upstream.max_allowed_table_size = 256
        conn = TwoHopConnection(Transcoder(Decoder(), encoder))

        wire_out = conn.transcoder.transcode(
            conn.downstream.encode([HeaderTuple(b"x-a", b"v")]),
        )
        assert wire_out[0] & 0xE0 == 0x20
        assert list(upstream.decode(wire_out, raw=True)) == [HeaderTuple(b"x-a", b"v")]

        wire_out = conn.transcoder.transcode(
            conn.downstream.encode([HeaderTuple(b"x-b", f"{'v' * 200}".encode())]),
        )
        assert list(upstream.decode(wire_out, raw=True)) == [
            HeaderTuple(b"x-b", b"v" * 200),
        ]
        assert conn.transcoder.encoder.header_table_size == 256


class TestHeaderTableSnapshot:
    def test_snapshot_and_restore_round_trip(self):
        table = Encoder().header_table
        table.add(b"a", b"1")
        table.add(b"b", b"2")
        snapshot = table.snapshot()

        table.add(b"c", b"3")
        table.maxsize = 64  # triggers eviction and sets the resized flag
        assert (b"a", b"1") not in table.dynamic_entries
        assert table.resized is True

        table.restore(snapshot)
        assert list(table.dynamic_entries) == [(b"b", b"2"), (b"a", b"1")]
        assert table.maxsize == 4096
        assert table.resized is False

    def test_snapshot_is_independent_of_later_mutation(self):
        table = Encoder().header_table
        table.add(b"a", b"1")
        snapshot = table.snapshot()

        table.add(b"b", b"2")
        table.restore(snapshot)
        assert list(table.dynamic_entries) == [(b"a", b"1")]
