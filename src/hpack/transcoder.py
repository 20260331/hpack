"""
Block-at-a-time HPACK transcoding for intermediaries that rewrite fields.

An HPACK intermediary (for example a forwarding proxy) terminates compression
on the input side and originates it on the output side. The two sides are
**different compression contexts with different peers**:

* the input side is a :class:`~hpack.hpack.Decoder` whose dynamic table must
  mirror the table maintained by the peer sending blocks *to* the
  intermediary;
* the output side is an :class:`~hpack.hpack.Encoder` whose dynamic table must
  mirror the table of the peer the intermediary sends blocks *to*.

A naive ``decoder.decode(block) -> filter -> encoder.encode(headers)``
pipeline mutates both tables while it runs. Both :meth:`Decoder.decode` and
:meth:`Encoder.encode` apply dynamic-table updates as they go, so any failure
in the middle of a block leaves one table advanced although no matching block
was ever exchanged with the corresponding peer:

* a malformed or truncated input block leaves the input table half updated;
* a filter that rejects a field abandons the block after the input table has
  already advanced, with no guarantee about the output table state;
* an encoding failure leaves the output table advanced without a block ever
  being sent;
* converting the decoded fields to plain tuples or dictionaries loses the
  :class:`~hpack.struct.NeverIndexedHeaderTuple` marking, so sensitive values
  silently become indexable on the output side.

:class:`Transcoder` processes one *complete* block per call and makes the call
the state boundary for both tables.
"""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from .hpack import Decoder, Encoder

if TYPE_CHECKING:  # pragma: no cover
    from collections.abc import Callable
    from typing import TypeAlias

    from .struct import HeaderTuple
    from .table import HeaderTableState

log = logging.getLogger(__name__)

#: A field a rewrite callback allows through to the output block. It accepts
#: every field representation understood by :meth:`Encoder.encode`:
#: a :class:`~hpack.struct.HeaderTuple` (including
#: :class:`~hpack.struct.NeverIndexedHeaderTuple`), a ``(name, value)``
#: two-tuple, or a ``(name, value, sensitive)`` three-tuple.
TranscodeOutputField: TypeAlias = (
    "HeaderTuple"
    " | tuple[bytes | str, bytes | str]"
    " | tuple[bytes | str, bytes | str, bool | None]"
)

#: The result of a rewrite callback: a field to emit (possibly rewritten), or
#: ``None`` to drop the field from the output block.
TranscodedField: TypeAlias = "TranscodeOutputField | None"

#: A callback invoked once per decoded field, in decoding order. The field is
#: always a raw :class:`~hpack.struct.HeaderTuple` of byte strings, so its
#: :attr:`~hpack.struct.HeaderTuple.indexable` marking is preserved.
FieldRewriteCallback: TypeAlias = "Callable[[HeaderTuple], TranscodedField]"


def _passthrough(field: HeaderTuple) -> HeaderTuple:
    """Default rewrite callback: forwards every field unchanged."""
    return field


class Transcoder:
    """
    Rewrites complete HPACK header blocks across two independent HPACK
    contexts.

    The two contexts are independent by construction: the transcoder owns (or
    is given) a separate :class:`~hpack.hpack.Decoder` and
    :class:`~hpack.hpack.Encoder`, each with its own dynamic table. Dynamic
    table size updates received in an input block are applied to the input
    table only; they are never propagated to the output context, whose table
    size is governed separately by its own peer's ``SETTINGS_HEADER_TABLE_SIZE``.

    Fields are handed to the rewrite callback *raw* (as byte strings) and, by
    default, the :class:`~hpack.struct.NeverIndexedHeaderTuple` marking of an
    inbound sensitive field is carried onto the output block unchanged.

    **State boundary.** A single :meth:`transcode` call is atomic with respect
    to both dynamic tables:

    * Failure while **decoding** the input block: the input table is restored
      to the state it had before the call and the output table is untouched.
      The exception is re-raised. Note that per RFC 7541 a malformed block is
      a connection error on the input hop regardless; the rollback leaves the
      local context at a clean, well-defined boundary and keeps the output
      context usable on the other hop.
    * Failure while running the **callback** or while **encoding**: the
      complete input block was valid and has therefore already advanced the
      sending peer's encoder, so the input table stays *committed*. The output
      table (including any pending table size change) is rolled back to the
      state it had before the call and the exception is re-raised. No bytes are
      returned, so nothing needs to be sent on the output hop.
    * **Success**: both tables are committed and the encoded output block is
      returned. The output side must be considered committed by the caller at
      that point, as the returned block already references the committed output
      table state.
    """

    def __init__(
        self,
        decoder: Decoder | None = None,
        encoder: Encoder | None = None,
    ) -> None:
        """
        :param decoder: Input-side context. A fresh :class:`Decoder` is created
                        when omitted.
        :param encoder: Output-side context. A fresh :class:`Encoder` is
                        created when omitted.
        :raises ValueError: If both sides share the same dynamic header table,
                            for example when the same context object is given
                            twice.
        """
        decoder = decoder if decoder is not None else Decoder()
        encoder = encoder if encoder is not None else Encoder()
        if decoder.header_table is encoder.header_table:
            msg = "Transcoder requires independent Decoder and Encoder contexts"
            raise ValueError(msg)
        self._decoder = decoder
        self._encoder = encoder

    @property
    def decoder(self) -> Decoder:
        """The input-side decoding context, owned by this transcoder."""
        return self._decoder

    @property
    def encoder(self) -> Encoder:
        """The output-side encoding context, owned by this transcoder."""
        return self._encoder

    def transcode(
        self,
        data: bytes,
        callback: FieldRewriteCallback | None = None,
        huffman: bool = True,
    ) -> bytes:
        """
        Decodes one complete HPACK block, rewrites its fields and encodes a
        new block for the output context.

        :param data: A complete HPACK-encoded header block received on the
                     input side.
        :param callback: A callable invoked for every decoded field, in order.
                         It returns the field to emit (optionally rewritten or
                         re-typed), or ``None`` to drop the field. Raising
                         from the callback aborts the block; see the class
                         documentation for the resulting state boundary. When
                         omitted every field is forwarded unchanged, retaining
                         its never-indexed marking.
        :param huffman: Whether literal values on the output block are
                        Huffman-encoded, as in :meth:`Encoder.encode`.
        :returns: The HPACK-encoded output block.
        :raises Exception: Any exception raised while decoding, by the
                           callback, or while encoding is re-raised after the
                           documented rollback has been performed.
        """
        rewrite = _passthrough if callback is None else callback

        input_table = self._decoder.header_table
        output_table = self._encoder.header_table

        # Stage both contexts before anything can be mutated.
        input_state = input_table.snapshot()
        output_state = output_table.snapshot()
        output_size_changes = list(self._encoder.table_size_changes)

        try:
            fields = self._decoder.decode(data, raw=True)
        except Exception:
            # Malformed block: neither side may advance.
            log.debug("Transcode failed during decode; restoring both sides")
            input_table.restore(input_state)
            self._restore_output(output_state, output_size_changes)
            raise

        # The complete block decoded: this is the input-side commit point. The
        # sending peer's encoder has advanced, so the input table must stay in
        # step with it whatever happens next.

        output_fields: list[TranscodeOutputField] = []
        try:
            for field in fields:
                rewritten = rewrite(field)
                if rewritten is not None:
                    output_fields.append(rewritten)
        except Exception:
            log.debug("Transcode aborted by callback; rolling back output side")
            self._restore_output(output_state, output_size_changes)
            raise

        try:
            return self._encoder.encode(output_fields, huffman=huffman)
        except Exception:
            log.debug("Transcode failed during encode; rolling back output side")
            self._restore_output(output_state, output_size_changes)
            raise

    def _restore_output(
        self,
        output_state: HeaderTableState,
        output_size_changes: list[int],
    ) -> None:
        """Rolls the output context back to a staged snapshot."""
        self._encoder.header_table.restore(output_state)
        self._encoder.table_size_changes = output_size_changes
