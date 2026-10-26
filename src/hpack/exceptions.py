"""
Exceptions used in hpack.
"""
from __future__ import annotations


class HPACKError(Exception):
    """
    The base class for all ``hpack`` exceptions.
    """



class HPACKDecodingError(HPACKError):
    """
    An error has been encountered while performing HPACK decoding.
    """



class InvalidTableIndexError(HPACKDecodingError):
    """
    An invalid table index was received.

    .. versionadded:: 4.1.0
    """

class InvalidTableIndex(InvalidTableIndexError):  # noqa: N818
    """
    An invalid table index was received.

    .. deprecated:: 4.1.0
       Renamed to :class:`InvalidTableIndexError`, use it instead.
    """


class OversizedHeaderListError(HPACKDecodingError):
    """
    A header list that was larger than we allow has been received. This may be
    a DoS attack.

    .. versionadded:: 2.3.0
    """


class InvalidTableSizeError(HPACKDecodingError):
    """
    An attempt was made to change the decoder table size to a value larger than
    allowed, or the list was shrunk and the remote peer didn't shrink their
    table size.

    .. versionadded:: 3.0.0
    """


class InvalidContextError(HPACKError):
    """
    A compression context passed to :meth:`Encoder.import_context
    <hpack.Encoder.import_context>` or :meth:`Decoder.import_context
    <hpack.Decoder.import_context>` was malformed, used an unsupported version,
    or failed consistency validation.

    The target object is left completely unchanged when this error is raised:
    validation always happens before any state is committed.

    .. versionadded:: 4.3.0
    """
