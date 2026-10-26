hpack API
=========

This document provides the HPACK API.

.. autoclass:: hpack.Encoder
   :members: header_table_size, encode, export_context, import_context

.. autoclass:: hpack.Decoder
   :members: header_table_size, decode, export_context, import_context

.. autodata:: hpack.CONTEXT_VERSION

.. autoclass:: hpack.HeaderTuple
   :members: indexable

.. autoclass:: hpack.NeverIndexedHeaderTuple
   :members: indexable

.. autoclass:: hpack.HPACKError

.. autoclass:: hpack.HPACKDecodingError

.. autoclass:: hpack.InvalidTableIndexError

.. autoclass:: hpack.InvalidTableIndex

.. autoclass:: hpack.OversizedHeaderListError

.. autoclass:: hpack.InvalidTableSizeError

.. autoclass:: hpack.InvalidContextError
