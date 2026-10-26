hpack API
=========

This document provides the HPACK API.

.. autoclass:: hpack.Encoder
   :members: header_table_size, encode

.. autoclass:: hpack.Decoder
   :members: header_table_size, decode

.. autoclass:: hpack.HeaderBlock
   :members: from_bytes, to_bytes, nodes, literals

.. autoclass:: hpack.Representation
   :members: raw, start, end, dirty

.. autoclass:: hpack.IndexedHeaderField
   :members: index

.. autoclass:: hpack.LiteralHeaderField
   :members: mode, name, value, set_name, set_value

.. autoclass:: hpack.DynamicTableSizeUpdate
   :members: new_size

.. autoclass:: hpack.IndexingMode
   :members: INCREMENTAL, WITHOUT_INDEXING, NEVER_INDEXED

.. autoclass:: hpack.IndexedName
   :members: index, literal

.. autoclass:: hpack.LiteralName
   :members: literal, dirty

.. autoclass:: hpack.WireString
   :members: huffman, payload, value, dirty, set

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

.. autoclass:: hpack.InvalidBlockEditError
