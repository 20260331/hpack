# flake8: noqa
from __future__ import annotations

from collections import deque
import logging
from typing import Any, Optional

from .exceptions import InvalidContextError, InvalidTableIndex

log = logging.getLogger(__name__)

#: Largest integer carried by the portable context format. HPACK table sizes
#: are SETTINGS_HEADER_TABLE_SIZE values, which are uint32.
MAX_CONTEXT_INT = 2 ** 32 - 1

_CONTEXT_TABLE_KEYS = {"maxsize", "current_size", "resized", "entries"}


def strict_nonnegative_int(value: Any, field: str, *, maximum: int = MAX_CONTEXT_INT) -> int:
    """
    Validates that ``value`` is a plain nonnegative integer not exceeding
    ``maximum``. Booleans are rejected despite being ``int`` subclasses, since
    the context format does not treat them as integers.
    """
    if type(value) is not int:
        msg = f"{field} must be an integer, got {type(value).__name__}"
        raise InvalidContextError(msg)
    if not 0 <= value <= maximum:
        msg = f"{field} must be between 0 and {maximum}, got {value}"
        raise InvalidContextError(msg)
    return value


def table_entry_size(name: bytes, value: bytes) -> int:
    """
    Calculates the size of a single entry

    This size is mostly irrelevant to us and defined
    specifically to accommodate memory management for
    lower level implementations. The 32 extra bytes are
    considered the "maximum" overhead that would be
    required to represent each entry in the table.

    See RFC7541 Section 4.1
    """
    return 32 + len(name) + len(value)


class HeaderTable:
    """
    Implements the combined static and dynamic header table

    The name and value arguments for all the functions
    should ONLY be byte strings (b'') however this is not
    strictly enforced in the interface.

    See RFC7541 Section 2.3
    """
    #: Default maximum size of the dynamic table. See
    #:  RFC7540 Section 6.5.2.
    DEFAULT_SIZE = 4096

    #: Constant list of static headers. See RFC7541 Section
    #:  2.3.1 and Appendix A
    STATIC_TABLE = (
        (b':authority'                  , b''             ),  # noqa
        (b':method'                     , b'GET'          ),  # noqa
        (b':method'                     , b'POST'         ),  # noqa
        (b':path'                       , b'/'            ),  # noqa
        (b':path'                       , b'/index.html'  ),  # noqa
        (b':scheme'                     , b'http'         ),  # noqa
        (b':scheme'                     , b'https'        ),  # noqa
        (b':status'                     , b'200'          ),  # noqa
        (b':status'                     , b'204'          ),  # noqa
        (b':status'                     , b'206'          ),  # noqa
        (b':status'                     , b'304'          ),  # noqa
        (b':status'                     , b'400'          ),  # noqa
        (b':status'                     , b'404'          ),  # noqa
        (b':status'                     , b'500'          ),  # noqa
        (b'accept-charset'              , b''             ),  # noqa
        (b'accept-encoding'             , b'gzip, deflate'),  # noqa
        (b'accept-language'             , b''             ),  # noqa
        (b'accept-ranges'               , b''             ),  # noqa
        (b'accept'                      , b''             ),  # noqa
        (b'access-control-allow-origin' , b''             ),  # noqa
        (b'age'                         , b''             ),  # noqa
        (b'allow'                       , b''             ),  # noqa
        (b'authorization'               , b''             ),  # noqa
        (b'cache-control'               , b''             ),  # noqa
        (b'content-disposition'         , b''             ),  # noqa
        (b'content-encoding'            , b''             ),  # noqa
        (b'content-language'            , b''             ),  # noqa
        (b'content-length'              , b''             ),  # noqa
        (b'content-location'            , b''             ),  # noqa
        (b'content-range'               , b''             ),  # noqa
        (b'content-type'                , b''             ),  # noqa
        (b'cookie'                      , b''             ),  # noqa
        (b'date'                        , b''             ),  # noqa
        (b'etag'                        , b''             ),  # noqa
        (b'expect'                      , b''             ),  # noqa
        (b'expires'                     , b''             ),  # noqa
        (b'from'                        , b''             ),  # noqa
        (b'host'                        , b''             ),  # noqa
        (b'if-match'                    , b''             ),  # noqa
        (b'if-modified-since'           , b''             ),  # noqa
        (b'if-none-match'               , b''             ),  # noqa
        (b'if-range'                    , b''             ),  # noqa
        (b'if-unmodified-since'         , b''             ),  # noqa
        (b'last-modified'               , b''             ),  # noqa
        (b'link'                        , b''             ),  # noqa
        (b'location'                    , b''             ),  # noqa
        (b'max-forwards'                , b''             ),  # noqa
        (b'proxy-authenticate'          , b''             ),  # noqa
        (b'proxy-authorization'         , b''             ),  # noqa
        (b'range'                       , b''             ),  # noqa
        (b'referer'                     , b''             ),  # noqa
        (b'refresh'                     , b''             ),  # noqa
        (b'retry-after'                 , b''             ),  # noqa
        (b'server'                      , b''             ),  # noqa
        (b'set-cookie'                  , b''             ),  # noqa
        (b'strict-transport-security'   , b''             ),  # noqa
        (b'transfer-encoding'           , b''             ),  # noqa
        (b'user-agent'                  , b''             ),  # noqa
        (b'vary'                        , b''             ),  # noqa
        (b'via'                         , b''             ),  # noqa
        (b'www-authenticate'            , b''             ),  # noqa
    )  # noqa

    STATIC_TABLE_LENGTH = len(STATIC_TABLE)

    STATIC_TABLE_MAPPING: dict[bytes, tuple[int, dict[bytes, int]]]

    def __init__(self) -> None:
        self._maxsize = HeaderTable.DEFAULT_SIZE
        self._current_size = 0
        self.resized = False
        self.dynamic_entries: deque[tuple[bytes, bytes]] = deque()

    def get_by_index(self, index: int) -> tuple[bytes, bytes]:
        """
        Returns the entry specified by index

        Note that the table is 1-based ie an index of 0 is
        invalid.  This is due to the fact that a zero value
        index signals that a completely unindexed header
        follows.

        The entry will either be from the static table or
        the dynamic table depending on the value of index.
        """
        original_index = index
        index -= 1
        if 0 <= index:
            if index < HeaderTable.STATIC_TABLE_LENGTH:
                return HeaderTable.STATIC_TABLE[index]

            index -= HeaderTable.STATIC_TABLE_LENGTH
            if index < len(self.dynamic_entries):
                return self.dynamic_entries[index]

        raise InvalidTableIndex("Invalid table index %d" % original_index)

    def __repr__(self) -> str:
        return "HeaderTable(%d, %s, %r)" % (
            self._maxsize,
            self.resized,
            self.dynamic_entries
        )

    def add(self, name: bytes, value: bytes) -> None:
        """
        Adds a new entry to the table

        We reduce the table size if the entry will make the
        table size greater than maxsize.
        """
        # We just clear the table if the entry is too big
        size = table_entry_size(name, value)
        if size > self._maxsize:
            self.dynamic_entries.clear()
            self._current_size = 0
        else:
            # Add new entry
            self.dynamic_entries.appendleft((name, value))
            self._current_size += size
            self._shrink()

    def search(self, name: bytes, value: bytes) -> Optional[tuple[int, bytes, Optional[bytes]]]:
        """
        Searches the table for the entry specified by name
        and value

        Returns one of the following:
            - ``None``, no match at all
            - ``(index, name, None)`` for partial matches on name only.
            - ``(index, name, value)`` for perfect matches.
        """
        partial = None

        header_name_search_result = HeaderTable.STATIC_TABLE_MAPPING.get(name)
        if header_name_search_result:
            index = header_name_search_result[1].get(value)
            if index is not None:
                return index, name, value
            else:
                partial = (header_name_search_result[0], name, None)

        offset = HeaderTable.STATIC_TABLE_LENGTH + 1
        for (i, (n, v)) in enumerate(self.dynamic_entries):
            if n == name:
                if v == value:
                    return i + offset, n, v
                elif partial is None:
                    partial = (i + offset, n, None)
        return partial

    @property
    def maxsize(self) -> int:
        return self._maxsize

    @maxsize.setter
    def maxsize(self, newmax: int) -> None:
        newmax = int(newmax)
        log.debug("Resizing header table to %d from %d", newmax, self._maxsize)
        oldmax = self._maxsize
        self._maxsize = newmax
        self.resized = (newmax != oldmax)
        if newmax <= 0:
            self.dynamic_entries.clear()
            self._current_size = 0
        elif oldmax > newmax:
            self._shrink()

    def _shrink(self) -> None:
        """
        Shrinks the dynamic table to be at or below maxsize
        """
        cursize = self._current_size
        while cursize > self._maxsize:
            name, value = self.dynamic_entries.pop()
            cursize -= table_entry_size(name, value)
            log.debug("Evicting %s: %s from the header table", name, value)
        self._current_size = cursize

    def export_state(self) -> dict[str, Any]:
        """
        Snapshots the dynamic table as plain, portable data.

        Entries are emitted newest-first, exactly in dynamic table index
        order, as ``[name, value]`` byte-string pairs.
        """
        return {
            "maxsize": self._maxsize,
            "current_size": self._current_size,
            "resized": self.resized,
            "entries": [[name, value] for name, value in self.dynamic_entries],
        }

    @classmethod
    def restore_state(cls, state: Any) -> HeaderTable:
        """
        Validates a dynamic-table state document and builds a brand new
        :class:`HeaderTable` from it. No existing table is mutated, so callers
        can atomically swap the returned table into an Encoder or Decoder
        only after validation has fully succeeded.
        """
        if not isinstance(state, dict):
            msg = "header_table context must be an object"
            raise InvalidContextError(msg)
        if set(state.keys()) != _CONTEXT_TABLE_KEYS:
            msg = "header_table context has missing or unexpected fields"
            raise InvalidContextError(msg)

        maxsize = strict_nonnegative_int(state["maxsize"], "header_table.maxsize")
        current_size = strict_nonnegative_int(
            state["current_size"], "header_table.current_size",
        )

        resized = state["resized"]
        if type(resized) is not bool:
            msg = "header_table.resized must be a boolean"
            raise InvalidContextError(msg)

        raw_entries = state["entries"]
        if not isinstance(raw_entries, (list, tuple)):
            msg = "header_table.entries must be a list"
            raise InvalidContextError(msg)

        entries: deque[tuple[bytes, bytes]] = deque()
        computed_size = 0
        for position, raw_entry in enumerate(raw_entries):
            name, value, entry_size = _validate_table_entry(raw_entry, position, maxsize)
            computed_size += entry_size
            entries.append((name, value))

        if computed_size != current_size:
            msg = "header_table.current_size does not match the size of the entries"
            raise InvalidContextError(msg)
        if computed_size > maxsize:
            msg = "header table entries exceed header_table.maxsize"
            raise InvalidContextError(msg)

        table = cls()
        table._maxsize = maxsize
        table._current_size = computed_size
        table.resized = resized
        table.dynamic_entries = entries
        return table


def _validate_table_entry(
    raw_entry: Any, position: int, maxsize: int,
) -> tuple[bytes, bytes, int]:
    """
    Validates one ``[name, value]`` dynamic-table entry and returns the
    byte strings together with their RFC 7541 entry size.
    """
    if not isinstance(raw_entry, (list, tuple)) or len(raw_entry) != 2:
        msg = f"header_table.entries[{position}] must be a (name, value) pair"
        raise InvalidContextError(msg)
    name, value = raw_entry
    if not isinstance(name, bytes) or not isinstance(value, bytes):
        msg = f"header_table.entries[{position}] name and value must be bytes"
        raise InvalidContextError(msg)
    entry_size = table_entry_size(name, value)
    if entry_size > maxsize:
        msg = f"header_table.entries[{position}] does not fit in header_table.maxsize"
        raise InvalidContextError(msg)
    return name, value, entry_size


def _build_static_table_mapping() -> dict[bytes, tuple[int, dict[bytes, int]]]:
    """
    Build static table mapping from header name to tuple with next structure:
    (<minimal index of header>, <mapping from header value to it index>).

    static_table_mapping used for hash searching.
    """
    static_table_mapping: dict[bytes, tuple[int, dict[bytes, int]]] = {}
    for index, (name, value) in enumerate(HeaderTable.STATIC_TABLE, 1):
        header_name_search_result = static_table_mapping.setdefault(name, (index, {}))
        header_name_search_result[1][value] = index
    return static_table_mapping


HeaderTable.STATIC_TABLE_MAPPING = _build_static_table_mapping()
