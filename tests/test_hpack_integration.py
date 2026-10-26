"""
This module defines substantial HPACK integration tests. These can take a
very long time to run, so they're outside the main test suite, but they need
to be run before every change to HPACK.
"""
from binascii import unhexlify
from pytest import skip

from hpack import Decoder, Encoder, HeaderTuple


class TestHPACKDecoderIntegration:
    def test_can_decode_a_story(self, story):
        d = Decoder()

        # We test against draft 9 of the HPACK spec.
        if story['draft'] != 9:
            skip("We test against draft 9, not draft %d" % story['draft'])

        for case in story['cases']:
            try:
                d.header_table_size = case['header_table_size']
            except KeyError:
                pass
            decoded_headers = d.decode(unhexlify(case['wire']))

            # The correct headers are a list of dicts, which is annoying.
            correct_headers = [
                (item[0], item[1])
                for header in case['headers']
                for item in header.items()
            ]
            correct_headers = correct_headers
            assert correct_headers == decoded_headers
            assert all(
                isinstance(header, HeaderTuple) for header in decoded_headers
            )

    def test_can_decode_a_fragmented_story(self, story):
        # The same stories, but every header block is delivered split into
        # fragments through an incremental decode session. The fragment size
        # rotates per case so the corpus as a whole exercises byte-by-byte
        # delivery as well as coarser chunks.
        fragment_sizes = (1, 2, 3, 5, 8)
        d = Decoder()

        # We test against draft 9 of the HPACK spec.
        if story['draft'] != 9:
            skip("We test against draft 9, not draft %d" % story['draft'])

        for case_index, case in enumerate(story['cases']):
            try:
                d.header_table_size = case['header_table_size']
            except KeyError:
                pass
            wire = unhexlify(case['wire'])
            fragment_size = fragment_sizes[case_index % len(fragment_sizes)]

            session = d.begin_decode()
            for offset in range(0, len(wire), fragment_size):
                session.feed(wire[offset:offset + fragment_size])
            decoded_headers = session.finish()

            # The correct headers are a list of dicts, which is annoying.
            correct_headers = [
                (item[0], item[1])
                for header in case['headers']
                for item in header.items()
            ]
            assert correct_headers == decoded_headers
            assert all(
                isinstance(header, HeaderTuple) for header in decoded_headers
            )

    def test_can_encode_a_story_no_huffman(self, raw_story):
        d = Decoder()
        e = Encoder()

        for case in raw_story['cases']:
            # The input headers are a list of dicts, which is annoying.
            input_headers = [
                (item[0], item[1])
                for header in case['headers']
                for item in header.items()
            ]

            encoded = e.encode(input_headers, huffman=False)
            decoded_headers = d.decode(encoded)

            assert input_headers == decoded_headers
            assert all(
                isinstance(header, HeaderTuple) for header in decoded_headers
            )

    def test_can_encode_a_story_with_huffman(self, raw_story):
        d = Decoder()
        e = Encoder()

        for case in raw_story['cases']:
            # The input headers are a list of dicts, which is annoying.
            input_headers = [
                (item[0], item[1])
                for header in case['headers']
                for item in header.items()
            ]

            encoded = e.encode(input_headers, huffman=True)
            decoded_headers = d.decode(encoded)

            assert input_headers == decoded_headers
