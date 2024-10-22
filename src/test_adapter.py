

import unittest

from src.adapter_idefics import IdeficsAdapter


class TestSelectRegex(unittest.TestCase):

    def setUp(self):
        self.parse_fn = IdeficsAdapter.parse_raw

    def test_matches_select(self):
        self.assertEqual(self.parse_fn(
            'Assistant: select A'), (set(['A']), set()))
        self.assertEqual(self.parse_fn(
            'Assistant: select A B'), (set(['A', 'B']), set()))
        self.assertEqual(self.parse_fn(
            'Assistant: select A B C'), (set(['A', 'B', 'C']), set()))

    def test_matches_deselect(self):
        self.assertEqual(self.parse_fn(
            'Assistant: deselect D'), (set(), set(['D'])))
        self.assertEqual(self.parse_fn(
            'Assistant: deselect D E'), (set(), set(['D', 'E'])))
        self.assertEqual(self.parse_fn(
            'Assistant: deselect D E F'), (set(), set(['D', 'E', 'F'])))

    def test_matches_select_and_deselect(self):
        self.assertEqual(self.parse_fn(
            'Assistant: deselect A B C select H I J'),
            (set(['H', 'I', 'J']), set(['A', 'B', 'C'])))
        self.assertEqual(self.parse_fn(
            'Assistant: select A B deselect H I'),
            (set(), set()))

    def test_funky(self):
        self.assertEqual(self.parse_fn(
            'Assistant: select select J'), (set([]), set()))
        self.assertEqual(self.parse_fn(
            'Assistant: select select J deselect A select J'), (set([]), set()))
        self.assertEqual(self.parse_fn(
            'Assistant: deselect deselect J'), (set(), set()))
        self.assertEqual(self.parse_fn(
            'Assistant: select deselect J'), (set(), set()))
        self.assertEqual(self.parse_fn(
            'Assistant: deselect select J'), (set(), set()))
        self.assertEqual(self.parse_fn(
            'Assistant: select select select J'), (set(), set()))
        self.assertEqual(self.parse_fn(
            'Assistant: deselect deselect deselect J'), (set(), set()))
        self.assertEqual(self.parse_fn(
            'Assistant: select selectelect deselect J'), (set(), set()))

    def test_gen(self):
        self.assertEqual(self.parse_fn(
            ' deselect D E F'), (set(), set(['D', 'E', 'F'])))
        self.assertEqual(self.parse_fn(
            'select A B C'), (set(['A', 'B', 'C']), set()))
        self.assertEqual(self.parse_fn(
            ' select J deselect A select J'), (set([]), set()))


if __name__ == '__main__':
    unittest.main()
