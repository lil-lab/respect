import unittest

from src.multitask import reweigh_dict


class TestSelectRegex(unittest.TestCase):

    def setUp(self):
        pass

    def test_simple(self):
        self.assertDictEqual(
            reweigh_dict({"base": 565, "igl": 3000},
                         {"base": 0.5, "igl": 0.5}),
            {'base': 3000, 'igl': 3000}
        )

    def test_ignore_one(self):
        self.assertDictEqual(
            reweigh_dict({"ignore": 2, "base": 565, "igl": 3000},
                         {"ignore": None, "base": 0.5, "igl": 0.5}),
            {'ignore': 2, 'base': 3000, 'igl': 3000}
        )

    def test_ignore_all(self):
        self.assertDictEqual(
            reweigh_dict({"ignore": 2, "base": 565, "igl": 3000},
                         {"ignore": None, "base": None, "igl": None}),
            {'ignore': 2, 'base': 565, 'igl': 3000}
        )

    def test_zeros(self):
        self.assertDictEqual(
            reweigh_dict({"ignore": 0, "base": 0, "igl": 0},
                         {"ignore": None, "base": None, "igl": None}),
            {'ignore': 0, 'base': 0, 'igl': 0}
        )

    def test_fit_min(self):
        self.assertDictEqual(
            reweigh_dict({"ignore": 2, "base": 50, "igl": 150},
                         {"ignore": None, "base": 0.5, "igl": 0.5},
                         fit="min"),
            {'ignore': 2, 'base': 50, 'igl': 50}
        )


if __name__ == '__main__':

    unittest.main()
