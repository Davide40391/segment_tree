import unittest

from segment_tree import Operation, SegmentTree


class TestSum(unittest.TestCase):
    def setUp(self):
        self.op = Operation(
            combine=lambda a, b: a + b,
            apply=lambda v, span: v * span,
            id_element=lambda: 0,
        )

    def test_basic_query(self):
        t = SegmentTree([1, 2, 3, 4, 5], self.op)
        self.assertEqual(t.query(0, 5), 15)
        self.assertEqual(t.query(1, 4), 9)
        self.assertEqual(t.query(2, 3), 3)

    def test_point_update(self):
        t = SegmentTree([1, 2, 3, 4, 5], self.op)
        t.update_point(2, 10)
        self.assertEqual(t.query(0, 5), 22)
        self.assertEqual(t.get(2), 10)

    def test_range_update_then_query(self):
        t = SegmentTree([1, 2, 3, 4, 5], self.op)
        t.update_range(1, 4, 10)
        self.assertEqual(t.query(1, 4), 30)
        self.assertEqual(t.query(0, 5), 36)
        self.assertEqual(t.get(0), 1)
        self.assertEqual(t.get(4), 5)

    def test_partial_range_update_overlaps(self):
        t = SegmentTree([1, 2, 3, 4, 5, 6, 7, 8], self.op)
        t.update_range(2, 6, 0)
        self.assertEqual(t.query(0, 8), 1 + 2 + 0 + 0 + 0 + 0 + 7 + 8)
        self.assertEqual(t.query(3, 5), 0)
        self.assertEqual(t.query(0, 3), 3)
        self.assertEqual(t.query(6, 8), 15)

    def test_negative_index(self):
        t = SegmentTree([10, 20, 30, 40], self.op)
        self.assertEqual(t.get(-1), 40)
        self.assertEqual(t.get(-4), 10)
        t.update_point(-1, 100)
        self.assertEqual(t.get(-1), 100)
        self.assertEqual(t.query(0, 4), 160)

    def test_empty_range_returns_identity(self):
        t = SegmentTree([1, 2, 3], self.op)
        self.assertEqual(t.query(2, 2), 0)
        self.assertEqual(t.query(0, 0), 0)

    def test_index_error_out_of_range(self):
        t = SegmentTree([1, 2, 3], self.op)
        with self.assertRaises(IndexError):
            t.get(5)
        with self.assertRaises(IndexError):
            t.get(-4)
        with self.assertRaises(IndexError):
            t.query(0, 4)
        with self.assertRaises(IndexError):
            t.update_range(0, 4, 0)

    def test_empty_tree(self):
        t = SegmentTree([], self.op)
        self.assertEqual(t.length, 0)
        self.assertEqual(t.query(0, 0), 0)
        self.assertEqual(t.all(), 0)

    def test_single_element(self):
        t = SegmentTree([42], self.op)
        self.assertEqual(t.query(0, 1), 42)
        self.assertEqual(t.all(), 42)
        t.update_range(0, 1, 7)
        self.assertEqual(t.get(0), 7)
        self.assertEqual(t.all(), 7)

    def test_length_property(self):
        t = SegmentTree([1, 2, 3, 4, 5], self.op)
        self.assertEqual(t.length, 5)


class TestMin(unittest.TestCase):
    def setUp(self):
        self.op = Operation(
            combine=min,
            apply=lambda v, span: v,
            id_element=lambda: float("inf"),
        )

    def test_min_query(self):
        t = SegmentTree([5, 3, 8, 1, 9, 2], self.op)
        self.assertEqual(t.query(0, 6), 1)
        self.assertEqual(t.query(0, 3), 3)
        self.assertEqual(t.query(3, 6), 1)

    def test_min_range_update(self):
        t = SegmentTree([5, 3, 8, 1, 9, 2], self.op)
        t.update_range(1, 4, 0)
        self.assertEqual(t.query(1, 4), 0)
        self.assertEqual(t.query(0, 6), 0)
        self.assertEqual(t.get(0), 5)
        self.assertEqual(t.get(5), 2)


class TestConcat(unittest.TestCase):
    """Verifies non-commutative associative operations work in correct order."""

    def setUp(self):
        self.op = Operation(
            combine=lambda a, b: a + b,
            apply=lambda v, span: v * span,
            id_element=lambda: "",
        )

    def test_concat_order(self):
        t = SegmentTree(["a", "b", "c", "d"], self.op)
        self.assertEqual(t.query(0, 4), "abcd")
        self.assertEqual(t.query(1, 3), "bc")
        self.assertEqual(t.query(2, 4), "cd")

    def test_concat_range_update(self):
        t = SegmentTree(["a", "b", "c", "d"], self.op)
        t.update_range(1, 3, "x")
        self.assertEqual(t.query(0, 4), "axxd")
        self.assertEqual(t.get(0), "a")
        self.assertEqual(t.get(1), "x")
        self.assertEqual(t.get(2), "x")
        self.assertEqual(t.get(3), "d")


class TestMax(unittest.TestCase):
    def setUp(self):
        self.op = Operation(
            combine=max,
            apply=lambda v, span: v,
            id_element=lambda: float("-inf"),
        )

    def test_max_query_and_update(self):
        t = SegmentTree([1, 5, 3, 9, 2, 7], self.op)
        self.assertEqual(t.query(0, 6), 9)
        self.assertEqual(t.query(2, 5), 9)
        t.update_range(2, 5, 100)
        self.assertEqual(t.query(0, 6), 100)
        self.assertEqual(t.get(0), 1)
        self.assertEqual(t.get(5), 7)


class TestNonPowerOfTwo(unittest.TestCase):
    """Sizes that are not powers of two exercise padding and partial spans."""

    def setUp(self):
        self.op = Operation(
            combine=lambda a, b: a + b,
            apply=lambda v, span: v * span,
            id_element=lambda: 0,
        )

    def test_size_three(self):
        t = SegmentTree([1, 2, 3], self.op)
        self.assertEqual(t.query(0, 3), 6)
        t.update_range(0, 2, 10)
        self.assertEqual(t.query(0, 3), 23)
        self.assertEqual(t.query(0, 2), 20)
        self.assertEqual(t.query(2, 3), 3)

    def test_size_seven(self):
        data = list(range(1, 8))
        t = SegmentTree(data, self.op)
        self.assertEqual(t.query(0, 7), 28)
        t.update_range(2, 6, 0)
        self.assertEqual(t.query(0, 7), 1 + 2 + 7)
        self.assertEqual(t.query(0, 2), 3)
        self.assertEqual(t.query(6, 7), 7)

    def test_overlapping_range_updates(self):
        t = SegmentTree([0, 0, 0, 0, 0, 0, 0, 0], self.op)
        t.update_range(0, 8, 1)
        self.assertEqual(t.query(0, 8), 8)
        t.update_range(2, 6, 3)
        self.assertEqual(t.query(0, 8), 4 + 12)
        self.assertEqual(t.query(2, 6), 12)
        self.assertEqual(t.get(0), 1)
        self.assertEqual(t.get(2), 3)
        self.assertEqual(t.get(6), 1)
        self.assertEqual(t.get(7), 1)


if __name__ == "__main__":
    unittest.main()
