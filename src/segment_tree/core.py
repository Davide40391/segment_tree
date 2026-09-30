"""Core segment tree implementation."""

from dataclasses import dataclass
from typing import Callable, Generic, List, TypeVar

T = TypeVar("T")


@dataclass
class Operation(Generic[T]):
    combine: Callable[[T, T], T]
    apply: Callable[[T, int], T]
    id_element: Callable[[], T]


class _Node:
    __slots__ = ("value", "lazy", "mark")

    def __init__(self):
        self.value = None
        self.lazy = None
        self.mark = False


class SegmentTree(Generic[T]):
    def __init__(self, data, op):
        self._op = op
        self._n = len(data)
        if self._n == 0:
            self._size = 0
            self._tree = []
            self._lazy = []
            self._mark = []
            return
        size = 1
        while size < self._n:
            size *= 2
        self._size = size
        self._tree = [None] * (2 * size)
        self._lazy = [None] * (2 * size)
        self._mark = [False] * (2 * size)
        for i in range(self._n):
            self._tree[size + i] = data[i]
        for i in range(self._n, size):
            self._tree[size + i] = op.id_element()
        for i in range(size - 1, 0, -1):
            self._tree[i] = op.combine(self._tree[2 * i], self._tree[2 * i + 1])

    @property
    def length(self):
        return self._n

    def _normalize_index(self, idx):
        if idx < 0:
            idx += self._n
        if idx < 0 or idx >= self._n:
            raise IndexError(idx)
        return idx

    def _normalize_range(self, left, right):
        if left < 0:
            left += self._n
        if right < 0:
            right += self._n
        if left < 0 or right > self._n or left > right:
            raise IndexError((left, right))
        return left, right

    def _push(self, node, node_lo, node_hi):
        if not self._mark[node]:
            return
        span = node_hi - node_lo
        self._tree[node] = self._op.apply(self._lazy[node], span)
        if node_lo != node_hi - 1:
            mid = (node_lo + node_hi) // 2
            left = 2 * node
            right = 2 * node + 1
            self._lazy[left] = self._lazy[node]
            self._mark[left] = True
            self._lazy[right] = self._lazy[node]
            self._mark[right] = True
        self._mark[node] = False

    def _update_range(self, node, node_lo, node_hi, left, right, value):
        self._push(node, node_lo, node_hi)
        if right <= node_lo or node_hi <= left:
            return
        if left <= node_lo and node_hi <= right:
            self._lazy[node] = value
            self._mark[node] = True
            self._push(node, node_lo, node_hi)
            return
        mid = (node_lo + node_hi) // 2
        self._update_range(2 * node, node_lo, mid, left, right, value)
        self._update_range(2 * node + 1, mid, node_hi, left, right, value)
        self._tree[node] = self._op.combine(self._tree[2 * node], self._tree[2 * node + 1])

    def _query(self, node, node_lo, node_hi, left, right):
        self._push(node, node_lo, node_hi)
        if right <= node_lo or node_hi <= left:
            return self._op.id_element()
        if left <= node_lo and node_hi <= right:
            return self._tree[node]
        mid = (node_lo + node_hi) // 2
        a = self._query(2 * node, node_lo, mid, left, right)
        b = self._query(2 * node + 1, mid, node_hi, left, right)
        return self._op.combine(a, b)

    def query(self, left, right):
        if self._n == 0:
            if left == 0 and right == 0:
                return self._op.id_element()
            raise IndexError((left, right))
        left, right = self._normalize_range(left, right)
        if left == right:
            return self._op.id_element()
        return self._query(1, 0, self._size, left, right)

    def get(self, index):
        if self._n == 0:
            raise IndexError(index)
        index = self._normalize_index(index)
        return self._query(1, 0, self._size, index, index + 1)

    def all(self):
        if self._n == 0:
            return self._op.id_element()
        self._push(1, 0, self._size)
        return self._query(1, 0, self._size, 0, self._n)

    def update_point(self, index, value):
        if self._n == 0:
            raise IndexError(index)
        index = self._normalize_index(index)
        self._update_range(1, 0, self._size, index, index + 1, value)

    def update_range(self, left, right, value):
        if self._n == 0:
            if left == 0 and right == 0:
                return
            raise IndexError((left, right))
        left, right = self._normalize_range(left, right)
        if left == right:
            return
        self._update_range(1, 0, self._size, left, right, value)
