# Segment Tree

A small Python library for answering range queries and applying range updates over an associative operation. All ranges are half-open `[left, right)`.

```python
from segment_tree import SegmentTree, Operation

sum_op = Operation(
    combine=lambda a, b: a + b,
    apply=lambda v, span: v * span,
    id_element=lambda: 0,
)

t = SegmentTree([1, 2, 3, 4, 5], sum_op)
t.update_range(1, 4, 10)
print(t.query(0, 5))  # 41
print(t.get(2))        # 10
print(t.all())        # 41
```

## Why this exists

The point of a segment tree is O(log n) range queries and lazy range updates. The catch is that lazy propagation requires knowing, for any update value, the aggregate over an arbitrary run of that value — including internal node spans that are not powers of two. That is what `Operation.apply(value, span)` provides. Separating `combine`, `apply`, and `id_element` into one dataclass avoids the common bug of overloading a single lambda to mean all three.

This library uses a boolean *mark* flag on each lazy node rather than a sentinel value to represent "no pending update". A sentinel fails the moment the operation's identity is itself a legitimate update value (e.g. appending the empty string, or setting a count to zero).

## Behaviour to know

- The operation must be associative; commutativity is not required. Operands are combined in left-to-right order, so string concatenation works.
- Indexing is Python-style: negative indices wrap from the end, out-of-range access raises `IndexError` (no silent clamping), and an empty range returns the identity element.
- `apply` must be correct for every span the tree materialises, not just `span == 1`. For `min`/`max` use `lambda v, span: v`; for `sum` use `lambda v, span: v * span`.

## Exported names

- `Operation` — dataclass with fields `combine`, `apply`, `id_element`.
- `SegmentTree(data, op)` — methods: `query(left, right)`, `get(index)`, `all()`, `update_point(index, value)`, `update_range(left, right, value)`, and the `length` property.

## Design notes

The window stores values eagerly rather than keeping running aggregates. Running
sums drift with floating point over long streams, and recomputing from a small
buffer is cheap enough that the drift is not worth the speed.

