#!/usr/bin/env python
from prio_replay import *
import unittest

class TestSumTree(unittest.TestCase):
  def test_indexof_simple(self):
    st = SumTree(num_elements=4)
    for i, v in enumerate([3, 1, 4, 1]): st.update(i, v)
    self.assertEqual(st.total(), 9)
    self.assertEqual(st.size, 4)
    for v, expected_idx in zip(range(9),
                               [0,0,0, 1, 2,2,2,2, 3]):
      self.assertEqual(st.index_of(v), expected_idx)

  def test_indexof_partially_filled(self):
    st = SumTree(num_elements=8)
    for i, v in enumerate([3, 1, 4, 1]): st.update(i, v)
    self.assertEqual(st.total(), 9)
    self.assertEqual(st.size, 4)
    for v, expected_idx in zip(range(9),
                               [0,0,0, 1, 2,2,2,2, 3]):
      self.assertEqual(st.index_of(v), expected_idx)

  def test_indexof_with_zeros(self):
    st = SumTree(num_elements=4)
    for i, v in enumerate([3, 0, 4, 0]): st.update(i, v)
    self.assertEqual(st.total(), 7)
    self.assertEqual(st.size, 4)
    for v, expected_idx in zip(range(7),
                               [0,0,0, 2,2,2,2]):
      self.assertEqual(st.index_of(v), expected_idx)

  def test_indexof_throws_error_when_empty(self):
    st = SumTree(num_elements=4)
    self.assertEqual(st.total(), 0)
    self.assertEqual(st.size, 0)
    with self.assertRaises(ValueError):
      st.index_of(0)

  def test_indexof_growing(self):
    st = SumTree(num_elements=4)

    st.update(0, 3)
    st.update(1, 4)
    self.assertEqual(st.total(), 7)
    self.assertEqual(st.size, 2)
    for v, expected_idx in zip(range(7),
                               [0,0,0, 1,1,1,1]):
      self.assertEqual(st.index_of(v), expected_idx)

    st.update(2, 2)
    st.update(3, 2)
    self.assertEqual(st.total(), 11)
    self.assertEqual(st.size, 4)
    for v, expected_idx in zip(range(11),
                               [0,0,0, 1,1,1,1, 2,2, 3,3]):
      self.assertEqual(st.index_of(v), expected_idx)

  def test_indexof_wrapping(self):
    st = SumTree(num_elements=4)

    # fill tree
    for i, v in enumerate([3, 4, 2, 2]): st.update(i, v)
    self.assertEqual(st.total(), 11)
    self.assertEqual(st.size, 4)
    for v, expected_idx in zip(range(11),
                               [0,0,0, 1,1,1,1, 2,2, 3,3]):
      self.assertEqual(st.index_of(v), expected_idx)

    # wraps and clobbers 1st element => [5, 4, 2, 2]
    st.update(0, 5)
    self.assertEqual(st.total(), 13)
    self.assertEqual(st.size, 4)
    for v, expected_idx in zip(range(13),
                               [0,0,0,0,0, 1,1,1,1, 2,2, 3,3]):
      self.assertEqual(st.index_of(v), expected_idx)

    # wraps and clobbers 2nd element => [5, 0, 2, 2]
    st.update(1, 0)
    self.assertEqual(st.total(), 9)
    self.assertEqual(st.size, 4)
    for v, expected_idx in zip(range(9),
                               [0,0,0,0,0, 2,2, 3,3]):
      self.assertEqual(st.index_of(v), expected_idx)

    # clobber last two...
    st.update(2, 0)
    st.update(3, 0)
    # ... and rewrites entire set
    for i, v in enumerate([5,9,2,6]): st.update(i, v)
    self.assertEqual(st.total(), 22)
    self.assertEqual(st.size, 4)
    for v, expected_idx in zip(range(22),
                               [0]*5 + [1]*9 + [2]*2 + [3]*6):
      self.assertEqual(st.index_of(v), expected_idx)

  def __test_update(self):
    st = SumTree(num_elements=4)
    # fill tree
    for v in [3, 4, 2, 2]: st.add(v)
    st.dump()
    # update two elements
    st.update(2, 6)  # => [3, 4, 6, 2]
    st.dump()
    st.update(0, 0)  # => [0, 4, 6, 2]
    st.dump()
    # check index
    self.assertEqual(st.total(), 12)
    for v, expected_idx in zip(range(12),
                               [1,1,1,1, 2,2,2,2,2,2, 3,3]):
      self.assertEqual(st.index_of(v), expected_idx)

  def __test_indexof_sample(self):
    st = SumTree(num_elements=8)
    for v in [5,9,2,6,5,3,5,8]: st.add(v)
    from collections import Counter
    c = Counter()
    for _ in range(10):
      c.update(st.sample(3))
    vs = []
    for k, v in c.iteritems():
      print k, v
      vs.append(v)
    print vs

if __name__ == '__main__':
  unittest.main()
