import numpy as np


class SumTree:
    """Efficiently stores and retrieves priorities for PER."""
    def __init__(self, capacity):
        """Initializes SumTree with given capacity."""
        self.capacity = capacity
        # Binary tree array
        self.tree = np.zeros(2 * capacity - 1)
        # Data array for transitions
        self.data = np.zeros(capacity, dtype=object)
        self.data_pointer = 0
        self.size = 0
        self.max_priority_index = capacity - 1
        self.min_priority_index = capacity - 1

    def update(self, tree_index, priority):
        """Updates priority of a specific leaf and propagates changes."""
        max_p, min_p = self.tree[self.max_priority_index], self.tree[self.min_priority_index]

        change = priority - self.tree[tree_index]
        self.tree[tree_index] = priority

        # Update max/min priority indices
        if priority >= max_p:
            self.max_priority_index = tree_index
        elif tree_index == self.max_priority_index:
            self.max_priority_index = np.argmax(self.tree[self.capacity-1:self.capacity+self.size-1]) + self.capacity - 1
        if priority <= min_p:
            self.min_priority_index = tree_index
        elif tree_index == self.min_priority_index:
            self.min_priority_index = np.argmin(self.tree[self.capacity-1:self.capacity+self.size-1]) + self.capacity - 1

        # Propagate change up the tree
        while not tree_index == 0:
            tree_index = (tree_index - 1) // 2
            self.tree[tree_index] += change

    def add(self, data, priority):
        """Adds new data with given priority."""
        tree_index = self.data_pointer + self.capacity - 1
        self.data[self.data_pointer] = data
        self.data_pointer += 1
        if self.data_pointer >= self.capacity:
            self.data_pointer = 0

        self.size += 1
        if self.size > self.capacity:
             self.size = self.capacity

        self.update(tree_index, priority)

    def get_leaf(self, v):
        """Retrieves leaf node for a given cumulative priority value."""
        parent_index = 0

        while True:
            left_child_index = 2 * parent_index + 1
            right_child_index = left_child_index + 1

            if left_child_index >= len(self.tree):
                leaf_index = parent_index
                break
            else:
                if v <= self.tree[left_child_index]:
                    parent_index = left_child_index
                else:
                    v -= self.tree[left_child_index]
                    parent_index = right_child_index

        data_index = leaf_index - self.capacity + 1

        return leaf_index, self.tree[leaf_index], self.data[data_index]

    @property
    def total_priority(self):
        """Returns sum of all priorities."""
        return self.tree[0]

    @property
    def max_priority(self):
        """Returns maximum priority in the tree."""
        return self.tree[self.max_priority_index]

    @property
    def min_priority(self):
        """Returns minimum priority in the tree."""
        return self.tree[self.min_priority_index]


