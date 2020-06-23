import numpy as np

class SumTree:
    pointer = 0
    def __init__(self, capacity):
        """
        # of leaf nodes = capaticy
        # of all nodes in the tree = 2 * capacity -1
        """
        self.capacity = capacity
        self.tree = np.zeros(2 * capacity -1)
        self.data = np.zeros(capacity, dtype =object)

    def total(self):
        return self.tree[0]

    def add(self, priority, data):
        """
        1. update tree
        2. add transition data
        3. adjust pointer
        """
        # update tree
        tree_idx = self.pointer + self.capacity -1
        self.update(tree_idx, priority)
        # add transition data
        self.data[self.pointer] = data
        # adjust pointer; when exceed the capacity, restart from the beginning
        self.pointer += 1
        if self.pointer >= self.capacity:
            self.pointer = 0

    def update(self, tree_idx, priority):
        """
        1. update the leaf node priority
        2. update all the related parent nodes (add change to the old parent node value)
        """
        # difference between the new value and old value
        change = priority - self.tree[tree_idx]

        # update the leaf node priority
        self.tree[tree_idx]  = priority

        while tree_idx !=0:
            tree_idx = (tree_idx - 1)//2 # the index of parent node
            self.tree[tree_idx] += change

    def _retrieve(self, idx, s):
        """
        idx: parent node index
        left: left child node index
        right: right child node index
        """
        left = 2 * idx + 1
        right = left + 1

        # reach bottom, end search
        if left >= len(self.tree):
            return idx

        if s <= self.tree[left]:
            # put the left child node as the new parent node, keep searching
            return self._retrieve(left, s)
        else:
            # put the right child node as the new parent node, keep searching
            return self._retrieve(right, s-self.tree[left])

    def get(self, s):
        idx = self._retrieve(0, s)
        dataIdx = idx - self.capacity + 1
        return (idx, self.tree[idx], self.data[dataIdx])
