"""Reducible Lazy Evaluation"""
import heapq


class ReducibleLazyEvaluation:
    def __init__(self, K, V):
        self.K = K
        self.V = V

    def evaluate(self, dictionary, K, V):
        keys = dictionary.keys()
        if keys and keys is not None:
            return dictionary.values()
        dictionary[K] = V
        return dictionary


class PriorityQueue:
    """A max heap"""

    def __init__(self):
        self._data = []
        self._index = 0

    def push(self, item, priority):
        heapq.heappush(self._data, (-priority, self._index, item))
        self._index += 1

    def pop(self):
        return heapq.heappop(self._data)[-1]
