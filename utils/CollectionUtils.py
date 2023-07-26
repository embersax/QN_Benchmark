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

class MinHeap:
    def __init__(self):
        self._data = []
        self._length = 0
    
    def push(self, priority, item):
        self._length += 1
        heapq.heappush(self._data, (priority, item))
                       
    def pop(self):
        self._length -= 1
        return heapq.heappop(self._data)
    
    def get_length(self):
        return self._length