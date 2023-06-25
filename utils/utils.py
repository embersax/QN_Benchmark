import numpy as np
from itertools import combinations, groupby
import math
from collections.abc import MutableMapping


def dynSearch(xMin, xMax, yTarget, f, fIsIncreasing, precision):
    x = (xMin + xMax) / 2
    step = x
    for i in range(0, 101):
        step /= 2
        y = f(x)
        if abs(y - yTarget) < abs(precision):
            a = abs(y - yTarget)
            break
        if (y > yTarget) ^ fIsIncreasing:
            x += step
        else:
            x -= step
    return x


def groupby_dict(s, f):
    tmp_dict = {}
    for a, b in groupby(s, f):
        tmp = list(b)
        # for thing in b:
        tmp_dict[a] = tmp
        # tmp[a] = b
    return tmp_dict


def groupby_dict_(s, f):
    tmp_dict = {}
    for a, b in groupby(s, f):
        tmp = list(b)
        # for thing in b:
        if a in tmp_dict:
            tmp_dict[a].append(tmp[0])
        else:
            tmp_dict[a] = tmp
    return tmp_dict


# here I implemet factorial as two input functions instead of an extended function of Int as Shouqian does
def factorial_(n, until=1):
    # initial = 0
    # for i in range(until + 1, first+1):
    #     initial *= i
    # return initial
    result = 1
    for i in range(until, n + 1):
        result *= i
    return result




def C_(totalNum, targetNum):
    return math.factorial(totalNum) // (math.factorial(targetNum) * math.factorial(totalNum - targetNum))


def list_minus(list1, list2):
    result = []
    for i in range(len(list1)):
        result.append(list1[i] - list2[i])
    return result


def length(x):
    return np.linalg.norm(np.array(x))


#  here I din't extend dict class directly , I tried something following the link of this
#  https://stackoverflow.com/questions/3387691/how-to-perfectly-override-a-dict which I think is a better implememnbtation
#  the different thing of this ReducibleLazyEvaluatio is that for _get_ method, it uses three functions as input which I treated them
#  as attribute to init
#  initializer calculates a value based on key, pre transform a new key, post calculates a value based on (key,value)
#  There are use cases that pre and post default is to be unchanged, and only initilizer needed to be passed .
class ReducibleLazyEvaluation(MutableMapping):
    """A dictionary that applies an arbitrary key-altering
       function before accessing the keys"""

    def __init__(self, initilizer, pre=lambda x: x, post=lambda K, V: V):
        self.store = dict()
        # self.update(dict(*args, **kwargs))  # use the free update to set keys
        self.initializer = initilizer
        self.pre = pre
        self.post = post

    def __getitem__(self, key):
        # return self.store[self._keytransform(key)]
        res = None
        res = self.store[self._keytransform(self.pre(key))]
        if res is not None:
            return self.post(key, res)
        res = self.initializer(key)
        self.store[self._keytransform(key)] = res

        return self.post(key, res)

    def __setitem__(self, key, value):
        self.store[self._keytransform(key)] = value

    def __delitem__(self, key):
        del self.store[self._keytransform(key)]

    def __iter__(self):
        return iter(self.store)

    def __len__(self):
        return len(self.store)

    def _keytransform(self, key):
        return key


def node_dict(p):
    result = {}
    for node in p:
        result[node] = set()

    return result


def removeUntil(T, f):
    while T is not []:
        e = T.pop(0)
        if f(e):
            return e
    return None


def foldLeft(fun, arr, initial=None):
    """
    fun takes in acc and element and returns acc everytime after applying some function
    """
    if initial is None:
        acc = arr[0]
        del arr[0]
    else:
        acc = initial

    for element in arr:
        acc = fun(acc, element)

    return acc


def flatMap(xs, f=lambda x:x):
    ys = []
    for x in xs:
        ys.extend(f(x))
    return ys

def listtoString(s, seperator):
    s = list(map(lambda x: str(x), s))
    return seperator.join(s)

def divide_chunks(l, n):
    # looping till length l
    for i in range(0, len(l), n):
        yield l[i:i + n]

def flat_map(xs, f=lambda x:x):
    ys = []
    for x in xs:
        ys.extend(f(x))
    return ys