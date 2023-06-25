from utils.utils import ReducibleLazyEvaluation
from dataclasses import dataclass
from itertools import takewhile
from enum import Enum
import random
import re
import os

randGenSeed = 19900111
random.seed(randGenSeed)

edgeLen = 100.0
maxSearchHops = 15

nList = [100, 50, 200, 400, 800]
dList = [6, 5, 4, 3]
kList = [3, 6, 0, 10000]
qList = [0.9, 0.8, 0.85, 0.95, 1.0]
pList = [0.6, 0.3, 0.9, 0.1]
# It has to be until 11 (it is until 10 in the kotlin code) because ranges in python are non-inclusive.
nsdList = [i for i in range(10, 11)]
topoRange = [1]

allAvailableSettings = []
for d in dList:
    for p in pList:
        for n in nList:
            for q in qList:
                for k in kList:
                    for nsd in nsdList:
                        deviation = 0
                        if k != kList[0]: deviation += 1
                        if d != dList[0]: deviation += 1
                        if q != qList[0]: deviation += 1
                        if p != pList[0]: deviation += 1
                        if n != nList[0]: deviation += 1
                        if nsd != nsdList[-1]: deviation += 1

                        if deviation <= 1: allAvailableSettings.append((d, n, p, q, k, nsd))

referenceSetting = (dList[0], nList[0], pList[0], qList[0], kList[0], nsdList[-1])


# """$n#$topoIdx-$q-$k-$p-$d-$numPairs-${name}"""
def id(n, topoIdx, q, k, p, d, numPairs, name): return "{}#{}-{}-{}-{}-{}-{}-{}".format(
    n, topoIdx, q, k, p, d, numPairs, name)




# TODO: reducible lazy evaluation part. Not sure how to do it with only two params. Will do it with a dict meanwhile.
records = {}


class Type(Enum):
    Online = 1
    Offline = 0


@dataclass
class RecoveryPath2:
    path: []
    width: int
    good: int
    taken: int


@dataclass
class RecoveryPath1:
    path: []
    occupiedChannels: int
    goodChannels: int


@dataclass
class MajorPath:
    path: []
    width: int
    succ: int
    type: Type
    # The type of the elements of the recoveryPath list is RecoveryPath2
    recoveryPaths: []


@dataclass
class Record:
    # ops: List<Pair<Int, Int>>
    ops: []
    majorPaths: []
    rpCnt: int
    rpChannelCnt: int


def parseLog(fn):
    f = open(fn, 'r')
    if not records[os.path.splitext(fn)[0]]:
        try:
            currRecord = None
            currMajorPath = None
            for line in f:
                if line.startswith('-----'): continue
                if line.strip() == '' or not line.strip(): continue
                try:
                    indent = len(list(takewhile(lambda x: x == ' ', line)))
                    if indent == 0:
                        currMajorPath = None
                        if currRecord is not None:
                            if len(records[os.path.splitext(fn)[0]]) == 0:
                                records[os.path.splitext(fn)[0]] = [currRecord]
                            else:
                                records[os.path.splitext(fn)[0]].append(currRecord)
                            # TODO: Need to check out the \d escape sequence.
                            tmp0 = re.compile("[^\d]+").split(line)
                            tmp1 = [int(x) for x in tmp0]
                            tmp2 = zip(*[iter(tmp1)] * 2)
                            currRecord = Record(tmp2, [], 0, 0)
                    elif indent == 1:
                        if line.contains('recovery'):
                            seg = re.compile("[^\d]+").split(line.strip())[1:-1]
                            taken = [int(x) for x in seg[-1]]
                            currRecord.rpCnt += 1
                            currRecord.rpChannelCnt += taken
                        else:
                            if not line.contains("[") or not line.contains("],"): raise Exception('incomplete')
                            l = line
                            type = Type.Online
                            if line.contains('//'):
                                if line.contains('offline'): type = Type.Offline
                                l = line.split('//')[0].strip()

                            seg = re.compile("[^\d]+").split(l.strip())[1:]
                            tmp = [int(x) for x in seg[-2:]]
                            width, succ = tmp[0], tmp[1]
                            currMajorPath = MajorPath([int(x) for x in seg[:-2]], width, succ, type, [])
                            if (currMajorPath.path[0], currMajorPath.path[-1]) not in currRecord.ops: raise Exception(
                                'incomplete')
                            currRecord.majorPaths.append(currMajorPath)
                    else:
                        seg = re.compile("[^\d]+").split(line.strip())[1:]
                        tmp = [int(x) for x in seg[-3:]]
                        width, succ, taken = tmp[0], tmp[1], tmp[2]
                        currMajorPath.recoveryPaths.append(
                            RecoveryPath2([int(x) for x in seg[:-3]], width, succ, taken))

                except:
                    currRecord = None
                    currMajorPath = None
        except:
            pass
    return records[os.path.splitext(fn)[0]]
