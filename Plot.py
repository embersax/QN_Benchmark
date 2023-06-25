from configs import referenceSetting, topoRange, parseLog, id, qList, kList, dList, pList, nList, nsdList
import numpy as np
from configs import *
from utils.utils import *
from itertools import groupby
from string import Template
import statistics


class Plot:
    def __init__(self):
        self.chidren=[]
        self.nameMapping={"SL":"SLMP","Online":"Q-CAST","Online-R":"Q-CAST\\\\R", "CR":"Q-PASS","CR-R":"Q-PASS\\\\R","Greedy_H" :"Greedy"}
        self.names=["Online", "SL", "Greedy_H", "CR", "BotCap", "SumDist", "MultiMetric"]

    def rp2Cdf_nsd(self):

        def getName1(d, n, p, q, k):
            return "a2-rp-len-cdf-{}-{}-{}-{}-{}-nsd".format(d, n, p, q, k).replace('.', '')

        def getSolutionList1(nsdList):
            return list(map(lambda it: '{} S-D pairs'.format(it) if it > 1 else '{} S-D pair'.format(it), sorted(nsdList)))

        def getX1(max1):
            return list(range(max1+1))

        def getY1(result1, max1):
            return list(map(lambda l: l+[1.0 for _ in range(max1+1-len(l))] if len(l) < max1+1 else l, result1))

        def getName2(d, n, p, q, k):
            return "a2-rp-wid-cdf-{}-{}-{}-{}-{}-nsd".format(d, n, p, q, k).replace('.', '')

        def getSolutionList2(nsdList):
            return list(map(lambda it: '{} S-D pairs'.format(it) if it > 1 else '{} S-D pair'.format(it), sorted(nsdList)))

        def getX2(max2):
            return list(range(max2+1))

        def getY2(result2, max2):
            return list(map(lambda l: l+[1.0 for _ in range(max2+1-len(l))] if len(l) < max2+1 else l, result2))

        def getName3(d, n, p, q, k):
            return "a2-rp-wid-per-mp-cdf-{}-{}-{}-{}-{}-nsd".format(d, n, p, q, k).replace('.', '')

        def getSolutionList3(nsdList):
            return list(map(lambda it: '{} S-D pairs'.format(it) if it > 1 else '{} S-D pair'.format(it), sorted(nsdList)))

        def getX3(max3):
            return list(range(max3+1))

        def getY3(result3, max3):
            return list(map(lambda l: l+[1 for _ in range(max3+1-len(l))] if len(l) < max3+1 else l, result3))

        def getName4(d, n, p, q, k):
            return "rp-throughput-cdf-{}-{}-{}-{}-{}-nsd".format(d, n, p, q, k).replace('.', '')

        def getSolutionList4(names):
            return list(map(lambda it: ' {} '.format(self.nameMapping[it]) if it in self.nameMapping else it, names))

        def getX4(max4):
            return list(range(max4+1))

        def getSubX4(max4):
            x = getX4(max4)
            return list(map(lambda it: it[0], [x[i:i+5] for i in range(0, len(x), 5)]))

        def getY4(result4, max4):
            return list(map(lambda l: l+[1 for _ in range(max4+1-len(l))] if len(l) < max4+1 else l, result4))


        d, n, p, q, k, nsd = referenceSetting
        max1 = 0
        result1 = []

        for idx in range(len(nsdList)):
            nsd = nsdList[idx]
            s = 0
            summ = ReducibleLazyEvaluation(lambda x: 0)
            y = flatMap(flatMap(topoRange, lambda topoIdx: parseLog('dist/'+id(n, topoIdx, q, k, p, d, nsd, 'Online')+'.txt')), lambda it: flatMap(it.majorPaths, lambda it: map(lambda it: len(it.path)-1, it.recoveryPaths)))

            for k, v in sorted(list(map(lambda k, v: (k, len(v)), list(groupby(y, lambda it: it)))), key=lambda it: it[0]):
                max1 = max(max1, k)
                s += v
                for i in range(k, 1001):
                    summ[i] = s

            res = list(map(lambda it: summ[it]/s, list(range(max1+1))))
            result1.append(res)

        temp1 = Template("""
        {
          "markerSize": 0,
          "name": "${getName1}",
          "solutionList": ${getSolutionList1},
          "legendColumn": 2,
          "legendFontSize": 10,
          "legendAutomaticallyReorder": False,
          "xTitle": "Length of recovery path",
          "yTitle": "CDF",
          "xLimit": [0, ${max1}],
          "yLimit": [0, 1],
          "x": ${getX1},
          "y": ${getY1}
        }""")

        string1 = temp1.substitute(getName1=getName1(d, n, p, q, k), getSolutionList1=getSolutionList1(nsdList), max1=max1, getX1=getX1(max1), getY1=getY1(result1, max1))
        self.children.append(string1)

        max2 = 0
        result2 = []

        for idx in range(len(nsdList)):
            nsd = nsdList[idx]
            s = 0
            summ = ReducibleLazyEvaluation(lambda x: 0)
            y = flatMap(flatMap(topoRange,
                                lambda topoIdx: parseLog('dist/' + id(n, topoIdx, q, k, p, d, nsd, 'Online') + '.txt')),
                        lambda it: flatMap(it.majorPaths, lambda it: map(lambda it: it.width, it.recoveryPaths)))

            for k, v in sorted(list(map(lambda k, v: (k, len(v)), list(groupby(y, lambda it: it)))),
                               key=lambda it: it[0]):
                max2 = max(max2, k)
                s += v
                for i in range(k, 1001):
                    summ[i] = s

            res = list(map(lambda it: summ[it] / s, list(range(max2 + 1))))
            result2.append(res)

        temp2 = Template("""
        {
          "markerSize": 0,
          "name": "${getName2}",
          "solutionList": ${getSolutionList2},
          "legendColumn": 2,
          "legendFontSize": 10,
          "legendAutomaticallyReorder": False,
          "xTitle": "Width of recovery path",
          "yTitle": "CDF",
          "xLimit": [0, ${max2}],
          "yLimit": [0, 1],
          "x": ${getX2},
          "y": ${getY2}
        }""")

        string2 = temp2.substitute(getName2=getName2(d, n, p, q, k), getSolutionList2=getSolutionList2(nsdList),
                                 max2=max2, getX2=getX2(max2), getY2=getY2(result2, max2))
        self.children.append(string2)

        max3 = 0
        result3 = []

        for idx in range(len(nsdList)):
            nsd = nsdList[idx]
            s = 0
            summ = ReducibleLazyEvaluation(lambda x: 0)
            y = flatMap(flatMap(topoRange,
                                lambda topoIdx: parseLog('dist/' + id(n, topoIdx, q, k, p, d, nsd, 'Online') + '.txt')),
                        lambda it: flatMap(it.majorPaths, lambda it: map(lambda it: it.width, it.recoveryPaths)))

            for k, v in sorted(list(map(lambda k, v: (k, len(v)), list(groupby(y, lambda it: it)))),
                               key=lambda it: it[0]):
                max3 = max(max3, k)
                s += v
                for i in range(k, 1001):
                    summ[i] = s

            res = list(map(lambda it: summ[it] / s, list(range(max3 + 1))))
            result3.append(res)

        temp3 = Template("""
        {
          "markerSize": 0,
          "name": "${getName3}",
          "solutionList": ${getSolutionList3},
          "legendColumn": 2,
          "legendFontSize": 10,
          "legendAutomaticallyReorder": False,
          "xTitle": "# recovery paths per major path",
          "yTitle": "CDF",
          "xLimit": [0, ${max3}],
          "yLimit": [0, 1],
          "x": ${getX3},
          "y": ${getY3}
        }""")

        string3 = temp3.substitute(getName3=getName3(d, n, p, q, k), getSolutionList3=getSolutionList3(nsdList),
                                   max3=max3, getX3=getX3(max3), getY3=getY3(result3, max3))
        self.children.append(string3)
        maxx, names, result = 0, ['Online', 'Online-R', 'CR', 'CR-R'], []

        for name in names:
            s = 0
            summ = ReducibleLazyEvaluation(lambda x: 0)
            y = map(lambda it: sum(it.majorPaths.succ), flatMap(topoRange,
                                lambda topoIdx: parseLog('dist/' + id(n, topoIdx, q, k, p, d, nsd, name) + '.txt'),
                        ))

            for k, v in sorted(list(map(lambda k, v: (k, len(v)), list(groupby(y, lambda it: it)))),
                               key=lambda it: it[0]):
                maxx = max(maxx, k)
                s += v
                for i in range(k, 1001):
                    summ[i] = s

            res = list(map(lambda it: summ[it] / s, list(range(maxx + 1))))
            result.append(res)

        temp4 = Template("""
        {
          "markerSize": 0,
            "name": "${getName4}",
          "solutionList": ${getSolutionList4},
          "xTitle": "Throughput (eps)",
          "yTitle": "CDF",
          "xLimit": [0, ${max4}],
          "yLimit": [0, 1],
          "x": ${getX4},
            "xTicks&Labels": {getSubX4}
          "y": ${getY4}
        }""")

        string4 = temp4.substitute(getName4=getName4(d, n, p, q, k), getSolutionList4=getSolutionList4(nsdList),
                                   max4=maxx, getX4=getX4(maxx), getSubX4=getSubX4(maxx), getY3=getY4(result, maxx))
        self.children.append(string4)


    def rp2Cdf_n(self):

        def getName1(d, n, p, q, k):
            return "a2-rp-len-cdf-{}-{}-{}-{}-{}-nsd".format(d, n, p, q, k).replace('.', '')

        def getSolutionList1(nList):
            return list(map(lambda it: ' |V| = {} '.format(it), sorted(nList)))

        def getX1(max1):
            return list(range(max1+1))

        def getY1(result1, max1):
            return list(map(lambda l: l+[1.0 for _ in range(max1+1-len(l))] if len(l) < max1+1 else l, result1))


        def getName2(d, n, p, q, k):
            return "a2-rp-wid-cdf-{}-{}-{}-{}-{}-nsd".format(d, n, p, q, k).replace('.', '')

        def getSolutionList2(nList):
            return getSolutionList1(nList)

        def getX2(max2):
            return list(range(max2+1))

        def getY2(result2, max2):
            return list(map(lambda l: l+[1.0 for _ in range(max2+1-len(l))] if len(l) < max2+1 else [l[idx] for idx in range(len(l)) if idx <= max2], result2))


        def getName3(d, n, p, q, k):
            return "a2-rp-wid-per-mp-cdf-{}-{}-{}-{}-{}-nsd".format(d, n, p, q, k).replace('.', '')

        def getSolutionList3(nList):
            return list(map(lambda it: ' |V| = {} '.format(it), sorted(nList)))

        def getX3(max3):
            return list(range(max3+1))

        def getY3(result3, max3):
            return list(map(lambda l: l+[1 for _ in range(max3+1-len(l))] if len(l) < max3+1 else l, result3))

        d, n, p, q, k, nsd = referenceSetting
        max1 = 0
        result1 = []

        for idx in range(len(nList)):
            nsd = nList[idx]
            s = 0
            summ = ReducibleLazyEvaluation(lambda x: 0)
            y = flatMap(flatMap(topoRange,
                                lambda topoIdx: parseLog('dist/' + id(n, topoIdx, q, k, p, d, nsd, 'Online') + '.txt')),
                        lambda it: flatMap(it.majorPaths,
                                           lambda it: map(lambda it: len(it.path) - 1, it.recoveryPaths)))

            for k, v in sorted(list(map(lambda k, v: (k, len(v)), list(groupby(y, lambda it: it)))),
                               key=lambda it: it[0]):
                max1 = max(max1, k)
                s += v
                for i in range(k, 1001):
                    summ[i] = s

            res = list(map(lambda it: summ[it] / s, list(range(max1 + 1))))
            result1.append(res)

        temp1 = Template("""
        {
          "markerSize": 0,
          "name": "${getName1}",
          "solutionList": ${getSolutionList1},
          "xTitle": "Length of recovery path",
          "yTitle": "CDF",
          "xLimit": [0, ${max1}],
          "yLimit": [0, 1],
          "x": ${getX1},
          "y": ${getY1}
        }""")

        string1 = temp1.substitute(getName1=getName1(d, n, p, q, k), getSolutionList1=getSolutionList1(nsdList),
                                   max1=max1, getX1=getX1(max1), getY1=getY1(result1, max1))
        self.children.append(string1)

        max2 = 0
        result2 = []

        for idx in range(len(nList)):
            nsd = nList[idx]
            s = 0
            summ = ReducibleLazyEvaluation(lambda x: 0)
            y = flatMap(flatMap(topoRange,
                                lambda topoIdx: parseLog('dist/' + id(n, topoIdx, q, k, p, d, nsd, 'Online') + '.txt')),
                        lambda it: flatMap(it.majorPaths, lambda it: map(lambda it: it.width, it.recoveryPaths)))

            for k, v in sorted(list(map(lambda k, v: (k, len(v)), list(groupby(y, lambda it: it)))),
                               key=lambda it: it[0]):
                max2 = max(max2, k)
                s += v
                for i in range(k, 1001):
                    summ[i] = s

            res = list(map(lambda it: summ[it] / s, list(range(max2 + 1))))
            result2.append(res)

        max2 = max(list(map(lambda it: [it.index(y) for y in it if y > 0.99], result2)))

        temp2 = Template("""
                {
                  "markerSize": 0,
                  "name": "${getName2}",
                  "solutionList": ${getSolutionList2},
                  "xTitle": "Width of recovery path",
                  "yTitle": "CDF",
                  "xLimit": [0, ${max2}],
                  "yLimit": [0, 1],
                  "x": ${getX2},
                  "y": ${getY2}
                }""")

        string2 = temp2.substitute(getName2=getName2(d, n, p, q, k), getSolutionList2=getSolutionList2(nsdList),
                                   max2=max2, getX2=getX2(max2), getY2=getY2(result2, max2))
        self.children.append(string2)


        max3 = 0
        result3 = []

        for idx in range(len(nList)):
            nsd = nList[idx]
            s = 0
            summ = ReducibleLazyEvaluation(lambda x: 0)
            y = flatMap(flatMap(topoRange,
                                lambda topoIdx: parseLog('dist/' + id(n, topoIdx, q, k, p, d, nsd, 'Online') + '.txt')),
                        lambda it: flatMap(it.majorPaths, lambda it: map(lambda it: it.width, it.recoveryPaths)))

            for k, v in sorted(list(map(lambda k, v: (k, len(v)), list(groupby(y, lambda it: it)))),
                               key=lambda it: it[0]):
                max3 = max(max3, k)
                s += v
                for i in range(k, 1001):
                    summ[i] = s

            res = list(map(lambda it: summ[it] / s, list(range(max3 + 1))))
            result3.append(res)

        temp3 = Template("""
        {
          "markerSize": 0,
          "name": "${getName3}",
          "solutionList": ${getSolutionList3},
          "xTitle": "# recovery paths per major path",
          "yTitle": "CDF",
          "xLimit": [0, ${max3}],
          "yLimit": [0, 1],
          "x": ${getX3},
          "y": ${getY3}
        }""")

        string3 = temp3.substitute(getName3=getName3(d, n, p, q, k), getSolutionList3=getSolutionList3(nsdList),
                                   max3=max3, getX3=getX3(max3), getY3=getY3(result3, max3))
        self.children.append(string3)


    def rp2Nsd(self):

        def getName1(d, n, p, q, k):
            return "a2-rp-throughput-{}-{}-{}-{}-{}-nsd".format(d, n, p, q, k).replace('.', '')

        def getSolutionList1(names):
            return list(map(lambda it: ' {} '.format(self.nameMapping[it]) if it in self.nameMapping else it, names))

        def getX1(nsdList):
            return nsdList

        def getY1(results1):
            return results1

        def getName2(d, n, p, q, k):
            return "a2-rp-succ-pairs-{}-{}-{}-{}-{}-nsd".format(d, n, p, q, k).replace('.', '')

        def getSolutionList2(names):
            return list(map(lambda it: ' {} '.format(self.nameMapping[it]) if it in self.nameMapping else it, names))

        def getX2(nsdList):
            return nsdList

        def getY2(results2):
            return results2

        d, n, p, q, k, nsd = referenceSetting
        names  = ['Online', 'Online-R']

        results1 = []
        for name in names:
            result = []
            for nsd in sorted(nsdList):
                rList = flatMap(topoRange, lambda topoIdx: parseLog('dist/' + id(n, topoIdx, q, k, p, d, nsd, name) + '.txt'))
                res = list(map(lambda it: sum(it.majorPaths.succ), rList))
                result.append(sum(res)/len(res))
            results1.append(result)

        temp1 = Template("""
        {
          "name": "${getName1}",
          "solutionList": ${getSolutionList1},
          "xTitle": "# S-D pairs in one time slot",
          "yTitle": "Throughput (eps)",
          "x": ${getX1},
          "y": ${getY1}
        }""")

        string1 = temp1.substitute(getName1=getName1(d, n, p, q, k), getSolutionList1=getSolutionList1(nsdList), getX1=getX1(nsdList), getY1=getY1(results1))
        self.children.append(string1)

        results2 = []
        for name in names:
            result = []
            for nsd in sorted(nsdList):
                rList = flatMap(topoRange,
                                lambda topoIdx: parseLog('dist/' + id(n, topoIdx, q, k, p, d, nsd, name) + '.txt'))
                res = list(map(lambda it: len([i for i in it.majorPaths if i.succ > 0]), rList))
                result.append(sum(res)/len(res))
            results2.append(result)

        temp2 = Template("""
                {
                  "name": "${getName2}",
                  "solutionList": ${getSolutionList2},
                  "xTitle": "# S-D pairs in one time slot",
                  "yTitle": "Success S-D pairs",
                  "x": ${getX2},
                  "y": ${getY2}
                }""")

        string2 = temp2.substitute(getName2=getName2(d, n, p, q, k), getSolutionList2=getSolutionList2(nsdList),
                                   getX2=getX2(nsdList), getY2=getY2(results2))
        self.children.append(string2)


    def rp2N(self):

        def getName1(d, n, p, q, k):
            return "a2-rp-throughput-{}-{}-{}-{}-{}-nsd".format(d, n, p, q, k).replace('.', '')

        def getSolutionList1(names):
            return list(map(lambda it: ' {} '.format(self.nameMapping[it]) if it in self.nameMapping else it, names))

        def getX1(nList):
            return nList

        def getY1(results1):
            return results1

        def getTicksAndLabels1(nList):
            return [sorted(nList), list(map(lambda it: ' {} '.format(it), sorted(nList)))]

        def getName2(d, n, p, q, k):
            return "a2-rp-succ-pairs-{}-{}-{}-{}-{}-nsd".format(d, n, p, q, k).replace('.', '')

        def getSolutionList2(names):
            return list(map(lambda it: ' {} '.format(self.nameMapping[it]) if it in self.nameMapping else it, names))

        def getX2(nList):
            return nList

        def getY2(results2):
            return results2

        def getTicksAndLabels2(nList):
            return getTicksAndLabels1(nList)

        d, n, p, q, k, nsd = referenceSetting
        names = ['Online', 'Online-R']

        results1 = []
        for name in names:
            result = []
            for nsd in sorted(nsdList):
                rList = flatMap(topoRange,
                                lambda topoIdx: parseLog('dist/' + id(n, topoIdx, q, k, p, d, nsd, name) + '.txt'))
                res = list(map(lambda it: sum(it.majorPaths.succ), rList))
                result.append(sum(res) / len(res))
            results1.append(result)

        temp1 = Template("""
        {
          "markerSize": 0,
          "name": "${getName1}",
          "solutionList": ${getSolutionList1},
          "xTitle": "|V|",
          "xLog" : true,
          "xTicks&Labels": {getTicksAndLabels1},
          "yTitle": "Throughput (eps)",
          "x": ${getX1},
          "y": ${getY1}
        }""")

        string1 = temp1.substitute(getName1=getName1(d, n, p, q, k), getSolutionList1=getSolutionList1(nsdList),
                                   getTicksAndLabels1=getTicksAndLabels1(nList), getX1=getX1(nList), getY1=getY1(results1))
        self.children.append(string1)

        results2 = []
        for name in names:
            result = []
            for nsd in sorted(nsdList):
                rList = flatMap(topoRange,
                                lambda topoIdx: parseLog('dist/' + id(n, topoIdx, q, k, p, d, nsd, name) + '.txt'))
                res = list(map(lambda it: len([i for i in it.majorPaths if i.succ > 0]), rList))
                result.append(sum(res) / len(res))
            results2.append(result)

        temp2 = Template("""
                {
                  "markerSize": 0,
                  "name": "${getName2}",
                  "solutionList": ${getSolutionList2},
                  "xTitle": "|V|",
                  "xLog" : true,
                  "xTicks&Labels": {getTicksAndLabels2},
                  "yTitle": "Success S-D pairs",
                  "x": ${getX2},
                  "y": ${getY2}
                }""")

        string2 = temp2.substitute(getName2=getName2(d, n, p, q, k), getSolutionList2=getSolutionList2(nList),
                                   getTicksAndLabels2=getTicksAndLabels2(nList), getX2=getX2(nList), getY2=getY2(results2))
        self.children.append(string2)





    def plot(self):
        file = open('../plot/last-plot-data.json', mode='w', encoding='utf-8')
        str=f"""
{
  "type": "line",
  "figWidth": 600,
  "figHeight": 350,
  "usetex": false,

  "legendLoc": "best",
  "legendColumn": 1,

  "markerSize": 8,
  "lineWidth": 1,

  "xLog": false,
  "yLog": false,
  "xGrid": false,
  "yGrid": false,

  "xFontSize": 22,
  "xTickRotate": false,
  "yFontSize": 22,
  "legendFontSize": 14,
  "output": true,
  "show": false,

  "children": {self.chidren}
}
"""
        file.write(str)
        file.close()
    def throughputCdf(self):
        nameMapping = {"SL": "SLMP", "Online": "Q-CAST", "Greedy_H": "Greedy"}
        names=["Online", "SL", "Greedy_H", "CR", "BotCap", "SumDist", "MultiMetric"]
        for i in range (1,4):
            d, n, p, q, k, nsd = referenceSetting
            if i ==2:
                p=0.3
            elif i== 3:
                n=400
            max_item = 0
            def f(name):
                s = 0
                sum1 = ReducibleLazyEvaluation(initilizer=lambda x: 0)
                tmp1 = list(flat_map(lambda topoIdx: parseLog( "dist/" +id(n, topoIdx, q, k, p, d, nsd, name)+".txt" ),self,topoRange ))
                tmp2 = list(map( lambda x :sum(x.majorPaths.succ),tmp1))
                tmp3 = groupby_dict(tmp2,lambda x:x)
                tmp4 = list(map(lambda k: (k,len(tmp4[k])), tmp3))
                tmp5 = sorted(tmp4,key=lambda x:x[0])
                for k,v in tmp5:
                    max_item = max(max_item,k)
                    s+=v
                    for i in range(k,1001):
                        sum1[i] = s
                return list(map( lambda  x:sum1[x]/s ,[i for i in range(0,max_item+1)]))

            result = list(map(lambda name: f(name) ,names))
            y_string=[]
            for item in result:
                if len(item)<max_item+1:
                    y_string.append(item+ [1.0]*(max_item+1-len(item)))
                else:
                    y_string.append(item)
            tmp_str = f"""
        {{
          "markerSize": 0,
          "name":  {listtoString(["throughput-cdf-",f"{d}-",f"{n}-",f"{p}",f"{q}-",f"{k}-",f"{nsd}"],"").replace(".","")}
          "solutionList": {  listtoString(list(map( lambda name: f"{nameMapping[name]}",names)))}
          "xTitle": "Throughput (eps)",
          "xTicks&Labels":{  listtoString( list(map(lambda x: x[0],list(divide_chunks([i for i in range(0,max_item+1)],5))) ))},

          "xLimit": [0, {max_item}],
          "yLimit": [0, 1],
          "x": {[i for i in range(0,max_item+1)]}
          "y": {y_string}
        }}
"""
            self.chidren.append(tmp_str)
        def throughPutNsd():
            d, n, p, q, k, nsd = referenceSetting
            def f_Nsd_1(name):

                return( list(map(  lambda nsd:
                                 statistics.mean(list(map(lambda x : sum([ path.succ for path in x.majorPaths]) ,
                                                       # rstlist
                                                       list( flat_map( lambda topoIdx : parseLog("dist"+id(n, topoIdx, q, k, p, d, nsd, name)+"txt") ,  topoRange)  ))))

                        ,nsdList)))
            def f_Nsd_2(name):

                return( list(map(  lambda nsd:
                                 statistics.mean(list(map(lambda x : sum([ 1 for path in x.majorPaths  if path.succ>0]) ,
                                                       # rstlist
                                                       list( flat_map( lambda topoIdx : parseLog("dist"+id(n, topoIdx, q, k, p, d, nsd, name)+"txt") ,  topoRange)  ))))

                        ,nsdList)))

            def f_Nsd_3(name):
                return (list(map(lambda nsd:
                                 statistics.mean(list(map(lambda x:      len( set ( list( map(lambda path: [path[0],path[-1]] ,[path for path in x.majorPaths if path.succ>0]))))  ,
                                                          # rstlist
                                                          list(flat_map(lambda topoIdx: parseLog(
                                                              "dist" + id(n, topoIdx, q, k, p, d, nsd, name) + "txt"),
                                                                        topoRange)))))

                                 ,nsdList)))

            result = list(map(lambda name: f_Nsd_1(name), names))
            result2 = list(map(lambda name: f_Nsd_2(name), names))
            result3 = list(map(lambda name: f_Nsd_3(name), names))
            tmp_str1 = f"""
                    {{
                      "name":  {listtoString(["throughput-cdf-", f"{d}-", f"{n}-", f"{p}", f"{q}-", f"{k}-", f"{nsd}"], "").replace(".", "")}
                      "solutionList": {listtoString(list(map(lambda name: f"{nameMapping[name]}", names)))}
                      "xTitle": "# S-D pairs in one time slot",
                      "yTitle": "Throughput (eps)",,
                      "x": {nsdList}
                      "y": {result}
                    }}
            """
            tmp_str2 = f"""
                    {{
                      "name":  {listtoString(["throughput-cdf-", f"{d}-", f"{n}-", f"{p}", f"{q}-", f"{k}-", f"{nsd}"], "").replace(".", "")}
                      "solutionList": {listtoString(list(map(lambda name: f"{nameMapping[name]}", names)))}
                      "xTitle": "# S-D pairs in one time slot",
                      "yTitle": "Throughput (eps)",,
                      "x": {nsdList}
                      "y": {result2}
                    }}
            """
            tmp_str3 = f"""
                    {{
                      "name":  {listtoString(["throughput-cdf-", f"{d}-", f"{n}-", f"{p}", f"{q}-", f"{k}-", f"{nsd}"], "").replace(".", "")}
                      "solutionList": {listtoString(list(map(lambda name: f"{nameMapping[name]}", names)))}
                      "xTitle": "# S-D pairs in one time slot",
                      "yTitle": "Throughput (eps)",,
                      "x": {nsdList}
                      "y": {result3}
                    }}
            """
            self.chidren.append([tmp_str1,tmp_str2,tmp_str3])
        def throughPutD():
            d, n, p, q, k, nsd = referenceSetting
            def f_D_1(name):

                return( list(map(  lambda nsd:
                                 statistics.mean(list(map(lambda x : sum([ path.succ for path in x.majorPaths]) ,
                                                       # rstlist
                                                       list( flat_map( lambda topoIdx : parseLog("dist"+id(n, topoIdx, q, k, p, d, nsd, name)+"txt") ,  topoRange)  ))))

                        ,dList)))
            def f_D_2(name):

                return( list(map(  lambda nsd:
                                 statistics.mean(list(map(lambda x : sum([ 1 for path in x.majorPaths  if path.succ>0]) ,
                                                       # rstlist
                                                       list( flat_map( lambda topoIdx : parseLog("dist"+id(n, topoIdx, q, k, p, d, nsd, name)+"txt") ,  topoRange)  ))))

                        ,dList)))

            def f_D_3(name):
                return (list(map(lambda nsd:
                                 statistics.mean(list(map(lambda x:      len( set ( list( map(lambda path: [path[0],path[-1]] ,[path for path in x.majorPaths if path.succ>0]))))  ,
                                                          # rstlist
                                                          list(flat_map(lambda topoIdx: parseLog(
                                                              "dist" + id(n, topoIdx, q, k, p, d, nsd, name) + "txt"),
                                                                        topoRange)))))

                                 ,dList)))

            result = list(map(lambda name: f_D_1(name), names))
            result2 = list(map(lambda name: f_D_2(name), names))
            result3 = list(map(lambda name: f_D_3(name), names))
            tmp_str1 = f"""
                    {{
                      "name":  {listtoString(["throughput-d-", f"{d}-", f"{n}-", f"{p}", f"{q}-", f"{k}-", f"{nsd}"], "").replace(".", "")}
                      "solutionList": {listtoString(list(map(lambda name: f"{nameMapping[name]}", names)))}
                      "xTitle": "Average node degree",
                      "yTitle": "Throughput (eps)",,
                      "x": {dList}
                      "y": {result}
                    }}
            """
            tmp_str2 = f"""
                    {{
                      "name":  {listtoString(["throughput-d-", f"{d}-", f"{n}-", f"{p}", f"{q}-", f"{k}-", f"{nsd}"], "").replace(".", "")}
                      "solutionList": {listtoString(list(map(lambda name: f"{nameMapping[name]}", names)))}
                      "xTitle": "Average node degree",
                      "yTitle": "Throughput (eps)",,
                      "x": {dList}
                      "y": {result2}
                    }}
            """
            tmp_str3 = f"""
                    {{
                      "name":  {listtoString(["throughput-d-", f"{d}-", f"{n}-", f"{p}", f"{q}-", f"{k}-", f"{nsd}"], "").replace(".", "")}
                      "solutionList": {listtoString(list(map(lambda name: f"{nameMapping[name]}", names)))}
                      "xTitle": "Average node degree",
                      "yTitle": "Throughput (eps)",,
                      "x": {dList}
                      "y": {result3}
                    }}
            """
            self.chidren.append([tmp_str1,tmp_str2,tmp_str3])

        def throughPutP():
            d, n, p, q, k, nsd = referenceSetting

            def f_P_1(name):
                return (list(map(lambda nsd:
                                 statistics.mean(list(map(lambda x: sum([path.succ for path in x.majorPaths]),
                                                          # rstlist
                                                          list(flat_map(lambda topoIdx: parseLog(
                                                              "dist" + id(n, topoIdx, q, k, p, d, nsd,
                                                                          name) + "txt"), topoRange)))))

                                 , pList)))

            def f_P_2(name):
                return (list(map(lambda nsd:
                                 statistics.mean(
                                     list(map(lambda x: sum([1 for path in x.majorPaths if path.succ > 0]),
                                              # rstlist
                                              list(flat_map(lambda topoIdx: parseLog(
                                                  "dist" + id(n, topoIdx, q, k, p, d, nsd, name) + "txt"),
                                                            topoRange)))))

                                 , pList)))

            def f_P_3(name):
                return (list(map(lambda nsd:
                                 statistics.mean(list(map(lambda x: len(set(list(
                                     map(lambda path: [path[0], path[-1]],
                                         [path for path in x.majorPaths if path.succ > 0])))),
                                                          # rstlist
                                                          list(flat_map(lambda topoIdx: parseLog(
                                                              "dist" + id(n, topoIdx, q, k, p, d, nsd,
                                                                          name) + "txt"),
                                                                        topoRange)))))

                                 , pList)))

            result = list(map(lambda name: f_P_1(name), names))
            result2 = list(map(lambda name: f_P_2(name), names))
            result3 = list(map(lambda name: f_P_3(name), names))
            tmp_str1 = f"""
                    {{
                      "name":  {listtoString(["throughput-d-", f"{d}-", f"{n}-", f"{p}", f"{q}-", f"{k}-", f"{nsd}"], "").replace(".", "")}
                      "solutionList": {listtoString(list(map(lambda name: f"{nameMapping[name]}", names)))}
                      "xTitle": "Average node degree",
                      "yTitle": "Throughput (eps)",,
                      "x": {pList}
                      "y": {result}
                    }}
            """
            tmp_str2 = f"""
                    {{
                      "name":  {listtoString(["throughput-d-", f"{d}-", f"{n}-", f"{p}", f"{q}-", f"{k}-", f"{nsd}"], "").replace(".", "")}
                      "solutionList": {listtoString(list(map(lambda name: f"{nameMapping[name]}", names)))}
                      "xTitle": "Average node degree",
                      "yTitle": "Throughput (eps)",,
                      "x": {pList}
                      "y": {result2}
                    }}
            """
            tmp_str3 = f"""
                    {{
                      "name":  {listtoString(["throughput-d-", f"{d}-", f"{n}-", f"{p}", f"{q}-", f"{k}-", f"{nsd}"], "").replace(".", "")}
                      "solutionList": {listtoString(list(map(lambda name: f"{nameMapping[name]}", names)))}
                      "xTitle": "Average node degree",
                      "yTitle": "Throughput (eps)",,
                      "x": {pList}
                      "y": {result3}
                    }}
            """
            self.chidren.append([tmp_str1, tmp_str2, tmp_str3])
        def throughPutN():
            d, n, p, q, k, nsd = referenceSetting
            def f_N_1(name):

                return( list(map(  lambda nsd:
                                 statistics.mean(list(map(lambda x : sum([ path.succ for path in x.majorPaths]) ,
                                                       # rstlist
                                                       list( flat_map( lambda topoIdx : parseLog("dist"+id(n, topoIdx, q, k, p, d, nsd, name)+"txt") ,  topoRange)  ))))

                        ,nList)))
            def f_N_2(name):

                return( list(map(  lambda nsd:
                                 statistics.mean(list(map(lambda x : sum([ 1 for path in x.majorPaths  if path.succ>0]) ,
                                                       # rstlist
                                                       list( flat_map( lambda topoIdx : parseLog("dist"+id(n, topoIdx, q, k, p, d, nsd, name)+"txt") ,  topoRange)  ))))

                        ,nList)))

            def f_N_3(name):
                return (list(map(lambda nsd:
                                 statistics.mean(list(map(lambda x:      len( set ( list( map(lambda path: [path[0],path[-1]] ,[path for path in x.majorPaths if path.succ>0]))))  ,
                                                          # rstlist
                                                          list(flat_map(lambda topoIdx: parseLog(
                                                              "dist" + id(n, topoIdx, q, k, p, d, nsd, name) + "txt"),
                                                                        topoRange)))))

                                 ,nList)))

            result = list(map(lambda name: f_N_1(name), names))
            result2 = list(map(lambda name: f_N_2(name), names))
            result3 = list(map(lambda name: f_N_3(name), names))
            tmp_str1 = f"""
                    {{
                      "name":  {listtoString(["throughput-d-", f"{d}-", f"{n}-", f"{p}", f"{q}-", f"{k}-", f"{nsd}"], "").replace(".", "")}
                      "solutionList": {listtoString(list(map(lambda name: f"{nameMapping[name]}", names)))}
                      "xTitle": "Average node degree",
                      "yTitle": "Throughput (eps)",,
                      "x": {nList}
                      "y": {result}
                    }}
            """
            tmp_str2 = f"""
                    {{
                      "name":  {listtoString(["throughput-d-", f"{d}-", f"{n}-", f"{p}", f"{q}-", f"{k}-", f"{nsd}"], "").replace(".", "")}
                      "solutionList": {listtoString(list(map(lambda name: f"{nameMapping[name]}", names)))}
                      "xTitle": "Average node degree",
                      "yTitle": "Throughput (eps)",,
                      "x": {nList}
                      "y": {result2}
                    }}
            """
            tmp_str3 = f"""
                    {{
                      "name":  {listtoString(["throughput-d-", f"{d}-", f"{n}-", f"{p}", f"{q}-", f"{k}-", f"{nsd}"], "").replace(".", "")}
                      "solutionList": {listtoString(list(map(lambda name: f"{nameMapping[name]}", names)))}
                      "xTitle": "Average node degree",
                      "yTitle": "Throughput (eps)",,
                      "x": {nList}
                      "y": {result3}
                    }}
            """
            self.chidren.append([tmp_str1,tmp_str2,tmp_str3])

        def throughPutQ():
            d, n, p, q, k, nsd = referenceSetting
            def f_Q_1(name):

                return( list(map(  lambda nsd:
                                 statistics.mean(list(map(lambda x : sum([ path.succ for path in x.majorPaths]) ,
                                                       # rstlist
                                                       list( flat_map( lambda topoIdx : parseLog("dist"+id(n, topoIdx, q, k, p, d, nsd, name)+"txt") ,  topoRange)  ))))

                        ,qList)))
            def f_Q_2(name):

                return( list(map(  lambda nsd:
                                 statistics.mean(list(map(lambda x : sum([ 1 for path in x.majorPaths  if path.succ>0]) ,
                                                       # rstlist
                                                       list( flat_map( lambda topoIdx : parseLog("dist"+id(n, topoIdx, q, k, p, d, nsd, name)+"txt") ,  topoRange)  ))))

                        ,qList)))

            def f_Q_3(name):
                return (list(map(lambda nsd:
                                 statistics.mean(list(map(lambda x:      len( set ( list( map(lambda path: [path[0],path[-1]] ,[path for path in x.majorPaths if path.succ>0]))))  ,
                                                          # rstlist
                                                          list(flat_map(lambda topoIdx: parseLog(
                                                              "dist" + id(n, topoIdx, q, k, p, d, nsd, name) + "txt"),
                                                                        topoRange)))))

                                 ,qList)))

            result = list(map(lambda name: f_Q_1(name), names))
            result2 = list(map(lambda name: f_Q_2(name), names))
            result3 = list(map(lambda name: f_Q_3(name), names))
            tmp_str1 = f"""
                    {{
                      "name":  {listtoString(["throughput-d-", f"{d}-", f"{n}-", f"{p}", f"{q}-", f"{k}-", f"{nsd}"], "").replace(".", "")}
                      "solutionList": {listtoString(list(map(lambda name: f"{nameMapping[name]}", names)))}
                      "xTitle": "Average node degree",
                      "yTitle": "Throughput (eps)",,
                      "x": {qList}
                      "y": {result}
                    }}
            """
            tmp_str2 = f"""
                    {{
                      "name":  {listtoString(["throughput-d-", f"{d}-", f"{n}-", f"{p}", f"{q}-", f"{k}-", f"{nsd}"], "").replace(".", "")}
                      "solutionList": {listtoString(list(map(lambda name: f"{nameMapping[name]}", names)))}
                      "xTitle": "Average node degree",
                      "yTitle": "Throughput (eps)",,
                      "x": {qList}
                      "y": {result2}
                    }}
            """
            tmp_str3 = f"""
                    {{
                      "name":  {listtoString(["throughput-d-", f"{d}-", f"{n}-", f"{p}", f"{q}-", f"{k}-", f"{nsd}"], "").replace(".", "")}
                      "solutionList": {listtoString(list(map(lambda name: f"{nameMapping[name]}", names)))}
                      "xTitle": "Average node degree",
                      "yTitle": "Throughput (eps)",,
                      "x": {qList}
                      "y": {result3}
                    }}
            """
            self.chidren.append([tmp_str1,tmp_str2,tmp_str3])
        def throughPutK():
            d, n, p, q, k, nsd = referenceSetting
            def f_K_1(name):

                return( list(map(  lambda nsd:
                                 statistics.mean(list(map(lambda x : sum([ path.succ for path in x.majorPaths]) ,
                                                       # rstlist
                                                       list( flat_map( lambda topoIdx : parseLog("dist"+id(n, topoIdx, q, k, p, d, nsd, name)+"txt") ,  topoRange)  ))))

                        ,kList)))
            def f_K_2(name):

                return( list(map(  lambda nsd:
                                 statistics.mean(list(map(lambda x : sum([ 1 for path in x.majorPaths  if path.succ>0]) ,
                                                       # rstlist
                                                       list( flat_map( lambda topoIdx : parseLog("dist"+id(n, topoIdx, q, k, p, d, nsd, name)+"txt") ,  topoRange)  ))))

                        ,kList)))

            def f_K_3(name):
                return (list(map(lambda nsd:
                                 statistics.mean(list(map(lambda x:      len( set ( list( map(lambda path: [path[0],path[-1]] ,[path for path in x.majorPaths if path.succ>0]))))  ,
                                                          # rstlist
                                                          list(flat_map(lambda topoIdx: parseLog(
                                                              "dist" + id(n, topoIdx, q, k, p, d, nsd, name) + "txt"),
                                                                        topoRange)))))

                                 ,kList)))

            result = list(map(lambda name: f_K_1(name), names))
            result2 = list(map(lambda name: f_K_2(name), names))
            result3 = list(map(lambda name: f_K_3(name), names))
            tmp_str1 = f"""
                    {{
                      "name":  {listtoString(["throughput-d-", f"{d}-", f"{n}-", f"{p}", f"{q}-", f"{k}-", f"{nsd}"], "").replace(".", "")}
                      "solutionList": {listtoString(list(map(lambda name: f"{nameMapping[name]}", names)))}
                      "xTitle": "Average node degree",
                      "yTitle": "Throughput (eps)",,
                      "x": {kList}
                      "y": {result}
                    }}
            """
            tmp_str2 = f"""
                    {{
                      "name":  {listtoString(["throughput-d-", f"{d}-", f"{n}-", f"{p}", f"{q}-", f"{k}-", f"{nsd}"], "").replace(".", "")}
                      "solutionList": {listtoString(list(map(lambda name: f"{nameMapping[name]}", names)))}
                      "xTitle": "Average node degree",
                      "yTitle": "Throughput (eps)",,
                      "x": {kList}
                      "y": {result2}
                    }}
            """
            tmp_str3 = f"""
                    {{
                      "name":  {listtoString(["throughput-d-", f"{d}-", f"{n}-", f"{p}", f"{q}-", f"{k}-", f"{nsd}"], "").replace(".", "")}
                      "solutionList": {listtoString(list(map(lambda name: f"{nameMapping[name]}", names)))}
                      "xTitle": "Average node degree",
                      "yTitle": "Throughput (eps)",,
                      "x": {kList}
                      "y": {result3}
                    }}
            """
            self.chidren.append([tmp_str1,tmp_str2,tmp_str3])

    def efficiency(self):
        for mode in range(1, 4):
            if mode == 1:
                names = ["Online", "SL", "Greedy_H", "CR"]
            elif mode == 2:
                names = ["Online", "SL", "Greedy_H"]
            else:
                names = ["Online", "Online-R", "CR", "CR-R"]
            d, n, p, q, k, nsd = referenceSetting
            maxi = 0
            channels = []
            for name in names:
                # TODO: reducible lazy evaluation with just two parameters
                result = {}
                records = []
                for topoIdx in topoRange:
                    records = parseLog("dist/" + id(n, topoIdx, q, k, p, d, nsd, name) + ".txt")
                prov_map = {}
                tmp = []
                for record in records:
                    first = sum(record.majorPaths.succ)
                    second = 0
                    for majorPath in records:
                        second += majorPath.width * (len(majorPath.path) - 1)
                        for recoveryPath in majorPath.recoveryPaths:
                            second += recoveryPath.width * (len(recoveryPath.path) - 1)
                    tmp.append((first, second))
                for pair in tmp:
                    if pair[0] not in prov_map:
                        prov_map[pair[0]] = [pair]
                    else:
                        prov_map[pair[0]].append(pair)
                for pair in prov_map:
                    maxi = max(maxi, pair[0])
                    if result[pair[0]] not in result:
                        result[pair[0]] = [np.mean(float(x) for x in pair[1])]
                    else:
                        result[pair[0]].append(np.mean(float(x) for x in pair[1]))
            # TODO: Check out this part
            for i in range(maxi + 1):
                channels[i] = result[i]
            nameString = "channels-throughput-{}-{}-{}-{}-{}-{}{}".format((d, n, p, q, k, nsd,
                                                                                            "" if mode == 1 else "-noA1"
                                                                                            if mode == 2 else "-rp")).replace(".", "")
            # TODO: Can we separate them with commas?
            solutionList = []
            for name in names:
                if self.nameMapping[name] is not None: solutionList.append(self.nameMapping[name])
                else: solutionList.append(name)
            solutionListString = ','.join(solutionList)
            x = [i for i in range(maxi+1)]
            # This part takes the first element of each sublist of 5 positions and makes a list out of it
            xTicksAndLabels = '.'.join([sublist[0] for sublist in [x[i:i + 5] for i in range(0, len(x), 5)]])
            provY = []
            for l in channels:
                if len(l) < maxi + 1: provY.append(l + [None]*int(maxi + 1 - len(l)) )
                else: provY.append(l)
            y = ','.join(provY)
            self.chidren.append(
                f"""
                "name": {nameString},
                "solutionList": {solutionListString},
                "xTitle": "Throughput (eps)",
                "yTitle": "# occupied channels",
                "x": {x},
                "xTicks&Labels": {xTicksAndLabels},
                "y": {provY}
                """
            )

    def fairness(self):
        d, n, p, q, k, nsd = referenceSetting
        maxi = 0
        result = []
        for name in self.names:
            s = 0
            # TODO: reducible lazy evaluation with just two parameters
            sum_ = {}
            topoRangeFlatMap = {}
            records = []
            for topoIdx in topoRange:
                records = parseLog("dist/" + id(n, topoIdx, q, k, p, d, nsd, name) + ".txt")
            prov1 = {}
            i = 0
            for record in records:
                majorPath = record.majorPaths
                if (majorPath.path[0], majorPath.path[-1]) not in prov1:
                    prov1[(majorPath.path[0], majorPath.path[-1])] = [majorPath]
                else:
                    prov1[(majorPath.path[0], majorPath.path[-1])].append(majorPath)
                if topoRange[i] not in topoRangeFlatMap:
                    topoRangeFlatMap[topoRange[i]] = [sum(majorPath.width for majorPath in prov1.values())]
                else:
                    topoRangeFlatMap[topoRange[i]].append(sum(majorPath.width for majorPath in prov1.values()))
                i += 1
            final = {}
            for topoIdx in topoRangeFlatMap:
                final[topoIdx] = len(topoRangeFlatMap[topoIdx])
            srtd = sorted(final)
            for k in srtd:
                v = srtd[k]
                maxi = max(maxi, k)
                s += v
                for i in range(k, 1001):
                    sum_[i] = s
            for it in range(0, maxi+1):
                result[it] = sum[it]/float(s)
        markerSize = "markerSize: 0"
        name = "mp-cdf-{}-{}-{}-{}-{}-{}".format((d, n, p, q, k, nsd)).replace(".", "")
        # TODO: Can we separate them with commas?
        solutionList = []
        for name in names:
            if self.nameMapping[name] is not None:
                solutionList.append(self.nameMapping[name])
            else:
                solutionList.append(name)
        solutionListString = ','.join(solutionList)
        x = [i for i in range(maxi + 1)]
        provY = []
        for l in channels:
            if len(l) < maxi + 1:
                provY.append(l + [None] * int(maxi + 1 - len(l)))
            else:
                provY.append(l)
        y = ','.join(provY)

        self.chidren.append(f"""
        {
          "markerSize": 0,
          "name": {name},
          "solutionList": {solutionListString},
          "xTitle": "Total width of allocated major paths",
          "yTitle": "CDF",
          "xLimit": [0, {str(maxi)}],
          "yLimit": [0, 1],
          "x": {x},
          "y": {provY}
        }""".strip())

    def recovery1(self):
        q, k, d = qList[0], kList[0], dList[0]
        for p in pList:
            for n in sorted(nList):
                deviation = 0
                if p != pList[0]: deviation += 1
                if n != nList[0]: deviation += 1
                if deviation >= 1: continue
                names = ["CR", "CR-R"]
            results = []
            for name in names:
                for nsd in sorted(nsdList):
                    rList = []
                    for topoIdx in topoRange:
                        rList.append(parseLog("dist/" + id(n, topoIdx, q, k, p, d, nsd, name) + ".txt"))
                    provList = []
                    for record in rList:
                        avg = np.mean(sum(float(majorPath.succ) for majorPath in record.majorPaths))
                        provList.append(avg)
                    results.append(provList)
            results2 = []
            for name in names:
                for nsd in sorted(nsdList):
                    rList = []
                    for topoIdx in topoRange:
                        rList.append(parseLog("dist/" + id(n, topoIdx, q, k, p, d, nsd, name) + ".txt"))
                    provList = []
                    for record in rList:
                        avg = np.mean(len([float(majorPath.succ) for majorPath in record.majorPaths if float(majorPath.succ) > 0]))
                        provList.append(avg)
                    results2.append(provList)

            nameString1 = "a1-rp-throughput-{}-{}-{}-{}-{}-nsd".format((d, n, p, q, k)).replace(".", "")
            nameString2 = "a1-rp-succ-pairs-{}-{}-{}-{}-{}-nsd".format((d, n, p, q, k)).replace(".", "")
            solutionList = []
            for name in names:
                if self.nameMapping[name] is not None:
                    solutionList.append(self.nameMapping[name])
                else:
                    solutionList.append(name)
            solutionListString = ','.join(solutionList)

            self.chidren.append([f"""{
            "name": {nameString1},
            "solutionList": {solutionList},
            "xTitle": "# S-D pairs in one time slot",
            "yTitle": "Throughput (eps)",
            "x": {configs.nsdList},
            "y": ${results}
          }""".strip(), f"""
          {
            "name": {nameString2}",
            "solutionList": {solutionListString},
            "xTitle": "# S-D pairs in one time slot",
            "yTitle": "# succ S-D pairs",
            "x": {configs.nsdList},
            "y": ${results2}
          }""".strip()])


if __name__ == '__main__':
    p = Plot()

    p.rp2Cdf_nsd()
    p.rp2Cdf_n()
    p.rp2N()
    p.rp2Nsd()

