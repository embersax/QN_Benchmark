from utils.utils import foldLeft
from string import Template


class AnalyticalPlot:

    def __init__(self):
        # created these variables to allow for passing different initial parameters for different fold functions
        self.t = None
        self.p = None

    def inFoldFunLine76(self, acc, element):
        return acc * (1 - self.t ** 2) + pow(self.p, 2 * element - 2) * 2 * self.p * self.t

    def inFoldFunLine96(self, acc, element):
        return self.inFoldFunLine76(acc, element)

    def inFoldFunLine103(self, acc, element):
        pk_1_1, pk_1_2 = acc
        return pk_1_1 * (1 - pow(self.t, 3)) + (
                pk_1_2 + pow(self.p, 3 * element - 3)) * 3 * self.p * self.t * self.t, pk_1_2 * (
                       3 * self.p * self.p * self.t + self.p * self.p * self.p) + pow(self.p,
                                                                                      3 * element - 3) * 3 * self.p * self.p * self.t

    def inFoldFunLine126(self, acc, element):
        return self.inFoldFunLine103(acc, element)

    def parallel(self):
        q, n = .95, 10
        Q = pow(q, n - 1)
        fileName = "../plot/last-plot-data.json"

        def getChildX1():
            return [i / 100 for i in range(101)]

        def getChildX2():
            return getChildX1()

        def getChildY1():
            return [[2 * pow((i / 100), 2) * q for i in range(101)],
                    list(map(lambda p: 2 * pow(p, 4) * q + 4 * pow(p, 2) * (1 - p) * q, [i / 100 for i in range(101)]))]

        def getChildY2():
            ele0 = list(map(lambda p: 2 * pow(p, n) * Q, getChildX2()))
            ele1 = []
            for p in getChildX2():
                self.p, self.t = p, 1 - p
                pn_1 = foldLeft(self.inFoldFunLine76, [i for i in range(2, n + 1)], 2 * self.p * self.t)
                app = pn_1 * Q + 2 * pow(p, 2 * n) * (Q ** 2 + 1 - pow(q, n - 1) * Q)
                ele1.append(app)
            return [ele0, ele1]

        def getChildX3():
            return getChildX1()

        def getChildY3():
            ele0 = list(map(lambda p: 3 * pow(p, n) * Q, getChildX3()))
            ele1 = []
            for p in getChildX3():
                self.p, self.t = p, 1 - p
                pn_1 = foldLeft(self.inFoldFunLine96, [i for i in range(2, n + 1)], 2 * self.p * self.t)
                app = pow(self.p, n) * Q + pn_1 * Q + 2 * pow(self.p, 2 * n) * (Q * Q + (1 - Q) * Q)
                ele1.append(app)
            ele2 = []
            for p in getChildX3():
                self.p, self.t = p, 1 - p
                pn_1, pn_2 = foldLeft(self.inFoldFunLine103, [i for i in range(2, n + 1)],
                                      (3 * self.p * self.t * self.t, 3 * self.p * self.p * self.t))
                pn_3 = pow(self.p, n * 3)
                app = pn_1 * Q + 2 * pn_2 * Q * Q + 3 * pn_3 * Q * Q * Q + 2 * 3 * pn_3 * Q * Q * (
                        1 - Q) + 3 * pn_3 * Q * pow((1 - Q), 2) + 2 * pn_2 * Q * (1 - Q)
                ele2.append(app)
            return [ele0, ele1, ele2]

        def getChildX4():
            return getChildX1()

        def getChildY4():
            ele = []
            for p in getChildX4():
                self.p, self.t = p, 1 - p
                pn_1, pn_2 = foldLeft(self.inFoldFunLine103, [i for i in range(2, n + 1)],
                                      (3 * self.p * self.t * self.t, 3 * self.p * self.p * self.t))
                pn_3 = pow(self.p, n * 3)
                app = pn_1, pn_2, pn_3

                ele.append(app)

            return [[item[0] for item in ele], [item[1] for item in ele], [item[2] for item in ele],
                    [sum(item) for item in ele]]

        def getChildX5():
            return [i for i in range(1, 11)]

        def getChildY5(p):
            self.p, self.t = p, 1 - p
            n = 10
            Q = pow(q, n - 1)
            ele0 = list(map(lambda n: pow(p, n) * Q, getChildX5()))

            ele1 = []
            for n in getChildX5():
                pn_1 = foldLeft(self.inFoldFunLine96, [i for i in range(2, n + 1)], 2 * self.p * self.t)
                app = pow(self.p, n) * Q + pn_1 * Q + 2 * pow(self.p, 2 * n) * (Q * Q + (1 - Q) * Q)
                ele1.append(app)

            ele2 = []
            for n in getChildX5():
                Q = pow(q, n-1)
                pn_1, pn_2 = foldLeft(self.inFoldFunLine103, [i for i in range(2, n + 1)],
                                      (3 * self.p * self.t * self.t, 3 * self.p * self.p * self.t))
                pn_3 = pow(self.p, n * 3)
                app = pn_1 * Q + 2 * pn_2 * Q * Q + 3 * pn_3 * Q * Q * Q + 2 * 3 * pn_3 * Q * Q * (1 - Q) + 3 * pn_3 * Q * pow((1 - Q), 2) + 2 * pn_2 * Q * (1 - Q)
                ele2.append(app)

            return [ele0, ele1, ele2]

        def getChildX6():
            return getChildX5()

        def getChildY6(p):
            return getChildY5(p)

        def getChildX7():
            return getChildX5()

        def getChildY7(p):
            return getChildY5(p)

        with open(fileName, mode='w') as file:
            temp = Template("""
{
  'type': "line",
  'figWidth': 600,
  'figHeight': 350,
  'usetex': False,
  'mainColors': ['#0072bc',
                '#d85119',
                '#edb021',
                '#7a8cbf',
                '#009d70',
                '#979797',
                '#53b2ea'],
  
  'legendLoc': 'best',
  'legendColumn': 1,
  
  'markerSize': 8,
  'lineWidth': 1,
  
  'xLog': False,
  'yLog': False,
  'xGrid': True,
  'yGrid': True,
  
  'xFontSize': 24,
  'xTickRotate': False,
  'yFontSize': 24,
  'legendFontSize': 18,
  'output': True,
  
  'children': [
    {
      'markerSize': 0,
      'name': 'E-p-2',
      'solutionList': ('Separate', 'Parallel'),
      'figTitle': "",
      'xTitle': 'P(one link success)',
      'yTitle': 'E(# entanglements)',
      'x': ${getChildX1},
      'y': ${getChildY1},
    },
    {
      'markerSize': 0,
      'name': 'E-p-10',
      'solutionList': ('Separate', 'Parallel'),
      'figTitle': "",
      'xTitle': 'P(one link success)',
      'yTitle': 'E(# entanglements)',
      'x': ${getChildX2},
      'y': ${getChildY2},
    },
    {
      'markerSize': 0,
      'name': 'E-p-10-3',
      'solutionList': ('-1-1-1-', '-1-2-', '-3-'),
      'figTitle': "",
      'xTitle': 'P(one link success)',
      'yTitle': 'E(# entanglements)',
      'x': ${getChildX3},
      'y': ${getChildY3},
    },
    {
      'markerSize': 0,
      'name': 'P-p-10',
      'solutionList': ('1 path', '2 paths', '3 paths', 'paths exist'),
      'figTitle': "",
      'xTitle': 'P(one link success)',
      'yTitle': 'P',
      'x': ${getChildX4},
      'y': ${getChildY4},
    },
    {
        'markerSize': 8,
        'name': 'E-hops-06',
        'solutionList': ('1-path', '2-path', '3-path'),
        'figTitle': "",
        'xTitle': 'Number of hops',
        'yTitle': 'EXT',
        'x': ${getChildX5},
        'y': ${getChildY5},
      }, {
        'markerSize': 8,
        'name': 'E-hops-08',
        'solutionList': ('1-path', '2-path', '3-path'),
        'figTitle': "",
        'xTitle': 'Number of hops',
        'yTitle': 'EXT',
        'x': ${getChildX6},
        'y': ${getChildY6},
      }, {
        'markerSize': 8,
        'name': 'E-hops-09',
        'solutionList': ('1-path', '2-path', '3-path'),
        'figTitle': "",
        'xTitle': 'Number of hops',
        'yTitle': 'EXT',
        'x': ${getChildX7},
        'y': ${getChildY7},
      }
  ]
}
""")
            string = temp.substitute(getChildX1=getChildX1(), getChildY1=getChildY1(), getChildX2=getChildX2(), getChildY2=getChildY2(), getChildX3=getChildX3(), getChildY3=getChildY3(), getChildX4=getChildX4(), getChildY4=getChildY4(), getChildX5=getChildX5(), getChildY5=getChildY5(.6), getChildX6=getChildX6(), getChildY6=getChildY6(.8), getChildX7=getChildX7(), getChildY7=getChildY7(.9))
            file.write(string)


if __name__ == '__main__':
    AnalyticalPlot().parallel()
