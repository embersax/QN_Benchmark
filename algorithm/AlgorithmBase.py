"""This class represents the base for the implementation of a quantum routing algorithm """
# You can check this link to study abstract class implementation in Python.
# https://stackoverflow.com/questions/7196376/python-abstractmethod-decorator
from topo import Topo
import abc


class Algorithm(metaclass=abc.ABCMeta):
    def __init__(self, topo):
        self.topo = topo
        self.name = ''
        self.settings = 'Simple'
        # TODO: Maybe we need a better name for the log writer.
        # Or maybe we can add a serial number to what we have, using the name of the algorithm.
        self.logWriter = open('log.txt', 'a')
        # This is a list of pairs of nodes: Pair<Node,Node>
        self.srcDstPairs = []
        self.established = []

    def work(self, pairs):
        assert self.topo.isClean()
        self.srcDstPairs.clear()
        self.srcDstPairs.extend(pairs)
        for p in pairs: self.logWriter.write("{}<->{}".format(p[0], p[1]))
        self.P2()
        self.tryEntanglement()
        self.P4()

        established = []
        for p in self.srcDstPairs:
            n1, n2 = p[0], p[1]
            established.append(((n1, n2), self.topo.getEstablishedEntanglements(n1, n2)))
        string = "[{}] Established:".format(self.settings)
        formatted_strings = ["[{}] Established:".format(self.settings)]
        for el in established:
            nodes, length = el[0], len(el[1])
            n1, n2 = nodes[0], nodes[1]
            # removed a set of () because we want individual variables and not tuple in the string formatting.
            # string.join("{}⟷{} × {}".format(n1.id, n2.id, length))
            formatted_strings.append("{}<->{} × {}".format(n1.id, n2.id, length))
        # string.join(' - {}'.format(self.name))
        formatted_strings.append(' - {}'.format(self.name))
        output = '  '.join(formatted_strings)
        print(output)
        self.topo.clearEntanglements()

        countNotEmpty = len(list(filter(lambda x: len(x[1]) > 0, established)))
        sumByLength = sum(len(p[1]) for p in established)
        return countNotEmpty, sumByLength

    # Tries an entanglement in each of the links of the topology
    def tryEntanglement(self):
        for link in self.topo.links:
            link.tryEntanglement()

    @abc.abstractmethod
    def prepare(self): return

    @abc.abstractmethod
    def P2(self): return

    @abc.abstractmethod
    def P4(self): return
