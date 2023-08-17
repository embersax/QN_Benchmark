import math
import numpy as np
from topo.Node import to, Edge
from topo import Topo
import abc
from PurificationAlg import KShortestPath
from algorithm.AlgorithmBase import Algorithm


from heapq import heappush, heappop
from itertools import count

from utils.utils import ReducibleLazyEvaluation
"""
Input:
    - network topology
    - SD pair i
    - number of entanglement paths K
    - minimum fidelity requirement, F*
    - Bell pair fidelity over each link Fl

Output:
    - set of entanglement paths
    - corresponding purification schem C = {{r ijl}}
"""

class EPP(Algorithm):

    def __init__(self, topo):
        super().__init__(topo)

        # (1) initiate the entanglement path iterative count: j
        self.j = 1
        # [done in init of link class] set weight of each link (l) to be (-ln(Fl))

        # minimum fidelity required for e2e entanglement
        self.minReqFid = 0.8


    def calc_phiU(self, N, k, piLinks):
        phi = 1
        for i in range(N):
            if i != k:
                phi *= piLinks[i]
        return phi

    def calc_phiV(self, N, piLinks):
        phi = 1
        for i in range(N):
            phi *= piLinks[i]

        return phi

    def calc_phiIK(self, N, piLinks, k):
        phi_ik = 1
        for i in range(k):
            if i == k:
                phi_ik = 0
            else:
                phi_ik *= (self.calc_phiU(N,k,piLinks) / piLinks[k])

        return phi_ik



    def calc_piOfLinks(self, rl):
        N = len(rl)
        piLinks = []
        for i in range(N):
            Fi = rl[i].n1.F
            pi = self.calc_pi(Fi)
            piLinks.append(pi)

        return piLinks

    def calc_pi(self, Fi):
        pi = 0.5 + 0.5 * math.sqrt(2 * Fi - 1)
        return pi

    def calc_U(self, N, piLinks):
        # initialize the U (probability for having odd number of bit flips
        U = 0

        for i in range(N):
            pi = piLinks[i]
            phi = self.calc_phiU(N, i, piLinks)

            U += (1 - pi) * phi

        return U

    def calc_V(self, N, piLinks, U):

        #pN = p of iteration N so it is the pi of last link
        pN = piLinks[N-1]
        phi = self.calc_phiV(N, piLinks)

        V = 2 * pN(1 - pN)*phi + math.pow(pN, 2) * U

        return V

    def calc_uDev(self, N, piLinks):
        uDev = 0
        for i in range(N):
            if i == N:
                break
            else:
                phi_ik = 0
                for k in range(N):
                    phi_ik = self.calc_phiIK(N, piLinks, k)
                uDev = phi_ik - N*self.calc_phiU(N,i, piLinks)

        return uDev

    def calc_vDev(self, N, piLinks, uDev):
        #for i in range(N):
            #if i == N-1:
        return 0.5





    def calc_largestFidelityDerivative(self, N, piLinks):

        largestDev = []
        devs = []

        for i in range(N):
            dpi_dFi = 1 / (4 * piLinks[i] - 2)

            # pN = p of iteration N so it is the pi of last link
            pN = piLinks[N-1]
            u_dev = self.calc_uDev(N, piLinks) * dpi_dFi
            v_dev = self.calc_vDev(N, piLinks) * dpi_dFi

            U = self.calc_U(N, piLinks)
            dF_dFi = -1 * (1 - U) * v_dev - (1 - self.calc_V(N, piLinks, U)) * u_dev
            devs.append(dF_dFi)
            largestDev.append(max(devs))

        return largestDev







# input: rl = path from purification scheme?
    def calc_E2EFidelity(self, rl):
        # N = length of path
        N = len(rl)
        # find pi of every link (bit flip probability)
        piLinks = self.calc_piOfLinks(rl)

        U = self.calc_U(N,piLinks)
        V = self.calc_V(N, piLinks, U)

        e2eFid = 1 + U*V - U -V

        return e2eFid

    def minFidelity(self, N):
        # if there are one or more hops in the path then calculate the minimum fidelty requirement
        if N >= 1:
            fMin = (pow((3 * N - 1), 2) + 1) / pow((3 * N), 2)
        else: # else it will be equal to the
            fMin = self.minReqFid
        return float(fMin)


    def calcNumOfSacrificialPairs(self):
        # dictionary with keys:link, value: T sarifical pairs
        reqSacPairs = {}

        for link in self.topo.links:

            # fidelity of current link iteration
            Fi = self.topo.nodes[link.n1.id].F
            # initial fidelity with nno sacrificial pairs
            Ft = Fi
            # T number of sacrificial bell pairs
            T = 0

            while Ft < self.minReqFid:
                # if the fidelity of the link is below the minimum required fidelity threshold add a sacrificial pair
                T += 1
                Ft = (Fi * Ft) / ((Fi * Ft) + (1 - Fi)*(1 - Ft))

            # update reqSacPairs
            reqSacPairs[link] = T
        # return the number reqSacPairs and corresponding links
        return reqSacPairs

    def work(self):

        A = set()
        C = set()

        index = 0

        ksp = KShortestPath

        kLengths, kPaths = []
        for n1, n2 in self.nodes:
            kLengths, kPaths = ksp.k_shortest_paths(self.topo, n1, n2)

        for k in kPaths:
            N = kLengths[index]
            index += 1
            fMin = self.minFidelity(N)

            reqSacPairs = self.calcNumOfSacrificialPairs()

            #update the purification scheme: add dictionary with keys:link value: required number of sacrificial pairs
            A.add(reqSacPairs)

            while A:
                # 1st time of A set
                rl = A.pop()
                e2eFid = self.calc_E2EFidelity(rl)

                if e2eFid >= self.minReqFid:
                    rijl = rl
                    C.add(rijl)
                    self.j += 1
                else:
                    piLinks = self.calc_piOfLinks(rl)
                    # critical links:  find set of entanglement links w/ largest derivative
                    e2eFid_derivative = self.calc_largestFidelityDerivative(len(rl), piLinks)


        return C












