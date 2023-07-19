from collections import defaultdict
from copy import deepcopy

from algorithm.AlgorithmBase import Algorithm
from topo.Topo import Topo, Path
from topo.Node import to
import numpy as np
import heapq
from utils.utils import ReducibleLazyEvaluation
import abc
from dataclasses import dataclass
from functools import reduce
from itertools import dropwhile
from algorithm.AlgorithmBase import Algorithm

import shortestpaths as sp

# Mainly P2 changes (maximize served quantum-user pairs and expected throughput)
class MultiEntanglement(Algorithm):
    def __init__(self, topo):
        super().__init__(topo)
        self.name="Multi_E"
        self.shortestPathsForPairs = []
    #prepare topology/entanglements?
    #def prepare(self):
    #def P2(self):
    #def P4(self):
    
    # Sort each S-D pair list of paths by cost in ascending order
    def sortPathsByCost(self):
        for i in range(0, len(self.shortestPathsForPairs)):
            self.shortestPathsForPairs[i][1].sort(key=lambda x: x[1])
    
    # Maximize source-destination pairs
    # Select main routing path for each pair
    # Mind Switchs' qubits assigned for paths
    def selectivePaths(self):
        expectedPathNum = len(self.srcDstPairs)
        retrievedPathNum = pow(expectedPathNum, 2)
        for pair in self.srcDstPairs:
            src, dst = pair[0], pair[1]
            shortestPaths = self.topo.shortestPathYenAlg(src.id, dst.id, retrievedPathNum)
            self.shortestPathsForPairs.append([pair, shortestPaths])
        
        # Sort paths by shortest to longest
        self.sortPathsByCost()
        
        # For each S-D pair, check if it has the correct number of paths
        for i in range(0, len(self.srcDstPairs)):
            pathNum = len(self.shortestPathsForPairs[i][1])
            if pathNum < expectedPathNum:
                # Add paths until M paths
                print("Need to add paths")
            elif pathNum > expectedPathNum:
                # Delete largest cost paths until M paths
                for extraPath in range(0, pathNum - expectedPathNum):
                    self.shortestPathsForPairs[i][1].pop(-1)
    
    def branchAndPrice1(self):
        # Given current path...
        
    #Select shortest distance path
    def integerSolution1(self):
        # tempIntSol = 0
        # sort ? in descending order
        # for pair,shortestPaths in self.shortestPathsForPairs:
        #     if ? == 1:
        #         ? = 1
        #         Mark pair
        # Find max ? <1 and satisfies s-d not entangled
        self.branchAndPrice1()
    
    def branchAndPrice2(self):
    
    # Maximize expected throughput of all source-destination pairs from Step1
    # Determine qubits assigned to paths from Step1
    # Mind Switchs' qubits assigned for paths
    def integerSolution2(self):