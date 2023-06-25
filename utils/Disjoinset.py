
class Disjointset:

    def __init__(self,n):
        self.parentOf=[0 for i in range(n)]

    def getRepresentative(self,i):
        stack=[]
        tmp=1

        while self.parentOf[tmp] != tmp:
            stack.append(tmp)
            tmp=self.parentOf[tmp]
        for item in stack:
            self.parentOf[item]=tmp
        return tmp

    def merge(self,i,j):
        if not self.sameDivision(i,j):
            self.parentOf[self.getRepresentative(j)]=self.getRepresentative(i)


    def sameDivision(self,i,j):
        return self.getRepresentative(i)==self.getRepresentative(j)



