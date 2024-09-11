class Population:
    Pop = {}

    def __init__(self, populationList, adjacency_mat):
        self.populationList = populationList
        self.bestScore = -1
        bestIndex = -1
        self.bestPop = []
        self.adjacency_mat = adjacency_mat
        self.scoresList = self.evaluate()
        self.sort()
        # for i in range(len(populationList)):
        #     Population.Pop[self.scoresList[i]]=tuple(self.populationList[i])
        # p=Population.Pop
        # pass

    def fitness(self, element):
        fitness_list = [self.adjacency_mat[element[i], element[i + 1]] for i in range(len(element) - 1)]

        # totalserviceTime=(len(self.populationList[0]) -2) * 10
        totalserviceTime = 0
        return sum(fitness_list) + totalserviceTime

    def evaluate(self):
        distances = [self.fitness(element) for element in self.populationList]
        return distances

    def sort(self):

        self.populationList = [x for y, x in sorted(zip(self.scoresList, self.populationList))]

        self.scoresList.sort()
        self.bestScore = self.scoresList[0]
        # find all the best
        for i in range(len(self.scoresList)):
            if self.scoresList[i] != self.scoresList[0]:
                break
            else:
                p = self.populationList[i]
                if p not in self.bestPop:
                    self.bestPop.append(p)

        # print("total time for each pop:")
        # print(self.scoresList)
        pass
