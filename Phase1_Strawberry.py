import copy
import os
import random
import time

import pandas as pd

from Problem_v2 import Problem_v2


class Phase1_Strawberry:
    def __init__(self, dataset, n_cities):
        ###### Experiment setting ######################################################
        # random.seed(100)
        DronesList = []

        self.prob = Problem_v2(n_cities, dataset=dataset, DronesList=DronesList)
        run_numbers = 5
        iterations = 1000
        earlystop = 50
        self.n_cities = n_cities

        ################################################################################
        NodesList = self.prob.NodesList
        numCities = len(NodesList)
        population_size = 1000
        rootRang = int(25 * numCities / 100)
        runnerRang = int(75 * numCities / 100)
        numRoots = int(numCities / 2)
        numRunners = int(numCities / 2)

        initPopulation_original = self.getInitialPpopulation(numpopulation=population_size)
        ############################################################################
        Exp = dataset + '_' + str(self.n_cities)  # + '_' + str(len(DronesList)) + '_' + DronesList[0]
        # ExpID = "SBA_{}_{}".format(Exp, datetime.now().strftime("%Y%m%d%H%M%S"))
        ExpID = "{}".format(Exp)

        dir = "{}\\results\\phase1_SBA\\{}".format(os.getcwd(), ExpID)
        os.makedirs(dir, exist_ok=True)
        runsummary = []
        runsummary.append("run_ID;best_score;num_iter; Best_iter;TSP_schedule;phase1_time\n")

        for run_number in range(run_numbers):
            rundir = "{}\\run_{}".format(dir, run_number)
            os.makedirs(rundir, exist_ok=True)
            print("run_ID={}".format(run_number + 1))
            initPopultion = copy.deepcopy(initPopulation_original)
            last_bestscore = float("inf")
            repeated = 0
            bestIteration = 0
            iterantionsummary = []
            iterantionsummary.append("iter_ID;best_score;TSP_schedule\n")
            runstart = time.time()
            # iterantiondetails = []

            for iteration_num in range(iterations):
                print(iteration_num)
                fullPopultion = self.recombine(initPopultion, numCities=numCities, rootRang=rootRang,
                                               runnerRang=runnerRang,
                                               numRoots=numRoots, numRunners=numRunners)
                evaluatedPopulation = self.fitnessEvaluation(fullPopultion, self.prob)
                initPopultion = []
                [initPopultion.append(x[1:-1]) for x in evaluatedPopulation.populationList if
                 x[1:-1] not in initPopultion]
                initPopultion = initPopultion[0:population_size]

                if last_bestscore > evaluatedPopulation.bestScore:
                    bestIteration = iteration_num
                    last_bestscore = evaluatedPopulation.bestScore
                    repeated = 0
                else:
                    repeated = repeated + 1
                    if (repeated == earlystop):  # no early stop
                        break
                iterantionsummary.append("{};{};{}\n".format(iteration_num, round(evaluatedPopulation.bestScore, 2),
                                                             evaluatedPopulation.bestPop))
                # iterantiondetails.append(evaluatedPopulation.scoresList[0:population_size])

            TSP_schedule = evaluatedPopulation.bestPop[0]
            print(TSP_schedule)
            runend1 = time.time()
            run_time = round(runend1 - runstart, 3)
            with open(rundir + "\\iterations_summary.csv", 'a') as file:
                for s in iterantionsummary:
                    file.write(s)
            # with open(rundir + "\\iterations_details.csv", 'a') as file:
            #     for s in iterantiondetails:
            #         file.write(';'.join(map(str, s)) + '\n')

            with open(rundir + "\\best_TSP.csv", 'a') as file:
                file.write("score;sch\n")
                finalList = []
                for i in range(len(evaluatedPopulation.populationList)):
                    t = [evaluatedPopulation.scoresList[i], evaluatedPopulation.populationList[i]]
                    if not t in finalList:
                        finalList.append(t)
                        file.write("{};{}\n".format(t[0], t[1]))

            runsummary.append(
                "{};{};{};{};{};{}\n".format(run_number, round(evaluatedPopulation.bestScore, 2), iteration_num + 1,
                                             bestIteration, evaluatedPopulation.bestPop, run_time))

            print("\nOutput>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
            print("Best TSP_schedule is: {}".format(evaluatedPopulation.bestPop))
            print("Best score={} , num_iter={} , Best_iter={} ,run_time={}".format(
                round(evaluatedPopulation.bestScore, 2), iteration_num + 1, bestIteration + 1, run_time))
            print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
        with open(dir + "\\runs_summary.csv", 'a') as file:
            for s in runsummary:
                file.write(s)
        with open(dir + "\\Exp_setting.csv", 'a') as file:
            file.write("number of Cities:{}\n".format(numCities - 2))
            file.write("population size:{}\n".format(population_size))
            file.write("rootRang:{}\n".format(rootRang))
            file.write("runnerRang:{}\n".format(runnerRang))
            file.write("numRoots:{}\n".format(numRoots))
            file.write("numRunners:{}\n".format(numRunners))
            file.write("number of runs:{}\n".format(run_numbers))
            file.write("number of iterations:{}\n".format(iterations))
            file.write("early-stop:{}\n".format(earlystop))

    def getInitialPpopulation(self, numpopulation):
        population = [[i for i in random.sample(range(1, self.n_cities + 1), k=self.n_cities)] for j in
                      range(numpopulation)]
        return population

    def recombine(self, initPopultion, numCities=15, rootRang=3, runnerRang=7, numRoots=4, numRunners=4):
        numCities = numCities - 2
        Root = list(range(1, rootRang))
        runner = list(range(rootRang, runnerRang))
        # print("root range is  {}".format(Root))
        # print("runner RANG is  {}".format(runner))
        direction = [-1, 1]
        newpopulation = []
        # print(population)
        Roots = copy.deepcopy(initPopultion)
        Runners = copy.deepcopy(initPopultion)
        for p in Roots:
            # root
            # print("old {}".format(p))
            for i in range(numRoots):
                selectCityPos1 = random.randint(0, numCities - 1)
                city1_newpos = selectCityPos1 + random.choice(Root) * random.choice(direction)
                # swap
                if city1_newpos >= 0 | city1_newpos < numCities:
                    p[selectCityPos1], p[city1_newpos] = p[city1_newpos], p[selectCityPos1]
                    # print("root swap {},{}".format(p[selectCityPos1], p[city1_newpos]))
                else:
                    # print("root canceled")
                    pass
            # print("new {}".format(p))
            # print("*" * 80)
        for p in Runners:
            # runner
            for i in range(numRunners):
                selectCityPos2 = random.randint(0, numCities - 1)
                city2_newpos = selectCityPos2 + random.choice(runner) * random.choice(direction)
                # swap

                if city2_newpos >= 0 | city2_newpos < numCities:
                    p[selectCityPos2], p[city2_newpos] = p[city2_newpos], p[selectCityPos2]
                    # print("runner swap {},{}".format(p[selectCityPos2], p[city2_newpos]))
                else:
                    # print("runner canceled")
                    pass
            # print("new {}".format(p))
            # print("*" * 80)
        fullPopultion = Roots + Runners + initPopultion

        return fullPopultion

    def fitnessEvaluation(self, fullPopultion, problem):
        num = len(fullPopultion[0]) + 1
        fullPopultion = [[0] + p + [num] for p in fullPopultion]
        evaluatedPopulation = problem.evaluate_population(fullPopultion)
        # print("\nBest TSP_Truck sheedule is: ")
        # print(evaluatedPopulation.best)
        # print(evaluatedPopulation.bestScore)
        # TSP_truck_schedule = evaluatedPopulation.best
        return evaluatedPopulation


def main():
    df = pd.read_csv('.//results//MSTS_Results.csv', sep=';')
    df = df[['dataset', 'n_cities']]
    df1 = df.drop_duplicates()

    # df1 = df1.loc[(df['n_cities'] == 25)]

    for index, row in df1.iterrows():
        Phase1_Strawberry(row['dataset'], row['n_cities'])


if __name__ == "__main__":
    main()
