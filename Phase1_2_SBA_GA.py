import copy
import os
import random
import time

import pandas as pd
from comtypes.safearray import numpy

from Problem_v2 import Problem_v2
from pygad import pygad
import Problem_v2 as Problem2


class genaticTest:

    def __init__(self, dataset, prob):
        self.numCities = len(prob.NodesList) - 2

        DronesList = prob.DronesList
        numpy.random.seed(42)

        self.prob = prob
        self.filepath = 'results\phase1_2\GA_{}_{}_{}_{}.csv'.format(dataset, self.numCities, len(prob.DronesList),
                                                                     DronesList[0].profile)
        self.filepathDetails = 'results\phase1_2\Details\{}_{}_{}_{}_Details.txt'.format(dataset, self.numCities,
                                                                                         len(prob.DronesList),
                                                                                         DronesList[0].profile)

    def __fitness_func__(self, solution, solution_idx):
        TSP_truck_route, total_cost, DronesSchedules, Truck_schedule, chrom1 = self.prob.evaluate_drones(chrom=solution)
        # print(total_cost)
        return 1 / total_cost

    @staticmethod
    def on_generation(ga_instance):
        # print(ga_instance.generations_completed , ga_instance.best_solution()[1])
        pass

    def run(self):
        import time

        start = time.time()

        ga_instance = pygad.GA(num_generations=3000
                               , num_parents_mating=50
                               , sol_per_pop=self.numCities * 10
                               , num_genes=self.numCities
                               , fitness_func=self.__fitness_func__
                               , gene_space=[0, 1]
                               , gene_type=int
                               , on_generation=genaticTest.on_generation
                               # , stop_criteria=['reach_0.0001']
                               , stop_criteria=['saturate_5']
                               )

        # Running the GA to optimize the parameters of the function.
        ga_instance.run()
        end = time.time()
        solution, solution_fitness, solution_idx = ga_instance.best_solution()
        time = end - start
        # print("--- {}".format(self.TSP_sch))

        TSP_truck_route, total_cost, DronesSchedules, Truck_schedule, chrom = self.prob.evaluate_drones(chrom=solution)
        Problem2.printDetails(self.prob, total_cost, self.filepathDetails, False)
        #
        # print("Parameters of the best solution : {solution}".format(solution=solution))
        # print(DronesSchedules)
        # print("Fitness value of the best solution = {solution_fitness}".format(solution_fitness=total_cost))
        # print("Truck only cost:{} ".format(self.prob.evaluate_population([self.TSP_sch]).scoresList[0]))
        # # print("Index of the best solution : {solution_idx}".format(solution_idx=solution_idx))
        # print("-" * 100)
        # Truck_only_Cost= self.prob.evaluate_population([self.TSP_sch]).scoresList[0]
        print("Number of generations passed is {generations_completed}".format(
            generations_completed=ga_instance.generations_completed))

        return total_cost, chrom, Truck_schedule, DronesSchedules, time


class Phase1_Strawberry:
    def __init__(self, dataset, n_cities, n_drones, profile):
        ###### Experiment setting ######################################################
        random.seed(100)

        DronesList = []
        for i in range(n_drones):
            DronesList.append(profile)

        self.prob = Problem_v2(n_cities, dataset=dataset, DronesList=DronesList)
        run_numbers = 1
        iterations = 1000
        earlystop = 50
        self.n_cities = n_cities
        self.dataset = dataset
        self.DronesList = DronesList
        ################################################################################

        NodesList = self.prob.NodesList
        numCities = len(NodesList)
        population_size = 100
        rootRang = int(25 * numCities / 100)
        runnerRang = int(75 * numCities / 100)
        numRoots = int(numCities / 2)
        numRunners = int(numCities / 2)

        initPopultion_original = self.getInitialPpopulation(numpopulation=population_size)

        ############################################################################
        Exp = dataset + '_' + str(self.n_cities) + '_' + str(len(DronesList)) + '_' + DronesList[0]
        # ExpID = "SBA_{}_{}".format(Exp, datetime.now().strftime("%Y%m%d%H%M%S"))
        ExpID = "SBA_{}".format(Exp)

        dir = "{}\\results\\phase1_2\\{}".format(os.getcwd(), ExpID)
        os.makedirs(dir, exist_ok=True)
        runsummary = []
        runsummary.append("run_ID;best_score;num_iter; Best_iter;TSP_schedule;phase1_time\n")

        for run_number in range(run_numbers):
            rundir = "{}\\run_{}".format(dir, run_number)
            os.makedirs(rundir, exist_ok=True)
            print("run_ID={}".format(run_number + 1))
            initPopultion = copy.deepcopy(initPopultion_original)
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
                [initPopultion.append(x[1][1:-1]) for x in evaluatedPopulation]
                initPopultion = initPopultion[0:population_size]
                if last_bestscore > evaluatedPopulation[0][0]:
                    bestIteration = iteration_num
                    last_bestscore = evaluatedPopulation[0][0]
                    repeated = 0
                else:
                    repeated = repeated + 1
                    if (repeated == earlystop):  # no early stop
                        break
                iterantionsummary.append("{};{};{};{}\n".format(iteration_num, round(evaluatedPopulation[0][0], 2),
                                                                evaluatedPopulation[0][1], evaluatedPopulation[0][2]))
                # iterantiondetails.append(evaluatedPopulation.scoresList[0:population_size])

            TSP_schedule = evaluatedPopulation[0][1]
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
                file.write("score;sch;chrom\n")
                finalList = []
                for i in range(len(evaluatedPopulation)):
                    t = evaluatedPopulation[i]
                    if not t in finalList:
                        finalList.append(t)
                        file.write("{};{};{}\n".format(t[0], t[1], t[2]))

            runsummary.append(
                "{};{};{};{};{};{}\n".format(run_number, round(evaluatedPopulation[0][0], 2), iteration_num + 1,
                                             bestIteration, evaluatedPopulation[0][1], run_time))

            print("\nOutput>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
            print("Best TSP_schedule is: {}".format(evaluatedPopulation[0][1]))
            print("Best score={} , num_iter={} , Best_iter={} ,run_time={}".format(
                round(evaluatedPopulation[0][0], 2), iteration_num + 1, bestIteration + 1, run_time))
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
        evaluatedPopulation = []
        for t in fullPopultion:
            self.prob.TSP_truck_schedule = t
            g = genaticTest(self.dataset, self.prob)
            total_cost, solution, Truck_schedule, DronesSchedules, phase2_time = g.run()
            tt = (total_cost, t, solution)
            if tt not in evaluatedPopulation:
                evaluatedPopulation.append(tt)
        evaluatedPopulation.sort(key=lambda i: i[0], reverse=False)
        # print("\nBest TSP_Truck sheedule is: ")
        # print(evaluatedPopulation.best)
        # print(evaluatedPopulation.bestScore)
        # TSP_truck_schedule = evaluatedPopulation.best
        return evaluatedPopulation


def main():
    df = pd.read_csv('./results/MSTS_Results.csv', sep=';')

    df1 = df.loc[(df['runID'] == 0) & (df['n_drones'] == 1) & (df['n_cities'] == 10) & (df['profile'] == 'L') & (
                df['dataset'] == 'RC101')]
    # df1 = df.loc[(df['runID'] == 0) & (df['n_drones'] == 1) ]

    for index, row in df1.iterrows():
        Phase1_Strawberry(row['dataset'], row['n_cities'], row['n_drones'], row['profile'])


if __name__ == "__main__":
    main()
