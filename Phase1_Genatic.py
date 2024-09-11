import os
import time

import numpy
import numpy as np
import pandas
import pandas as pd
from pygad import pygad

from Population import Population
from Problem_v2 import Problem_v2

prob = None
from os.path import exists
import Problem_v2 as Problem2


class Phase1_Genatic:

    def __init__(self, dataset, n_cities):
        self.numCities = n_cities

        DronesList = []
        # numpy.random.seed(42)
        self.TSP_sch = None
        self.prob = Problem_v2(self.numCities, dataset=dataset, DronesList=DronesList, TSP_truck_schedule=self.TSP_sch)
        # self.filepath = 'results\phase1_2\GA_{}_{}_{}_{}.csv'.format(dataset, self.numCities, n_drones, DronesList[0])
        # self.filepathDetails = 'results\phase1_2\Details\{}_{}_{}_{}_best_Details.txt'.format(dataset, self.numCities,
        #                                                                                      n_drones, DronesList[0])

    def __fitness_func__(self, solution, solution_idx):
        solution = list(solution)
        solution.append(len(solution) + 1)
        solution.insert(0, 0)

        fitness_list = [self.prob.AdjacencyMatrix[solution[i], solution[i + 1]] for i in range(len(solution) - 1)]
        total_cost = sum(fitness_list)
        return 1 / total_cost

    @staticmethod
    def on_generation(ga_instance):
        print(ga_instance.generations_completed, 1 / ga_instance.best_solution()[1])
        pass

    def run(self):
        import time

        start = time.time()
        gene_space = list(range(0, self.numCities + 1))
        gene_space.remove(0)
        ga_instance = pygad.GA(num_generations=3000
                               , num_parents_mating=round(0.1 * self.numCities * 20)
                               , sol_per_pop=self.numCities * 20
                               , num_genes=self.numCities
                               , fitness_func=self.__fitness_func__
                               , gene_space=gene_space
                               , gene_type=int
                               , on_generation=Phase1_Genatic.on_generation
                               , allow_duplicate_genes=False
                               # , stop_criteria=['reach_0.0001']
                               , stop_criteria=['saturate_10']
                               # , parallel_processing=["thread", 5]
                               , mutation_num_genes=2
                               # , keep_parents=5
                               )

        # Running the GA to optimize the parameters of the function.
        ga_instance.run()
        end = time.time()
        print("Number of generations passed is {generations_completed}".format(
            generations_completed=ga_instance.generations_completed))
        best_solution = ga_instance.best_solution()

        population = []
        for x in ga_instance.population:
            solution = list(x)
            solution.append(len(x) + 1)
            solution.insert(0, 0)
            population.append(solution)

        p = Population(population, self.prob.AdjacencyMatrix)
        return p.populationList, p.scoresList, ga_instance.best_solution_generation, ga_instance.generations_completed


def main():
    df = pd.read_csv('./results/MSTS_Results.csv', sep=';')
    df.sort_values(by=['n_cities'], inplace=True)
    df = df[['dataset', 'n_cities']]
    df1 = df.drop_duplicates()
    # df1 = df1.loc[(df['n_cities'] == 8)]

    for index, row in df1.iterrows():
        dataset = row['dataset']
        n_cities = row['n_cities']

        # print(row['Exp'])

        run_numbers = 5
        Exp = dataset + '_' + str(n_cities)
        Explist = [
            # 'R101_8',
            #         'R101_10',
            #         'R101_50',
            #         'R101_100',
            #         'RC101_10',
            #         'RC101_10',
            #         'RC101_50',
            #         'RC101_50',
            'RC101_100'
        ]
        if not Exp in Explist:
            continue
        print(Exp)
        ExpID = "{}".format(Exp)

        dir = "{}\\results\\phase1_GA\\{}".format(os.getcwd(), ExpID)
        os.makedirs(dir, exist_ok=True)
        runsummary = []
        runsummary.append("run_ID;best_score;num_iter; Best_iter;TSP_schedule;phase1_time\n")

        for run_number in range(run_numbers):
            rundir = "{}\\run_{}".format(dir, run_number)
            os.makedirs(rundir, exist_ok=True)
            print("run_ID={}".format(run_number + 1))
            runstart = time.time()

            g = Phase1_Genatic(dataset, n_cities)
            solutions, solutions_cost, best_solution_generation, generations_completed = g.run()

            TSP_schedule = solutions[0]
            print(TSP_schedule)
            runend1 = time.time()
            run_time = round(runend1 - runstart, 3)

            with open(rundir + "\\best_TSP.csv", 'a') as file:
                file.write("score;sch\n")
                finalList = []
                for i in range(len(solutions)):
                    t = [solutions_cost[i], solutions[i]]
                    if not t in finalList:
                        finalList.append(t)
                        file.write("{};{}\n".format(t[0], t[1]))

            runsummary.append(
                "{};{};{};{};{};{}\n".format(run_number, round(solutions_cost[0], 2), generations_completed + 1,
                                             best_solution_generation, solutions[0], run_time))

            print("\nOutput>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
            print("Best TSP_schedule is: {}".format(solutions[0]))
            print("Best score={} , num_iter={} , Best_iter={} ,run_time={}".format(
                round(solutions_cost[0], 2), generations_completed + 1, best_solution_generation + 1, run_time))
            print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
        with open(dir + "\\runs_summary.csv", 'a') as file:
            for s in runsummary:
                file.write(s)
        # with open(dir + "\\Exp_setting.csv", 'a') as file:
        #     file.write("number of Cities:{}\n".format(numCities - 2))
        #     file.write("population size:{}\n".format(population_size))
        #     file.write("rootRang:{}\n".format(rootRang))
        #     file.write("runnerRang:{}\n".format(runnerRang))
        #     file.write("numRoots:{}\n".format(numRoots))
        #     file.write("numRunners:{}\n".format(numRunners))
        #     file.write("number of runs:{}\n".format(run_numbers))
        #     file.write("number of iterations:{}\n".format(iterations))
        #     file.write("early-stop:{}\n".format(earlystop))


if __name__ == "__main__":
    main()
