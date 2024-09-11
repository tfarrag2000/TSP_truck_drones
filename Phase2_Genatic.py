import random
import numpy
import numpy as np
import pandas
import pandas as pd
from pygad import pygad
from Problem_v2 import Problem_v2

prob = None
from os.path import exists
import Problem_v2 as Problem2


class genaticTest:
    def __init__(self, dataset, n_cities, n_drones, profile, TSP_sch, TSP_id=0, optimum=None):
        self.optimum = optimum
        self.numCities = len(TSP_sch) - 2
        self.previosSolutionsFitness = {}
        DronesList = []
        for i in range(n_drones):
            DronesList.append(profile)
        # numpy.random.seed(42)

        self.prob = Problem_v2(self.numCities, dataset=dataset, DronesList=DronesList, TSP_truck_schedule=TSP_sch)
        self.filepathDetails = 'results\phase2\Details\{}_{}_{}_{}_TSP{}_Details.txt'.format(dataset, self.numCities,
                                                                                             n_drones, DronesList[0],
                                                                                             TSP_id)

    def __fitness_func__(self, solution, solution_idx):
        # if tuple(solution) in self.previosSolutionsFitness:
        #     fitness =self.previosSolutionsFitness[tuple(solution)]
        #     # self.gg=self.gg+1
        # else:
        TSP_truck_route, total_cost, DronesSchedules, Truck_schedule, chrom1 = self.prob.evaluate_drones(chrom=solution)
        fitness = 1 / total_cost
        # self.previosSolutionsFitness[tuple(solution)]=fitness
        return fitness

    @staticmethod
    def on_generation(ga_instance):
        print(ga_instance.generations_completed, 1 / ga_instance.best_solution()[1])
        pass

    @staticmethod
    def on_start(ga_instance):
        # print(ga_instance.generations_completed , ga_instance.best_solution()[1])
        pass

    def run(self):
        import time

        ini_pop = []
        # ini_pop.append([ 1, 0, 1, 0, 1, 1, 1, 0, 0, 1])
        for i in range(self.numCities):
            l = [0] * self.numCities
            l[i] = 1
            ini_pop.append(l)

        for i in range(self.numCities):
            l = []
            k = 1
            for j in range(self.numCities):
                l.append(k)
                if k == 1:
                    k = 0
                else:
                    k = 1
            ini_pop.append(l)
            l = []
            k = 0
            for j in range(self.numCities):
                l.append(k)
                if k == 1:
                    k = 0
                else:
                    k = 1
            ini_pop.append(l)

        stop_criteria = ['saturate_50']

        # if self.optimum != None:
        #    stop_criteria .append('reach_{}'.format(1 / self.optimum))

        ga_instance = pygad.GA(num_generations=3000
                               , num_parents_mating=round(0.1 * len(ini_pop))
                               # , sol_per_pop=self.numCities *20
                               , num_genes=self.numCities
                               , fitness_func=self.__fitness_func__
                               , gene_space=[0, 1]
                               , gene_type=int
                               , on_generation=genaticTest.on_generation
                               , on_start=genaticTest.on_start
                               , stop_criteria=stop_criteria
                               , initial_population=ini_pop
                               , mutation_num_genes=1
                               )

        # Running the GA to optimize the parameters of the function.
        start = time.time()
        ga_instance.run()
        end = time.time()
        solution, solution_fitness, solution_idx = ga_instance.best_solution()
        time = end - start
        # print("--- {}".format(self.time)
        TSP_truck_route, total_cost, DronesSchedules, Truck_schedule, chrom = self.prob.evaluate_drones(chrom=solution)
        # Problem2.printDetails(self.prob,total_cost,self.filepathDetails , False)
        #
        # print("Parameters of the best solution : {solution}".format(solution=solution))
        # print(DronesSchedules)
        # print("Fitness value of the best solution = {solution_fitness}".format(solution_fitness=total_cost))
        # print("Truck only cost:{} ".format(self.prob.evaluate_population([self.TSP_sch]).scoresList[0]))
        # print("Index of the best solution : {solution_idx}".format(solution_idx=solution_idx))
        # print("-" * 100)
        # Truck_only_Cost= self.prob.evaluate_population([self.TSP_sch]).scoresList[0]
        print("Number of generations passed is {}".format(ga_instance.generations_completed))

        return total_cost, chrom, Truck_schedule, DronesSchedules, time


def main():
    Phase1 = 'GA'

    df = pd.read_csv('./results/MSTS_Results.csv', sep=';')
    df.sort_values(by=['n_cities'], inplace=True)
    df1 = df.loc[(df['runID'] == 0) & (df['n_drones'] == 1)]
    # df1 = df1.loc[(df['n_cities'] ==100)]

    for index, row in df1.iterrows():
        dataset = row['dataset']
        n_cities = row['n_cities']
        n_drones = row['n_drones']
        profile = row['profile']
        opt_Total_Cost = row['Total Cost']
        opt_time = row['Best_Solution_Runtime']

        Exp = Phase1 + '_' + dataset + '_' + str(n_cities) + '_' + str(n_drones) + '_' + profile
        Explist = ['R101_10_1_L']

        if not row['Exp'] in Explist:
            continue
        print(row['Exp'])

        # phase 1 --------------------------------------------------------------------------------
        filepath = r".\results\phase1_{}\{}_{}\runs_summary.csv".format(Phase1, dataset, n_cities)
        phase1_data = pandas.read_csv(filepath, delimiter=";", header=0)
        phase1_best_score_min = phase1_data['best_score'].min()
        best_run_data = phase1_data.loc[phase1_data['best_score'] <= phase1_best_score_min].iloc[0]
        phase1_time = best_run_data['phase1_time'] * 1000
        phase1_bestRun_ID = best_run_data['run_ID']
        print('phase1_bestRun_ID: ', phase1_bestRun_ID)
        # to milliseconds
        filepath = r".\results\phase1_{}\{}_{}\run_{}\best_TSP.csv".format(Phase1, dataset, n_cities, phase1_bestRun_ID)
        phase1_TSP_sch = pandas.read_csv(filepath, delimiter=";", header=0, engine='python')
        phase1_TSP_sch = phase1_TSP_sch.drop_duplicates(subset=['sch'])
        # select number of top tsp sch from phase 1
        phase1_TSP_sch = phase1_TSP_sch.head(5)
        # -------------------------------------------------------------------------------------------------
        n = 0
        TSPList = []
        # TSPList.append([0, 5, 6, 8, 7, 10, 1, 9, 3, 4, 2, 11])

        for index, phase1row in phase1_TSP_sch.iterrows():
            t = list(map(int, phase1row['sch'].replace("[", "").replace("]", "").split(",")))
            TSPList.append(t)

        for tt in TSPList:
            print(row['Exp'])
            print('TSP:', n, 'optimum:', opt_Total_Cost)
            print(tt)
            best_TSP = tt
            g = genaticTest(dataset, n_cities, n_drones, profile, tt, n, optimum=opt_Total_Cost * 1.05)
            total_cost, solution, Truck_schedule, DronesSchedules, phase2_time = g.run()

            # to millisconds
            phase2_time = round(phase2_time * 1000, 0)
            total_time = phase1_time + phase2_time
            print("best_TSP : {}".format(best_TSP))
            print("best solution : {}".format(solution))
            print("DronesSchedules = {}".format(DronesSchedules))
            print("Truck_schedule :{} ".format(Truck_schedule))
            print("best Cost = {}".format(total_cost))
            print("-" * 100)
            filepath = 'results\phase2\GA_{}_{}_{}_{}_{}.csv'.format(Phase1, dataset, n_cities, n_drones, profile)

            if not exists(filepath):
                with open(filepath, 'w') as f:
                    f.write(
                        "best_TSP;best_solution;Truck_schedule;DronesSchedules;best Cost;opt_Total_Cost;phase1_time;phase2_time;total_time;opt_time\n")

            with open(filepath, 'a') as f:
                f.write(
                    "-{};{};{};{};{};{};{};{};{};{}\n".format(best_TSP, solution, Truck_schedule, DronesSchedules,
                                                              round(total_cost, 4), round(opt_Total_Cost, 4),
                                                              phase1_time,
                                                              phase2_time, total_time, opt_time))
            if total_cost <= opt_Total_Cost * 1.08:
                print("***************************************")
                print("#######################################")
                print(n)
                break
            n = n + 1


if __name__ == "__main__":
    main()
