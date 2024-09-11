import numpy
import pandas
import pandas as pd
from pygad import pygad

from Problem_v2 import Problem_v2

prob = None
from os.path import exists
import Problem_v2 as Problem2


class Phase2GenaticTest:

    def __init__(self, dataset, prob):
        self.numCities = len(prob.NodesList) - 2

        DronesList = prob.DronesList
        # numpy.random.seed(42)

        self.prob = prob
        # self.filepath = 'results\phase1_2\GA_{}_{}_{}_{}.csv'.format(dataset, self.numCities, len(prob.DronesList), DronesList[0].profile)
        # self.filepathDetails = 'results\phase1_2\Details\{}_{}_{}_{}_Details.txt'.format(dataset, self.numCities, len(prob.DronesList), DronesList[0].profile)

    def __fitness_func__(self, solution, solution_idx):
        TSP_truck_route, total_cost, DronesSchedules, Truck_schedule, chrom1 = self.prob.evaluate_drones(chrom=solution)
        # print(1 / total_cost)
        return 1 / total_cost

    @staticmethod
    def on_generation2(ga_instance):
        print("phase 2:", ga_instance.generations_completed, 1 / ga_instance.best_solution()[1])
        pass

    def run(self):
        import time

        start = time.time()
        ini_pop = []
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
            l[i] = 1
            ini_pop.append(l)

        ga_instance = pygad.GA(num_generations=3000
                               , num_parents_mating=10
                               , sol_per_pop=self.numCities * 5
                               , num_genes=self.numCities
                               , fitness_func=self.__fitness_func__
                               , gene_space=[0, 1]
                               , gene_type=int
                               , on_generation=Phase2GenaticTest.on_generation2
                               # , stop_criteria=['reach_0.0001']
                               , stop_criteria=['saturate_5']
                               , initial_population=ini_pop
                               , mutation_num_genes=1)

        # Running the GA to optimize the parameters of the function.
        ga_instance.run()
        end = time.time()
        solution, solution_fitness, solution_idx = ga_instance.best_solution()
        time = end - start
        # print("--- {}".format(self.TSP_sch))

        TSP_truck_schedule, total_cost, DronesSchedules, Truck_schedule, chrom = self.prob.evaluate_drones(
            chrom=solution)
        # Problem2.printDetails(self.prob,total_cost,self.filepathDetails , False)
        #
        # print("Parameters of the best solution : {solution}".format(solution=solution))
        # print(DronesSchedules)
        # print("Fitness value of the best solution = {solution_fitness}".format(solution_fitness=total_cost))
        # print("Truck only cost:{} ".format(self.prob.evaluate_population([self.TSP_sch]).scoresList[0]))
        # # print("Index of the best solution : {solution_idx}".format(solution_idx=solution_idx))
        # print("-" * 100)
        # Truck_only_Cost= self.prob.evaluate_population([self.TSP_sch]).scoresList[0]
        # print("phase 2: Number of generations passed is {generations_completed}".format(
        #     generations_completed=ga_instance.generations_completed))

        return total_cost, chrom, Truck_schedule, DronesSchedules, time


class Phase1GenaticTest:

    def __init__(self, dataset, n_cities, n_drones, profile, optimum=None):
        self.optimum = optimum
        self.numCities = n_cities
        self.dataset = dataset
        DronesList = []
        for i in range(n_drones):
            DronesList.append(profile)
        numpy.random.seed(42)
        self.TSP_sch = None
        self.prob = Problem_v2(self.numCities, dataset=dataset, DronesList=DronesList, TSP_truck_schedule=self.TSP_sch)
        self.filepath = 'results\phase1_2\GA_GA_{}_{}_{}_{}.csv'.format(dataset, self.numCities, n_drones,
                                                                        DronesList[0])
        # self.filepathDetails = 'results\phase1_2\Details\GA_GA_{}_{}_{}_{}_best_Details.txt'.format(dataset, self.numCities,
        #                                                                                      n_drones, DronesList[0])

    def __fitness_func__(self, solution, solution_idx):
        # add depot
        solution = list(solution)
        solution = [0] + solution + [self.numCities + 1]
        self.prob.TSP_truck_schedule = solution
        g2 = Phase2GenaticTest(self.dataset, self.prob)
        total_cost, chrom, Truck_schedule, DronesSchedules, phase2_time = g2.run()
        # print('phase 1:' ,total_cost,solution, chrom)
        return 1 / total_cost

    @staticmethod
    def on_generation(ga_instance):
        solution, solution_fitness, solution_idx = ga_instance.best_solution()
        print("~~phase 1:", ga_instance.generations_completed, 1 / ga_instance.best_solution()[1])
        pass

    def run(self):
        import time

        start = time.time()
        gene_space = list(range(1, self.numCities + 1))

        stop_criteria = None
        if self.optimum != None:
            stop_criteria = ['reach_{}'.format(1 / self.optimum)]
        ga_instance = pygad.GA(num_generations=10000
                               , num_parents_mating=30
                               , sol_per_pop=self.numCities * 10
                               , num_genes=self.numCities
                               , fitness_func=self.__fitness_func__
                               , gene_space=gene_space
                               , gene_type=int
                               , on_generation=Phase1GenaticTest.on_generation
                               , allow_duplicate_genes=False
                               , stop_criteria=stop_criteria
                               # , stop_criteria=['saturate_10']
                               # , mutation_percent_genes = 10
                               , mutation_num_genes=1
                               )

        # Running the GA to optimize the parameters of the function.
        ga_instance.run()
        end = time.time()
        solution, solution_fitness, solution_idx = ga_instance.best_solution()
        time = end - start
        # print("--- {}".format(self.TSP_sch))
        # print(solution)
        self.prob.TSP_truck_schedule = [0] + list(solution) + [11]
        g2 = Phase2GenaticTest(self.dataset, self.prob)
        total_cost, chrom, Truck_schedule, DronesSchedules, phase2_time = g2.run()

        print("phase 1 : Number of generations passed is {generations_completed}".format(
            generations_completed=ga_instance.generations_completed))

        return self.prob.TSP_truck_schedule, total_cost, chrom, Truck_schedule, DronesSchedules, time


def main():
    df = pd.read_csv('./results/MSTS_Results.csv', sep=';')
    df.sort_values(by=['n_cities'], inplace=True)
    df1 = df.loc[(df['runID'] == 0) & (df['n_drones'] == 1)]
    df1 = df1.loc[(df['n_cities'] == 100)]
    # df1 = df1.loc[(df['dataset'] == 'RC101')]

    for index, row in df1.iterrows():
        dataset = row['dataset']
        n_cities = row['n_cities']
        n_drones = row['n_drones']
        profile = row['profile']
        opt_Total_Cost = row['Total Cost']
        opt_time = row['Best_Solution_Runtime']

        # # Explist = ['C101_100_1_L',
        # #            'C101_100_1_M',
        # #            'R101_100_1_L',
        # #            'RC101_50_1_L',
        # #            'RC101_100_1_L',
        # #            'RC101_100_1_M']
        # #
        # Explist = ['RC101_10_1_L']
        # if not row['Exp'] in Explist:
        #     continue

        print(row['Exp'])
        g = Phase1GenaticTest(dataset, n_cities, n_drones, profile, optimum=opt_Total_Cost)
        best_TSP, total_cost, solution, Truck_schedule, DronesSchedules, phase_time = g.run()

        phase_time = round(phase_time * 1000, 0)

        print("best_TSP : {}".format(best_TSP))
        print("best solution : {}".format(solution))
        print("DronesSchedules = {}".format(DronesSchedules))
        print("Truck_schedule :{} ".format(Truck_schedule))
        print("best Cost = {}".format(total_cost))
        print("-" * 100)

        if not exists(g.filepath):
            with open(g.filepath, 'w') as f:
                f.write(
                    "best_TSP;best_solution;Truck_schedule;DronesSchedules;best Cost;opt_Total_Cost;phase_time;opt_time\n")

        with open(g.filepath, 'a') as f:
            f.write(
                "{};{};{};{};{};{};{};{}\n".format(best_TSP, solution, Truck_schedule, DronesSchedules,
                                                   round(total_cost, 4), round(opt_Total_Cost, 4),
                                                   phase_time, opt_time))


if __name__ == "__main__":
    main()
