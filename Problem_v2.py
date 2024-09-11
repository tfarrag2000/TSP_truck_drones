import math

import numpy as np

from Population import Population
from classes.Drone import Drone
from classes.Node import Node
from classes.Truck import Truck


class Problem_v2:

    def __init__(self, numCities_withoutdepot, dataset='C101', DronesList=['L'], TSP_truck_schedule=None):

        ###### prepare nodes list
        if dataset == 'C101':
            fn = 'Data\\C101_modified.txt'
        elif dataset == 'R101':
            fn = 'Data\\R101_modified.txt'
        elif dataset == 'RC101':
            fn = 'Data\\RC101_modified.txt'

        self.TSP_truck_schedule = TSP_truck_schedule
        nodes_tuples_list = []
        filename = "Data\\nodes_coord\\{}_nodes_coord_{}.txt".format(dataset, numCities_withoutdepot)
        fo = open(filename, "w")
        depotstr = None

        for t in open(fn).read().split():
            city = list(map(int, t.strip('()').split(',')))
            if depotstr == None:
                depotstr = t
            nodes_tuples_list.append(city)
            fo.write("{}\n".format(t))
            if len(nodes_tuples_list) + 1 == numCities_withoutdepot + 2:
                break
        depot = list(map(int, depotstr.strip('()').split(',')))
        depot[0] = numCities_withoutdepot + 1  ## change the id
        nodes_tuples_list.append(depot)
        fo.write("{}".format(depot).replace('[', '').replace(']', '').replace(' ', ''))
        fo.close()
        ##############
        self.DronesList = []
        self.NodesList = []
        self.TSP_truck_route = None
        self.AdjacencyMatrix = None

        Drone.drones_count = 0
        Drone.MaxDronesCapacity = -1
        Truck.trucks_count = 0

        self.__CreateNodesList(nodes_tuples_list)
        self.__getAjacencyMatrix()
        self.truck = Truck(self.NodesList, self.AdjacencyMatrix, speed=1)
        self.__CreateDronesList(DronesList)

        ## create a list of nodes that should be served by the truck depot + nodes that have heavy shipment
        self.nonCombinedList = [self.NodesList[0].id]
        for node in self.NodesList:
            if node.shipment_weight > Drone.MaxDronesCapacity:
                self.nonCombinedList.append(node.id)
        self.nonCombinedList.append(self.NodesList[-1].id)
        ######

    def __getAjacencyMatrix(self):
        # assume speed =1
        n = len(self.NodesList)
        self.AdjacencyMatrix = np.zeros((n, n))
        for i in range(n - 1):
            for j in range(i + 1, n):
                node1 = self.NodesList[i]
                node2 = self.NodesList[j]
                xdiff = node2.x - node1.x
                ydiff = node2.y - node1.y
                self.AdjacencyMatrix[i, j] = math.sqrt((xdiff ** 2) + (ydiff ** 2))
                self.AdjacencyMatrix[j, i] = self.AdjacencyMatrix[i, j]
        # fo = open("Truck_adjacency_mat.txt", "w")
        # fo.write('\n'.join(' '.join(str(x) for x in l) for l in self.AdjacencyMatrix))
        # fo.close()

    def __CreateDronesList(self, DroneList):
        Drone.MaxDronesCapacity = -1

        for prof in DroneList:
            d = Drone(prof, self)
            d.currentNode = self.NodesList[0]
            self.DronesList.append(d)

    def __CreateNodesList(self, nodes_tuples_list):
        for n in nodes_tuples_list:
            node = Node(n[0], n[1], n[2], n[3], n[6])
            self.NodesList.append(node)

    def __BuildingTruckRoute(self):
        self.TSP_truck_route = []
        for city in self.TSP_truck_schedule:
            node = self.NodesList[city]
            node.reset()
            self.TSP_truck_route.append(node)

    def __getPrevNode(self, node):
        dx = self.TSP_truck_route.index(node)
        prev = self.TSP_truck_route[dx - 1]
        return prev

    def __getNextNode(self, node):
        dx = self.TSP_truck_route.index(node)
        if dx == len(self.TSP_truck_route) - 1:
            return None
        next_node = self.TSP_truck_route[dx + 1]
        return next_node

    def __getTruckTime(self, node):
        # calculate total  time to reach the node from the truck current node
        id = self.truck.currentNode.id
        next_id = node.id
        time = self.truck.AdjacencyMatrix[id][next_id]
        # print("TruckTime = {}    {} ~~~ {}".format(time, id, next_id))
        return time

    def __BuildingDronesSchedules(self):
        ## the most important method
        ####### reset every object to start schaduling process (3 steps)

        # 1 # reset truck and assigned  nonCombined node to the truck
        self.TSP_truck_route[0].Launching = 1
        self.TSP_truck_route[-1].Retrieval = 1
        self.TSP_truck_route[0].served_by = self.truck.id
        self.truck.reset()
        self.truck.currentNode = self.TSP_truck_route[0]

        # 2 # reset all drones
        for drone in self.DronesList:
            drone.resetTotally()
            drone.currentNode = self.TSP_truck_route[0]
            if not (drone.currentNode.id in drone.currentschedule):
                drone.currentschedule.append(drone.currentNode.id)

        # 3 # determine a list of the unserved nodes
        NoneServedNodesList = [x for x in self.TSP_truck_route if x.served_by == '']
        ############################################################################333

        while len(NoneServedNodesList) > 0:

            for drone in self.DronesList:
                if drone.currentNode.Launching == 1:
                    drone.resetToStartTrip()

                node = NoneServedNodesList[0]

                if node.shipment_weight > Drone.MaxDronesCapacity:
                    self.truck.travel_serve_Node(node)
                    NoneServedNodesList.pop()

                if node.Retrieval == 1 or node.Launching == 1:
                    ## left side of flowchart
                    if len(drone.currentschedule) < 2:
                        self.truck.travel_serve_Node(node)
                        node.Launching = 1
                    else:
                        truck_time_toNext = self.__getTruckTime(node)
                        temp_sch = drone.currentschedule + [node.id]
                        drone_time_toNext = self.__getDroneCost(drone, temp_sch)
                        # if truck_time < drone_time:
                        #     print("^Truck faster to node {}  --- {} ,{}".format(next_node.id,truck_time,drone_time))
                        hover_time = max((truck_time_toNext - drone_time_toNext), 0)
                        energy, Tt_wt = drone.checkAbility_travel_hover([node], hover_time)

                        if energy >= 0:  # yes
                            self.truck.travel_serve_Node(node)
                            drone.travel_not_serve_Node(node, hover_time)
                            node.Retrieval = 1
                        else:
                            # orange block in  old flowchart
                            if energy == -1:
                                raise Exception("more than drone max capacity to serve node : {}".format(node.id))
                            elif energy == -2:
                                raise Exception("no enough power to serve node : {}".format(node.id))

                            truck_time_toNext = self.__getTruckTime(drone.currentNode)
                            temp_sch = drone.currentschedule
                            drone_time_toNext = self.__getDroneCost(drone, temp_sch)
                            hover_time = max((truck_time_toNext - drone_time_toNext), 0)
                            energy2 = drone.checkAbility_hover(hover_time, 0)
                            if energy2 >= 0:
                                self.truck.travel_serve_Node(drone.currentNode)
                                drone.wait_Truck(hover_time)
                                drone.currentNode.Retrieval = 1
                                self.truck.travel_serve_Node(node)
                                node.Launching = 1
                            else:
                                raise Exception("no enough power to serve node : {}".format(node.id))

                                prev_node = self.__getPrevNode(drone.currentNode)
                                drone.rollback_node(prev_node)
                                prev_node.Retrieval = 1
                                if drone.currentNode != prev_node:
                                    raise Exception("Check your code to serve node : {}".format(node.id))
                                self.truck.travel_serve_Node(drone.currentNode)

                else:
                    ## right side of flowchart
                    energy = drone.checkAbility_serve(node)
                    if energy >= 0:
                        next_node = self.__getNextNode(node)
                        truck_time_toNext = self.__getTruckTime(next_node)
                        temp_sch = drone.currentschedule + [node.id, next_node.id]
                        drone_time_toNext = self.__getDroneCost(drone, temp_sch)
                        hover_time = max((truck_time_toNext - drone_time_toNext), 0)
                        energy2, Tt_wt = drone.checkAbility_travel_hover([node, next_node], hover_time)

                        if energy2 >= 0:
                            drone.travel_serve_Node(node)
                        else:
                            node.Retrieval = 1

                    elif energy == -1:
                        raise Exception("more than drone max capacity to serve node : {}".format(node.id))
                        node.Retrieval = 1
                        pass
                    elif energy == -2:
                        raise Exception("no enough power to serve node : {}".format(node.id))
                        node.Retrieval = 1
                        pass

                # redetermine a list of the unserved nodes
                NoneServedNodesList = [x for x in self.TSP_truck_route if x.served_by == '']
                if len(NoneServedNodesList) == 0:
                    break

        # ensure to move truck to the last node
        final_truck_trip = [node.id for node in self.TSP_truck_route if node.served_by.find('T1') > -1]
        while (final_truck_trip[-1] != self.truck.currentNode.id):
            inx = final_truck_trip.index(self.truck.currentNode.id)
            next_id = final_truck_trip[inx + 1]
            self.truck.travel_serve_Node(self.NodesList[next_id])

        # quality of code checker
        for node in self.TSP_truck_route:
            if node.served_by == '':
                raise Exception("Check your code to serve node : {}".format(node.id))

    def __getNext(self, elem, aList):
        idx = aList.index(elem)
        if idx == len(aList) - 1:
            return None
        else:
            return aList[idx + 1]

    def __getTruckCost(self, final_truck_trip, strat, end):
        truck_trip_cost = 0
        t_node = strat
        t_next_node = None
        while (t_next_node != end):
            t_next_node = self.__getNext(t_node, final_truck_trip)
            travel_time = self.truck.AdjacencyMatrix[t_node][t_next_node]
            service_time = self.NodesList[t_next_node].service_time
            truck_trip_cost = truck_trip_cost + travel_time + service_time
            # print("~Truck trip {} ~~~ {} time={}".format(strat, end, truck_trip_cost))
            t_node = t_next_node

        return truck_trip_cost

    def __getDroneCost(self, drone, Drone_sch):
        drone_trip_cost = 0
        for node in Drone_sch:
            d_next_node = self.__getNext(node, Drone_sch)
            if d_next_node != Drone_sch[-1] and d_next_node != None:
                travel_time = drone.AdjacencyMatrix[node][d_next_node]
                service_time = self.NodesList[d_next_node].service_time
                drone_trip_cost = drone_trip_cost + travel_time + service_time
                # print("Drone trip time {} ~~~ {} time={}".format(node, d_next_node, travel_time + service_time))
            elif d_next_node == Drone_sch[-1]:
                travel_time = drone.AdjacencyMatrix[node][d_next_node]
                drone_trip_cost = drone_trip_cost + travel_time
                # print("~Drone trip time {} ~~~ {} time={}".format(node, d_next_node, travel_time))
        return drone_trip_cost

    def __calc_total_cost(self):
        # print("calc_total_cost function>>>>>>>>>>>>>>>>>>>>>>>>>><<<<<<>>>>>>>")
        final_truck_trip = [node.id for node in self.TSP_truck_route if node.served_by.find('T1') > -1]
        DronesSchedules = self.DronesList[0].schedules
        # final_truck_trip = [0, 7, 3, 6, 8, 14, 16, 15, 17, 25, 22, 21, 26]
        # DronesSchedules = [[0, 1, 4, 7], [7, 5, 3], [3, 2, 6], [6, 9, 8], [8, 10, 11, 14], [14, 12, 19, 15],
        #                    [15, 13, 17], [17, 18, 23, 25], [25, 24, 22], [22, 20, 21]]
        # final_truck_trip =[0, 2, 4, 6, 10, 11]
        # DronesSchedules = [ [0, 1, 2], [2, 5, 3, 6] ,[6, 7, 8, 10], [10, 9, 11] ]
        # calculating total cost of the best schedule
        #############################################
        total_cost = 0
        drone = self.DronesList[0]

        # first Truck only trip
        if DronesSchedules[0][0] != final_truck_trip[0]:
            t_node = final_truck_trip[0]
            end = DronesSchedules[0][0]
            truck_trip_cost = self.__getTruckCost(final_truck_trip, t_node, end)
            # print("last truck only: Truck time={}".format(truck_trip_cost))
            total_cost = total_cost + truck_trip_cost
            # print("{} - {} ".format(t_node,end) )

        for sch in DronesSchedules:
            drone_trip_cost = self.__getDroneCost(drone, sch)
            t_node = sch[0]
            end = sch[-1]
            truck_trip_cost = self.__getTruckCost(final_truck_trip, t_node, end)
            # print("Truck time={}".format(truck_trip_cost))
            # print("time {}\n".format(max(truck_trip_cost, drone_trip_cost)))
            total_cost = total_cost + max(truck_trip_cost, drone_trip_cost)
            next_sch = self.__getNext(sch, DronesSchedules)
            if next_sch == None:
                continue
            if sch[-1] != next_sch[0]:
                # trip trcuk only
                t_node = sch[-1]
                end = next_sch[0]
                truck_trip_cost = self.__getTruckCost(final_truck_trip, t_node, end)
                # print("trip trcuk only: Truck time={}".format(truck_trip_cost))
                total_cost = total_cost + truck_trip_cost

        # last truck only
        if sch[-1] != final_truck_trip[-1]:
            t_node = sch[-1]
            end = final_truck_trip[-1]
            truck_trip_cost = self.__getTruckCost(final_truck_trip, t_node, end)
            total_cost = total_cost + truck_trip_cost
        #### Tamer Printing
        # print ("############################# Temer")
        # self.__getTruckCost(final_truck_trip,final_truck_trip[0],final_truck_trip[-1])

        return total_cost, DronesSchedules, final_truck_trip

    def evaluate_population(self, population):
        return Population(population, self.AdjacencyMatrix)

    def evalute_truck_schedule(self, test):
        self.testlist = test
        total_p = []
        total_p.append(test)
        evaluatedPopulation = self.evaluate_population(total_p)
        return evaluatedPopulation

    def __chromosome2TruckOnlyList(self, chrom, TSP_truck_schedule):
        Truck_only = []
        # Truck_only=[0, 7, 3, 6, 8, 14, 16, 15, 17, 25, 22, 21, 26]
        for i in range(len(chrom)):
            if chrom[i] == 0:
                Truck_only.append(TSP_truck_schedule[i])
            elif TSP_truck_schedule[i] in self.nonCombinedList:
                Truck_only.append(TSP_truck_schedule[i])
        return Truck_only

    def __NodesList2chromosome(self, NodesList):
        chrom = [0] * len(NodesList)

        for node in NodesList:
            if node.served_by.find('T') != -1:
                chrom[node.id] = 0
            else:
                chrom[node.id] = 1

        return chrom

    def evaluate_drones(self, chrom):

        # in chrom 0 for truck 1 for drone
        ## add depot (as start and end node if not added
        if len(self.TSP_truck_schedule) == len(chrom) + 2:
            chrom = list(chrom)
            chrom.insert(0, 0)
            chrom.append(0)

        ## bad chrom id the depot not served by truck
        if chrom[0] != 0 or chrom[-1] != 0:
            total_cost = float('inf')
            DronesSchedules = 'bad chrom'
            Truck_schedule = ''
            return self.TSP_truck_route, total_cost, DronesSchedules, Truck_schedule

        self.__BuildingTruckRoute()

        ## ensure that noncombined nodes is assigned as Retrieval nodes
        Truck_only = self.__chromosome2TruckOnlyList(chrom, self.TSP_truck_schedule)
        for n in self.TSP_truck_route:
            if n.id in Truck_only:
                n.Retrieval = 1
            else:
                pass

        ## start scheduling process
        try:
            ### core step
            self.__BuildingDronesSchedules()
            ### calulate the final cost
            total_cost, DronesSchedules, Truck_schedule = self.__calc_total_cost()
        except Exception as e:
            total_cost = float('inf')
            DronesSchedules = e.args[0]
            Truck_schedule = e.args[0]

        # ch = self.__NodesList2chromosome(self.NodesList)
        # if ch != chrom:
        #     chrom.append(ch)
        #     pass
        return self.TSP_truck_schedule, total_cost, DronesSchedules, Truck_schedule, chrom

    def searchBest_truck_schedule(self, n_population, NodesList):
        np.random.seed(42)
        cities = list(range(len(NodesList)))
        total_p = []
        rest_elements = cities[1:-1]
        print("rest_elements = ", rest_elements)

        for i in range(n_population):
            p_temp_list = [cities[0]]
            for p_element in ((np.random.permutation(rest_elements)).tolist()):
                p_temp_list.append(p_element)
            p_temp_list.append(cities[-1])
            total_p.append(p_temp_list)

        evaluatedPopulation = self.evaluate_population(total_p)
        TSP_truck_schedule = evaluatedPopulation.bestPop[0]
        return TSP_truck_schedule, evaluatedPopulation


def main():
    dataset = 'R101'
    DronesList = ['M']
    # final_truck_trip =[0, 2, 4, 6, 10, 11]
    # DronesSchedules = [ [0, 1, 2], [2, 5, 3, 6] ,[6, 7, 8, 10], [10, 9, 11] ]
    test = [0, 6, 5, 8, 7, 10, 1, 9, 3, 4, 2, 11]
    chrom2 = [0, 1, 1, 0, 1, 1, 0, 1, 1, 0, 1, 0]
    numCities = len(test) - 2
    prob = Problem_v2(numCities, dataset=dataset, DronesList=DronesList, TSP_truck_schedule=test)
    TSP_truck_route, total_cost, DronesSchedules, Truck_schedule, chrom = prob.evaluate_drones(chrom=chrom2)
    fitness = sum([prob.AdjacencyMatrix[test[i], test[i + 1]] for i in range(len(test) - 1)])
    print(fitness)
    # print("\nOutput>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
    printDetails(prob, total_cost)

    # print("Best TSP_Truck_only Schedule is: {}".format(test))
    # print("Best score={} , run time={}".format(round(evaluatedPopulation.bestScore, 2), total_time))
    # print("Best Truck schedule is: {}".format(Truck_schedule))
    # print("Total Drones Schedules: in one list")
    # for d in prob.DronesList:
    #     print("{}: {} ".format(d.id, d.schedules))
    # print("total_cost = ", total_cost)
    # print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")


def printDetails(prob, total_cost, filepath=None, printOnScreen=True):
    summury = []
    summury.append("Number of drones :	{}".format(len(prob.DronesList)))
    summury.append("Number of Nodes(without Depot):	{}".format(len(prob.NodesList) - 2))
    TSP_schedule = [n.id for n in prob.TSP_truck_route]
    Truck_schedule = [n.id for n in prob.TSP_truck_route if n.served_by.find('T') > -1]

    summury.append("TSP schedule is: {}".format(TSP_schedule))
    summury.append("Truck schedule is:\n\t{}".format(Truck_schedule))
    summury.append("Drone Trips:")
    for d in prob.DronesList:
        summury.append("\tDrone {}: {} ".format(d.id, d.schedules))
    summury.append("total_cost = {} \n".format(total_cost))
    summury.append("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
    summury.append("Truck")
    summury.append("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
    summury.append(
        '{:<8}\t{:<8}\t{:<8}\t{:<8}\t{:<8}\t{:<8}'.format('C_Iid', 'Sv_T', 'Arr_T', 'Float', 'Dep_T', 'To_nxt'))
    for k, t in prob.truck.trips.items():
        summury.append('{:<8}\t{:<8}\t{:<8}\t{:<8}\t{:<8}\t{:<8}'.format(t['C_Iid'], t['Sv_T'], t['Arr_T'], t['Float'],
                                                                         t['Dep_T'],
                                                                         t['To_nxt']))

    summury.append("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
    for d in prob.DronesList:
        for k, t in d.trips.items():
            summury.append("\nDrone {} , trip {}".format(d.id, k))
            summury.append("~~~~~~~~~~~~~~~~~~~")
            summury.append('{:<8}\t{:<8}\t{:<8}\t{:<8}\t{:<8}\t{:<8}\t{:<8}\t{:<8}\t{:<8}\t{:<8}'.format('C_id', 'SV_T',
                                                                                                         'Arr_T',
                                                                                                         'Float',
                                                                                                         'Dep_T',
                                                                                                         'To_nxt',
                                                                                                         'Arr_Bat',
                                                                                                         'Dep_Bat',
                                                                                                         'Tt_wt', 'Q'))
            for k1, t1 in t.items():
                if k1 == list(t.keys())[-1]:
                    break
                summury.append(
                    '{:<8}\t{:<8}\t{:<8}\t{:<8}\t{:<8}\t{:<8}\t{:<8}\t{:<8}\t{:<8}\t{:<8}'.format(k1, t1['SV_T'],
                                                                                                  round(t1['Arr_T'], 2),
                                                                                                  t1['Float'],
                                                                                                  t1['Dep_T'],
                                                                                                  t1['To_nxt'],
                                                                                                  t1['Arr_Bat'],
                                                                                                  t1['Dep_Bat'],
                                                                                                  t1['Tt_wt'],
                                                                                                  t1['Q']))

            summury.append('{:<8}\t{:<8}\t{:<8}\t{:<8}\t'.format('RN#C_id', 'D_arr_T', 'V_arr_T', 'D_ar_bt'))
            summury.append(
                '{:<8}\t{:<8}\t{:<8}\t{:<8}\t'.format(k1, t[k1]['D_arr_T'], t[k1]['V_arr_T'], t[k1]['D_ar_bt']))
    if filepath == None:
        filepath = 'result_Details.txt'

    with open(filepath, 'w') as file:
        for s in summury:
            if printOnScreen:
                print(s)
            file.write(s + "\n")
    #
    # import matplotlib.pyplot as plt
    # fig, ax = plt.subplots()
    # ax.set(xlim=(0, 100), ylim=(0, 100))
    #
    # for idx in range(len(Truck_schedule) - 1):
    #     n1 = prob.NodesList[Truck_schedule[idx]]
    #     n2 = prob.NodesList[Truck_schedule[idx + 1]]
    #     # plt.arrow(n1.x,n1.y,n2.x-n1.x,n2.y-n1.y,width=.06,head_width=0.4, head_length=0.4,arrowprops={'arrowstyle': '->', 'lw': 4, 'color': 'blue'})
    #     ax.annotate('', xy=(n2.x, n2.y), xytext=(n1.x, n1.y),
    #                 arrowprops={'arrowstyle': '->', 'lw': 1, 'color': 'blue'},
    #                 va='center')

    # for n in prob.NodesList:
    #     col = 'Red'
    #     if n.served_by == 'T1':
    #         col = 'Green'
    #     cir = plt.Circle((n.x, n.y), 1, fill=True, edgecolor=col, facecolor=col, alpha=0.2)
    #     plt.text(n.x, n.y, str(n.id), size=10)
    #     # axes.set_aspect(1)
    #     ax.add_artist(cir)
    # plt.show()


if __name__ == "__main__":
    main()
