import os
import torch
import sys
sys.path.insert(0, 'C:/Users/hagst/Bachelors_thesis')

from utils.gen_utils import *

from ortools.constraint_solver import routing_enums_pb2
from ortools.constraint_solver import pywrapcp

def generate_and_solve():
    """Generates a random VRP and solves it.

    References:
        Setting up a VRP instance with OR-tools : https://developers.google.com/optimization/routing/vrp
    
    Returns:
        distance_matrix of shape (n_nodes, n_nodes) : The randomly generated distance matrix 
                                                      for the VRP.
        routes of shape (n_vehicles, n_nodes) : Route matrix giving the ordering of nodes
                                                on the routes of the vehicles. 
        adj_mat of shape (n_nodes, n_nodes) : Adjacency matrix of the solution graph.
        route_length : Optimal value of the objective function.
    """
    # Instantiate the data problem.
    coords, data = create_data_model(num_nodes=20)

    # Create the routing index manager.
    manager = pywrapcp.RoutingIndexManager(len(data['distance_matrix']),
                                           data['num_vehicles'], data['depot'])

    # Create Routing Model.
    routing = pywrapcp.RoutingModel(manager)


    # Create and register a transit callback.
    def distance_callback(from_index, to_index):
        """Returns the distance between the two nodes."""
        # Convert from routing variable Index to distance matrix NodeIndex.
        from_node = manager.IndexToNode(from_index)
        to_node = manager.IndexToNode(to_index)
        return data['distance_matrix'][from_node][to_node]

    transit_callback_index = routing.RegisterTransitCallback(distance_callback)

    # Define cost of each arc.
    routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)

    # Add Distance constraint.
    dimension_name = 'Distance'
    routing.AddDimension(
        transit_callback_index,
        0,  # no slack 
        10000,  # vehicle maximum travel distance
        True,  # start cumul to zero
        dimension_name)
    distance_dimension = routing.GetDimensionOrDie(dimension_name)
    distance_dimension.SetGlobalSpanCostCoefficient(100)

    # Setting first solution heuristic.
    search_parameters = pywrapcp.DefaultRoutingSearchParameters()
    search_parameters.first_solution_strategy = (
        routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC)
    
    # Setting local search metaheuristic
    search_parameters.local_search_metaheuristic = (
    routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH)  
    search_parameters.time_limit.seconds = 1
    #search_parameters.log_search = True

    # Solve the problem.
    solution = routing.SolveWithParameters(search_parameters)

    # Return coordinates, distance matrix, node solutions, adjacency matrix and route length for storage.
    if solution:
        #print_solution(data, manager, routing, solution)
        nodes = get_solution_nodes(solution, routing, manager)

        adj_mat = get_adjacency_from_nodes(nodes)

        route_length = get_total_route_length(data['distance_matrix'], adj_mat)
        
        return coords, data['distance_matrix'], nodes, adj_mat, route_length
    else:
        print('No solution found !')


def main():
    filename = input("Give name of desired file.\n")
    path = "data/" + filename

    generate = input("Do you wish to generate new data points? (y/n) \n")

    if generate == "y":
        it = int(input("How many datapoints to generate?\n"))
        start = 0
        if os.path.isfile(path):
            dataset = torch.load(path)
            coords_tensor = dataset[:][0]
            dist_mat_tensor = dataset[:][1]
            nodes_tensor = dataset[:][2]
            adj_mat_tensor = dataset[:][3]
            route_length_tensor = dataset[:][4]
        else:
            coords, dist_mat, nodes, adj_mat, route_length = generate_and_solve()

            coords_tensor = torch.tensor([coords], dtype=torch.int32)
            dist_mat_tensor = torch.tensor([dist_mat], dtype=torch.int32)
            nodes_tensor = torch.tensor([nodes], dtype=torch.int32)
            adj_mat_tensor = torch.tensor([adj_mat], dtype=torch.int32)
            route_length_tensor = torch.tensor([route_length], dtype=torch.int32)

            dataset = torch.utils.data.TensorDataset(coords_tensor,
                                                     dist_mat_tensor,
                                                     nodes_tensor,
                                                     adj_mat_tensor,
                                                     route_length_tensor)
            torch.save(dataset, path)

            start += 1

        for i in range(start, it):
            coords, dist_mat, nodes, adj_mat, route_length = generate_and_solve()

            coords_tensor = torch.cat([coords_tensor, torch.tensor([coords], dtype=torch.int32)], dim=0)
            dist_mat_tensor = torch.cat([dist_mat_tensor, torch.tensor([dist_mat], dtype=torch.int32)], dim=0)
            nodes_tensor = torch.cat([nodes_tensor, torch.tensor([nodes], dtype=torch.int32)], dim=0)
            adj_mat_tensor = torch.cat([adj_mat_tensor, torch.tensor([adj_mat], dtype=torch.int32)], dim=0)
            route_length_tensor = torch.cat([route_length_tensor, torch.tensor([route_length], dtype=torch.int32)], dim=0)

            dataset = torch.utils.data.TensorDataset(coords_tensor,
                                                        dist_mat_tensor,
                                                        nodes_tensor,
                                                        adj_mat_tensor,
                                                        route_length_tensor)
            torch.save(dataset, path)

            if i % 10 == 0:
                print(f"{i} data points generated.\n")
    else:
        dataset = torch.load(path)
        coords_data = dataset[:][0]
        dist_data = dataset[:][1]
        nodes_data = dataset[:][2]
        adj_mat_data = dataset[:][3]
        route_length_data = dataset[:][4]
        
        print(dataset)
        print(f"Shape of coordinate data: {coords_data.shape}")
        print(f"Shape of distance matrix data: {dist_data.shape}")
        print(f"Shape of tour nodes data: {nodes_data.shape}")
        print(f"Shape of target adjacency matrix data: {adj_mat_data.shape}")
        print(f"Shape of total route length data: {route_length_data.shape}")
    
        
        
if __name__ == '__main__':
    main()

