import random
import numpy as np
import scipy.spatial

DEPOT_TOKEN = 100

def generate_random_instance(num_nodes=20, grid=(1000, 1000), p=2):
    """Generates a random instance of the VRP.

    Args:
        n_nodes : Number of nodes in the VRP graph.
        grid : Tuple defining the grid of possible coordinate values.
        p : The p-norm used as the distance measure between coordinates. Default
            is 2-norm (Euclidean).

    Returns:
        coords of shape (n_nodes, 2) : The coordinates of the nodes
        dist_mat of shape (n_nodes, n_nodes) : Distance matrix
    """
    x_range = range(grid[0])
    y_range = range(grid[1])

    x_sample = random.sample(x_range, num_nodes)
    y_sample = random.sample(y_range, num_nodes)

    coords = list(zip(x_sample, y_sample))

    dist_mat = scipy.spatial.distance_matrix(coords, coords, p=p).round().astype(int)
    return coords, dist_mat

def create_data_model(num_nodes=20, num_vehicles=5):
    """Stores the data for the problem."""
    data = {}
    coords, data['distance_matrix'] = generate_random_instance(num_nodes=num_nodes)
    tokens = np.zeros((len(coords), 1), dtype=int)
    tokens[0] = DEPOT_TOKEN
    inputs = np.concatenate((coords, tokens), axis=1)
    data['num_vehicles'] = num_vehicles
    data['depot'] = 0
    return inputs, data


def print_solution(data, manager, routing, solution):
    """Prints solution on console."""
    print(f'Objective: {solution.ObjectiveValue()}')
    max_route_distance = 0
    for vehicle_id in range(data['num_vehicles']):
        index = routing.Start(vehicle_id)
        plan_output = 'Route for vehicle {}:\n'.format(vehicle_id)
        route_distance = 0
        while not routing.IsEnd(index):
            plan_output += ' {} -> '.format(manager.IndexToNode(index))
            previous_index = index
            index = solution.Value(routing.NextVar(index))
            route_distance += routing.GetArcCostForVehicle(
                previous_index, index, vehicle_id)
        plan_output += '{}\n'.format(manager.IndexToNode(index))
        plan_output += 'Distance of the route: {}m\n'.format(route_distance)
        print(plan_output)
        max_route_distance = max(route_distance, max_route_distance)
    print('Maximum of the route distances: {}m'.format(max_route_distance))


def get_solution_nodes(solution, routing, manager):
    """Get the VRP solution in node format. In the node format, every node is described
       by two features: the first descibes which route the node belongs and the second one
       describes its ordering in the corresponding route.

    Args:
        solution : Assignment object. 
        routing : RoutingModel object.
        manager : RoutingIndexManager object.

    Returns:
        nodes of shape (n_nodes, 2) : VRP solution in node format
    """
    nodes = np.zeros((manager.GetNumberOfNodes(), 2), dtype=int)
    for route_nbr in range(routing.vehicles()):
        index = routing.Start(route_nbr)
        order_nbr = 1

        while not routing.IsEnd(index):
            index = solution.Value(routing.NextVar(index))
            if manager.IndexToNode(index):
                nodes[manager.IndexToNode(index)] = (route_nbr+1, order_nbr)
                order_nbr += 1
            else:
                nodes[manager.IndexToNode(index)] = (0, 0)

    return np.asarray(nodes)


def get_adjacency_from_nodes(nodes):
    """Get the VRP solution graph as an adjaceny matrix from node format.

    Args:
        nodes of shape (n_nodes, 2) : VRP solution in node format

    Returns:
        adj_mat of shape (n_nodes, n_nodes) : Adjacency matrix of the solution graph.
    """

    n_nodes = nodes.shape[0]
    adj_mat = np.zeros((n_nodes, n_nodes), dtype=int)

    n_routes = np.max(nodes[:, 0])
    for route in range(1, n_routes+1):
        src_node = 0
        nodes_on_route = np.argwhere(nodes[:, 0] == route).flatten()
        for order_nbr in range(1, len(nodes_on_route)+1):
            dst_idx = np.argwhere(nodes[nodes_on_route][:, 1] == order_nbr)[0, 0]
            dst_node = nodes_on_route[dst_idx]

            adj_mat[src_node, dst_node] = 1
            src_node = dst_node
        
        adj_mat[src_node, 0] = 1

    return adj_mat

def get_total_route_length(dist_mat, adj_mat):
    """Gets the total route length described by the adjacency matrix.

    Args:
        dist_mat of shape (n_nodes) : Distance matrix of the VRP.
        adj_mat of shape (batch_size, n_nodes_per_batch, n_nodes_per_batch) : Flattened Adjacency matrix of the solution graph.

    Returns:
        The combined length of the routes.
    """
    
    dist_mat = dist_mat.reshape(adj_mat.shape)
    return np.sum(dist_mat * adj_mat)   
