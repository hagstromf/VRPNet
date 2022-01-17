import torch
import torch.nn.functional as F
import numpy as np

import utils.beam_search as beam_search


def tour_nodes_to_W(nodes, num_nodes):
    """Helper function to convert ordered list of tour nodes to edge adjacency matrix.
    """
    W = torch.zeros((num_nodes, num_nodes))
    for idx in range(len(nodes) - 1):
        i = int(nodes[idx])
        j = int(nodes[idx + 1])
        W[i][j] = 1
    # Add final connection of tour in edge target
    W[int(nodes[-1])][int(nodes[0])] = 1
    
    # Make sure there are no loops on the depot
    W[0][0] = 0
    return W    


def tour_nodes_to_tour_len(nodes, W_values):
    """Helper function to calculate tour length from ordered list of tour nodes.
    """
    tour_len = 0
    for idx in range(len(nodes) - 1):
        i = nodes[idx]
        j = nodes[idx + 1]
        tour_len += W_values[i][j]

    # Add final connection of tour in edge target
    tour_len += W_values[j][nodes[0]]
    return tour_len


def is_valid_tour(tour, num_nodes):
    """Sanity check: tour visits all nodes given.
    """
    if torch.is_tensor(tour):
        return all(elem in tour for elem in list(range(num_nodes))) and torch.count_nonzero(tour) == num_nodes-1
    else:
        return all(elem in tour for elem in list(range(num_nodes))) and np.count_nonzero(tour) == num_nodes-1


def W_to_tour_nodes(W):
    """Helper function to convert edge adjacency matrix to an ordered list of tour nodes.
    """
    depot_node = 0
    subtour_starts = np.argwhere(W[depot_node] == 1).flatten()
    tour = [depot_node]

    for idx in list(subtour_starts):
        i = depot_node
        j = idx
        while j != depot_node:
            tour.append(j)
            i = j
            j = np.argwhere(W[i] == 1).flatten()[0]
        tour.append(depot_node)

    return tour

def get_subtours(tour):
    """Get all sub-routes in an ordered list of tour nodes"""
    subtours = []
    depot = tour[0]

    curr_subtour = [depot]
    for node in tour[1:]:
        curr_subtour.append(node)
        if node == depot:
            subtours.append(curr_subtour)
            curr_subtour = [depot]

    if len(curr_subtour) > 1:
        subtours.append(curr_subtour)

    return subtours

def get_longest_subtour(tour, W_values):
    """Get the longest sub-route in an ordered list of tour nodes"""
    subtours = get_subtours(tour)
    longest_tour = subtours[0]
    longest_len = tour_nodes_to_tour_len(longest_tour, W_values)

    for sub in subtours[1:]:
        sub_len = tour_nodes_to_tour_len(sub, W_values)
        if sub_len > longest_len:
            longest_tour = sub
            longest_len = sub_len

    return longest_tour, longest_len


def beamsearch_tour_nodes(y_pred_edges, x_edges_values, beam_size, batch_size, num_nodes, num_vehicles, device, dtypeFloat=torch.FloatTensor, dtypeLong=torch.LongTensor, random_start=False):
    """
    Performs beamsearch procedure on edge prediction matrices and returns possible VRP tours.

    Args:
        y_pred_edges: Predictions for edges (batch_size, num_nodes, num_nodes, 2)
        x_edges_values: Input edge distance matrix (batch_size, num_nodes, num_nodes)
        beam_size: Beam size
        batch_size: Batch size
        num_nodes: Number of nodes in VRP tours
        num_vehicles: Number of available vehicles
        device: Current computing device
        dtypeFloat: Float data type (for GPU/CPU compatibility)
        dtypeLong: Long data type (for GPU/CPU compatibility)
        random_start: Flag for using fixed (at node 0) vs. random starting points for beamsearch

    Returns:
        tours: VRP solution tours in terms of node ordering (batch_size, num_nodes)
        lens: Length of longest sub-tour in VRP solution (batch_size, num_nodes)
    """

    # Compute softmax over edge prediction matrix
    y = F.softmax(y_pred_edges, dim=3)  # B x V x V x voc_edges
    # Consider the second dimension only
    y = y[:, :, :, 1]  # B x V x V

    # Perform beamsearch
    beamsearch = beam_search.Beamsearch(beam_size, batch_size, num_nodes, num_vehicles, device, dtypeFloat, dtypeLong, random_start)

    trans_probs = y.gather(1, beamsearch.get_current_state()).to(beamsearch.device)
    for step in range(num_nodes - 1 + num_vehicles - 1):
        beamsearch.advance(trans_probs)
        trans_probs = y.gather(1, beamsearch.get_current_state())
    # Find TSP tour with highest probability among beam_size candidates
    ends = torch.zeros(batch_size, 1).type(dtypeLong).to(beamsearch.device)
    tours =  beamsearch.get_hypothesis(ends)
    lens = [0] * len(tours)
    for idx in range(len(tours)):
        _, lens[idx] = get_longest_subtour(tours[idx].cpu().numpy(),
                                           x_edges_values[idx].cpu().numpy())
    return tours, lens



def beamsearch_tour_nodes_shortest(y_pred_edges, x_edges_values, beam_size, batch_size, num_nodes, num_vehicles, device,
                                   dtypeFloat=torch.FloatTensor, dtypeLong=torch.LongTensor, random_start=False):
    """
    Performs beamsearch procedure on edge prediction matrices and returns possible VRP tours.
    Final predicted tour is the one with the shortest longest sub-tour length.
    (Standard beamsearch returns the one with the highest probability and does not take length into account.)

    Args:
        y_pred_edges: Predictions for edges (batch_size, num_nodes, num_nodes, 2)
        x_edges_values: Input edge distance matrix (batch_size, num_nodes, num_nodes)
        beam_size: Beam size
        batch_size: Batch size
        num_nodes: Number of nodes in VRP tours
        num_vehicles: Number of available vehicles
        device: Current computing device
        dtypeFloat: Float data type (for GPU/CPU compatibility)
        dtypeLong: Long data type (for GPU/CPU compatibility)
        random_start: Flag for using fixed (at node 0) vs. random starting points for beamsearch

    Returns:
        shortest_tours: VRP solution tours in terms of node ordering (batch_size, num_nodes)
        shortest_lens: Length of longest sub-tour in VRP solution (batch_size, num_nodes)
    """

    # Compute softmax over edge prediction matrix
    y = F.softmax(y_pred_edges, dim=3)  # B x V x V x voc_edges
    # Consider the second dimension only
    y = y[:, :, :, 1]  # B x V x V

    # Perform beamsearch
    beamsearch = beam_search.Beamsearch(beam_size, batch_size, num_nodes, num_vehicles, device, dtypeFloat, dtypeLong, random_start)

    trans_probs = y.gather(1, beamsearch.get_current_state()).to(beamsearch.device)
    for step in range(num_nodes - 1 + num_vehicles - 1):
        beamsearch.advance(trans_probs)
        trans_probs = y.gather(1, beamsearch.get_current_state())
    # Initially assign shortest_tours as most probable tours i.e. standard beamsearch
    ends = torch.zeros(batch_size, 1).type(dtypeLong).to(beamsearch.device)
    shortest_tours = beamsearch.get_hypothesis(ends)
    
    # Compute current tour lengths
    shortest_lens = [1e6] * len(shortest_tours)
    for idx in range(len(shortest_tours)):
        _, shortest_lens[idx] = get_longest_subtour(shortest_tours[idx].cpu().numpy(),
                                                    x_edges_values[idx].cpu().numpy())
    # Iterate over all positions in beam (except position 0 --> highest probability)
    for pos in range(1, beam_size):
        ends = pos * torch.ones(batch_size, 1).type(dtypeLong).to(beamsearch.device)  # New positions
        hyp_tours = beamsearch.get_hypothesis(ends)
        for idx in range(len(hyp_tours)):
            hyp_nodes = hyp_tours[idx].cpu().numpy()
            _, hyp_len = get_longest_subtour(hyp_nodes, x_edges_values[idx].cpu().numpy())
            # Replace tour in shortest_tours if new length is shorter than current best
            if hyp_len < shortest_lens[idx] and is_valid_tour(hyp_nodes, num_nodes):
                shortest_tours[idx] = hyp_tours[idx]
                shortest_lens[idx] = hyp_len
    return shortest_tours, shortest_lens