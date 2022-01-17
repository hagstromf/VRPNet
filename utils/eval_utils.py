import torch
import matplotlib.pyplot as plt
from utils.tour_utils import *


def get_total_route_length(dist_mat, adj_mat):
    """Gets the total route length described by the adjacency matrix.

    Args:
        dist_mat of shape (n_nodes) : Distance matrix of the VRP.
        adj_mat of shape (batch_size, n_nodes_per_batch, n_nodes_per_batch) : Adjacency matrix of the solution graph.

    Returns:
        The combined length of the routes.
    """
    dist_mat = dist_mat.reshape(adj_mat.shape)
    return torch.sum(dist_mat * adj_mat, dim=(1, 2))


def average_beamsearch_optimality_gap(model, dataloader, beam_size, num_vehicles, device, shortest=True, n_iter=-1, use_preds=True):
    '''Returns the average optimality gap of the predicted solutions by beamsearch to the target solutions.

    '''
    with torch.no_grad():
        model.eval()
        pred_route_lens = torch.zeros(len(dataloader)).to(device)
        opt_gaps = torch.zeros(len(dataloader)).to(device)
        for i, (node_inputs, edge_inputs, targets, dist_mat, target_lengths, src_ids, dst_ids) in enumerate(dataloader):
            outputs = model(node_inputs, edge_inputs, src_ids, dst_ids)  
            
            batch_size, num_nodes, _ = node_inputs.shape
            if use_preds:
                y_preds = outputs[n_iter].reshape(batch_size, num_nodes, num_nodes, 2).to(device)
            else:
                y_preds = torch.ones(batch_size, num_nodes, num_nodes, 2).to(device)
                
            x_edges = dist_mat.reshape(batch_size, num_nodes, num_nodes).to(device)
            
            if shortest:
                tours, longest_subtour_lens = beamsearch_tour_nodes_shortest(y_preds, x_edges, beam_size, batch_size, num_nodes, num_vehicles, device)
            else:
                tours, longest_subtour_lens = beamsearch_tour_nodes(y_preds, x_edges, beam_size, batch_size, num_nodes, num_vehicles, device)

            longest_subtour_lens = torch.tensor(longest_subtour_lens).to(device)
            pred_route_lens[i] = torch.sum(longest_subtour_lens)

            opt_gap = torch.sum(longest_subtour_lens / target_lengths - 1)
            opt_gaps[i] = opt_gap

        return 1/(batch_size*len(dataloader)) * torch.sum(opt_gaps), 1/(batch_size*len(dataloader)) * torch.sum(pred_route_lens)


def accuracy_of_predicted_routes(model, dataloader):
    '''Computes the accuracy of the predicted edge labels. This isn't really a useful
    measure, since even if the model predicts zeros for every node, the accuracy will be high 
    (over 90%). This is because the activated nodes (1) are so few relative to the total amount
    of nodes.

    Args:
        model : VRPNet model
        dataloader : Dataloader object

    Returns:
        accuracy : The classification accuracy the VRPNet model
    '''
    with torch.no_grad():
        model.eval()
        n_corr_preds = 0    # Number of correct predictions
        n_preds = 0         # Total number of predictions
        for node_inputs, edge_inputs, targets, dist_mat, route_lengths, src_ids, dst_ids in iter(dataloader):
            
            outputs = model(node_inputs, edge_inputs, src_ids, dst_ids)
            
            final_output = outputs[-1].argmax(dim=2).flatten()
            
            n_preds += final_output.shape[0]
            n_corr_preds += torch.sum(final_output == targets)
            
        return n_corr_preds / n_preds


def evaluate_iteration(model, dataloader, beam_size, num_vehicles, device, filename="", title="Average optimality gap of each iteration of VRPNet", save=False):
    '''Evaluates and plots the performance of each iteration in the forward function of VRPNet over one mini-batch of data.

    Args:
        model : VRPNet model
        dataloader : Dataloader object
        beam_size : Desired beam size
        num_vehicles : The number of vehicles available
        device : Current computing device
        filename : Name of image file of the plot
        title : Desired title of the plot
        save : Defines if you wish the generated plot to be saved into plots/ folder

    Returns:
        avg_opt_gaps : A tensor array of the average optimality gaps at each iteration
    '''
    with torch.no_grad():
        model.eval()
        node_inputs, edge_inputs, targets, dist_mat, target_lengths, src_ids, dst_ids = next(iter(dataloader))
        
        outputs = model(node_inputs, edge_inputs, src_ids, dst_ids)  
            
        batch_size, num_nodes, _ = node_inputs.shape

        n_iters = model.n_iters
        avg_opt_gaps = torch.zeros(n_iters)
        for i in range(n_iters):
            y_preds = outputs[i].reshape(batch_size, num_nodes, num_nodes, 2).to(device)
            x_edges = dist_mat.reshape(batch_size, num_nodes, num_nodes).to(device)
            tours, longest_subtour_lens = beamsearch_tour_nodes_shortest(y_preds, x_edges, beam_size, batch_size, num_nodes, num_vehicles, device)

            longest_subtour_lens = torch.tensor(longest_subtour_lens).to(device)

            avg_opt_gap = (1/batch_size) * torch.sum(longest_subtour_lens / target_lengths - 1)
            avg_opt_gaps[i] = avg_opt_gap
            print(avg_opt_gap)
        
        plt.figure()
        plt.title(title)
        plt.plot(range(1, n_iters+1), avg_opt_gaps)
        plt.xlabel("Iteration")
        plt.ylabel("Average optimality gap after beam search")
        if save:
            plt.savefig("plots/" + filename + ".png")
        plt.show()
        return avg_opt_gaps

def evaluate_beam_size(model, dataloader, beam_sizes, num_vehicles, device, n_iter=-1,  filename="", title="Average optimality gap at different beam sizes", save=False):
    '''Evaluates and plots the performance at different beam sizes.

    Args:
        model : VRPNet model
        dataloader : Dataloader object
        beam_sizes : The beam sizes to evaluate at
        num_vehicles : The number of vehicles available
        device : Current computing device
        n_iter : Which iteration of the network to use 
        filename : Name of image file of the plot
        title : Desired title of the plot
        save : Defines if you wish the generated plot to be saved into plots/ folder

    Returns:
        avg_opt_gaps : A tensor array of the average optimality gaps at each beam size
    '''
    with torch.no_grad():
        model.eval()
        avg_opt_gaps = torch.zeros(len(beam_sizes))
        for i, beam_size in enumerate(beam_sizes):
            avg_opt_gap, avg_pred_tour_len = average_beamsearch_optimality_gap(model, dataloader, beam_size, num_vehicles, device, n_iter=n_iter)
            avg_opt_gaps[i] = avg_opt_gap
            print(avg_opt_gap)
        
        plt.figure()
        plt.title(title)
        plt.plot(beam_sizes, avg_opt_gaps)
        plt.xlabel("Beam size")
        plt.ylabel("Average optimality gap after beam search")
        if save:
            plt.savefig("plots/" + filename + ".png")
        plt.show()
        return avg_opt_gaps
