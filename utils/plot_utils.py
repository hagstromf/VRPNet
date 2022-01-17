import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import torch
import random

from utils.eval_utils import * 


def draw_target_and_prediction(target, pred, coords, target_length, pred_length, path='', save=False):
    """Plots a graph of a target solution and its corresponding VRPNet solution"""
    G_target = nx.from_numpy_matrix(target)
    G_pred = nx.from_numpy_matrix(pred)
    pos = dict(zip(range(len(coords)), coords.tolist()))
    colors = ['r'] + ['b'] * (len(coords) - 1)  # Red for 0th node, blue for others
    
    fig = plt.figure(figsize=(14, 7))

    ax1 = fig.add_subplot(121)
    nx.draw_networkx(G_target, pos=pos, node_color=colors, node_size=50, with_labels=False) 
    ax1.title.set_text(f"Target route: {target_length}")

    ax2 = fig.add_subplot(122)
    nx.draw_networkx(G_pred, pos=pos, node_color=colors, node_size=50, with_labels=False)
    ax2.title.set_text(f"Beamsearch prediction: {pred_length}")

    if save:
        plt.savefig(path)
    plt.show()


def draw_example_tours(model, dataloader, beam_size, num_vehicles, device, shortest=True, n_iter=-1, figname='', save=False):
    """Plots a few example target and VRPNet solution pairs"""
    with torch.no_grad():
        model.eval()

        node_inputs, edge_inputs, targets, dist_mat, route_lengths, src_ids, dst_ids = next(iter(dataloader))
        outputs = model(node_inputs, edge_inputs, src_ids, dst_ids)  
        
        batch_size, num_nodes, _ = node_inputs.shape
        y_preds = outputs[n_iter].reshape(batch_size, num_nodes, num_nodes, 2).to(device)     
        x_edges = dist_mat.reshape(batch_size, num_nodes, num_nodes).to(device)
        
        
        if shortest:
            tours, longest_subtour_lens = beamsearch_tour_nodes_shortest(y_preds, x_edges, beam_size, batch_size, num_nodes, num_vehicles, device)
        else:
            tours, longest_subtour_lens = beamsearch_tour_nodes(y_preds, beam_size, batch_size, num_nodes, num_vehicles, device)
        

        final_output = torch.zeros((batch_size, num_nodes, num_nodes)).to(device)
        targets = targets.reshape(batch_size, num_nodes, num_nodes).cpu().numpy()
        target_longest_subtour_lens = []
        for idx in range(batch_size):
            assert is_valid_tour(tours[idx], num_nodes), f"Tour {tours[idx]} at index {idx} not valid! y_preds {F.softmax(y_preds, dim=3)}"
            adj_mat = tour_nodes_to_W(tours[idx], num_nodes)
            final_output[idx] = adj_mat

            target_tour = W_to_tour_nodes(targets[idx])
            assert is_valid_tour(target_tour, num_nodes), f"Tour {target_tour} at index {idx} not valid! y_preds {F.softmax(y_preds, dim=3)}"
            _, longest_subtour_len = get_longest_subtour(target_tour, x_edges[idx].cpu().numpy())
            target_longest_subtour_lens.append(longest_subtour_len)

        indices = random.sample(range(batch_size), 3)
        final_output = final_output.cpu().numpy()
        for i, idx in enumerate(indices):
            path = "plots/" + figname + "_" + str(i) + ".png"
            draw_target_and_prediction(targets[idx], final_output[idx], node_inputs[idx, :, :2], target_longest_subtour_lens[idx], int(longest_subtour_lens[idx]), path, save)


def plot_batch_loss(batch_losses, name, batch_size, n_iter=15, title='Progression of batch loss', save=False):
    samples = np.array(range(1, len(batch_losses) + 1)) * batch_size * n_iter
    plt.figure()
    plt.title(title)
    plt.plot(samples, batch_losses, label='Loss')
    plt.xlabel('Number of samples')
    plt.ylabel('Loss')
    if save:
        plt.savefig("plots/" + name + ".png")
    plt.show()


def plot_loss(losses, name, title='Progression of loss', save=False):
    epochs = range(1, len(losses)+1)
    plt.figure()
    plt.title(title)
    plt.plot(epochs, losses, label='Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    if save:
        plt.savefig("plots/" + name + ".png")
    plt.show()


def plot_average_optimality_gap(avg_opt_gaps, name, title='Progression of average optimality gap', save=False):
    epochs = range(1, len(avg_opt_gaps)+1)
    plt.figure()
    plt.title(title)
    plt.plot(epochs, avg_opt_gaps, label='Average Optimality Gap')
    plt.xlabel('Epoch')
    plt.ylabel('Average optimality gap')
    if save:
        plt.savefig("plots/" + name + ".png")
    plt.show()


def plot_config_loss(res_dict, title, name='', save=False):
    configs = list(res_dict.keys())
    epochs = range(len(res_dict[configs[0]]))

    plt.figure()
    plt.title(title)

    for i in range(len(configs)):
        plt.plot(epochs, res_dict[configs[i]], label=configs[i])

    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    if save:
        plt.savefig("plots/" + name + ".png")
    plt.show()

def plot_config_avg_opt_gap(res_dict, title, name='', save=False):
    configs = list(res_dict.keys())
    epochs = range(len(res_dict[configs[0]]))

    plt.figure()
    plt.title(title)

    for i in range(len(configs)):
        plt.plot(epochs, res_dict[configs[i]], label=configs[i])

    plt.xlabel('Epoch')
    plt.ylabel('Average optimality gap')
    plt.legend()

    if save:
        plt.savefig("plots/" + name + ".png")
    plt.show()

