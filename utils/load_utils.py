import os
import torch
import itertools
import pickle
import numpy as np

from torch.utils.data import random_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import kneighbors_graph
from scipy.sparse import lil_matrix
from sklearn.utils.class_weight import compute_class_weight

from utils.eval_utils import W_to_tour_nodes, get_longest_subtour, is_valid_tour

SEED = 136

def graph_edges_fc(n_nodes, device):
    """Generates arrays of source and destination nodes for each edge in the input graph."""
    nodes = range(n_nodes)
    edges = list(itertools.product(nodes, nodes))
    
    for elem in edges:
        if elem[0] == elem[1]:
            edges.remove(elem)

    edges = torch.tensor(edges, dtype=torch.long).to(device)
    src_ids = edges[:, 0]
    dst_ids = edges[:, 1]
    
    return src_ids, dst_ids

def load_data(filename, batch_size, device, test_batch_size=1000, test_split=0.2, include_token=True):
    """Loads training and test data into dataloader objects."""
    dataset = torch.load(filename)
    
    outputs = dataset[:][3].flatten().numpy()
    classes = np.unique(outputs)
    
    class_weights = compute_class_weight('balanced', classes=classes, y=outputs)
    class_weights = torch.tensor(class_weights, dtype=torch.float).to(device)
    
    dataset_size = len(dataset)
    test_size = int(test_split * dataset_size)
    train_size = dataset_size - test_size


    targets = dataset[:][3]
    dist_mat = dataset[:][1]
    n_nodes = targets.shape[2]

    target_lengths = torch.zeros(dataset_size).to(device)
    for idx in range(dataset_size):
        target_tour = W_to_tour_nodes(targets[idx].cpu().numpy())
        assert is_valid_tour(target_tour, n_nodes), f"Tour {target_tour} at index {idx} not valid!"
        _, longest_subtour_len = get_longest_subtour(target_tour, dist_mat[idx].cpu().numpy())
        target_lengths[idx] = longest_subtour_len

    dataset[:][4][:] = target_lengths


    def collate(list_of_samples):
    
        inputs, dist_mat, nodes, adj_mat, target_lengths = zip(*list_of_samples)
        
        n_nodes = len(inputs[0])
        batch_size = len(inputs)

        if include_token:
            node_inputs = torch.stack(inputs).type(dtype=torch.float).to(device) 
            node_inputs[:, :, :2] = node_inputs[:, :, :2] / 1000
        else:
            node_inputs = torch.stack(inputs).type(dtype=torch.float).to(device)
            node_inputs = node_inputs[:, :, :2] / 1000  # Leave out depot tokens from the generated data 

        targets = torch.cat(adj_mat).type(torch.LongTensor).reshape(batch_size, n_nodes, n_nodes).to(device)
        targets[:, 0, 0] = 0

        dist_mat = torch.cat(dist_mat).to(device)
        
        dist_mat_scaled = StandardScaler().fit_transform(dist_mat.clone().cpu())
        dist_mat_scaled = torch.tensor(dist_mat_scaled, dtype=torch.float).reshape(-1, n_nodes, n_nodes).to(device)

        target_lengths = torch.tensor(target_lengths).to(device)
        dist_mat = dist_mat.reshape(batch_size, n_nodes, n_nodes)

        graph_src_ids, graph_dst_ids = graph_edges_fc(n_nodes, device)
        
        src_ids = graph_src_ids
        dst_ids = graph_dst_ids

        edge_inputs = dist_mat_scaled[0][graph_src_ids, graph_dst_ids]
        
        for i in range(1, batch_size):
            src_ids = torch.cat((src_ids, graph_src_ids+i*n_nodes), dim=0)
            dst_ids = torch.cat((dst_ids, graph_dst_ids+i*n_nodes), dim=0)
            edge_inputs = torch.cat((edge_inputs, dist_mat_scaled[i][graph_src_ids, graph_dst_ids]), dim=0)
            
        return node_inputs, edge_inputs.reshape(-1, 1), targets.flatten(), dist_mat, target_lengths, src_ids, dst_ids

    trainset, testset = random_split(dataset, 
                                     [train_size, test_size], 
                                     generator=torch.Generator().manual_seed(SEED))
    
    trainloader = torch.utils.data.DataLoader(trainset, 
                                              batch_size=batch_size, 
                                              collate_fn=collate,
                                              shuffle=False)
    
    testloader = torch.utils.data.DataLoader(testset, 
                                             batch_size=test_batch_size, 
                                             collate_fn=collate,
                                             shuffle=False)
    
    return trainloader, testloader, class_weights


def save_model(model, filename, confirm=True):
    """Saves a trained VRPNet model"""
    if confirm:
        try:
            save = input('Do you want to save the model (type yes to confirm)? ').lower()
            if save != 'yes':
                print('Model not saved.')
                return
        except:
            raise Exception('The notebook should be run or validated with skip_training=True.')

    torch.save(model.state_dict(), filename)
    print('Model saved to %s.' % (filename))


def load_model(model, filename, device):
    """Loads a trained VRPNet model"""
    model.load_state_dict(torch.load(filename, map_location=lambda storage, loc: storage))
    print('Model loaded from %s.' % filename)
    model.to(device)
    model.eval()

def save_dict(obj, name):
    """Saves a dictionary to a pickle file. Used to store different diagnostic data from training."""
    path = name + '.pkl'
    with open(path, 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def load_dict(name ):
    """Loads a dictionary from a pickle file."""
    path = name + '.pkl'
    if os.path.isfile(path):
        with open(path, 'rb') as f:
            return pickle.load(f)
    else:
        obj = {}
        save_dict(obj, name)
        return obj